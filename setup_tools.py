"""One-command setup for a new machine: Python packages, the MediaPipe hand model and Blender.

    python v5/setup_tools.py              # install what is missing (asks before the big Blender download)
    python v5/setup_tools.py --dry-run    # show what it would do, change nothing
    python v5/setup_tools.py --yes        # no questions (also downloads Blender if none is found)

What it does, in order (each step is skipped when it is already done, so it is safe to run again):

  1. Python packages from requirements.txt (mediapipe, opencv-python, numpy) with `pip install -r`.
  2. The MediaPipe HandLandmarker model `hand_landmarker.task` -> tools/models/ (about 7.5 MB, from Google's
     official model storage). A copy already sitting in the project root is reused instead of downloaded.
  3. Blender 5.1.2 (portable build from download.blender.org, about 395 MB) -> tools/blender-5.1.2-*/.
     Skipped when Blender is already installed (PATH, the usual install folders or tools/).
     Blender is only needed to open the .blend files, run the receiver and run the headless tests.

Safety: downloads use HTTPS from the two official hosts below only, each file is checked against a pinned
SHA-256 before it is used, archives are unpacked with path checks, a file is written as `*.part` and renamed
only when complete, downloads retry with backoff, and nothing outside the tools folder is changed except the
pip install in step 1 (use --skip-python to leave your Python alone, or run it inside a virtual environment).
A summary of what is installed is written to tools/tools.json.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOOLS = ROOT / "tools"
REQUIREMENTS = ROOT / "requirements.txt"
USER_AGENT = "hand-blend-setup/1.0"
CHUNK = 1024 * 256

# ---- pinned third-party downloads (sizes and hashes checked against the official hosts) -----------------------
MODEL = {
    "name": "hand_landmarker.task",
    "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "sha256": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    "bytes": 7819105,
}
BLENDER_VERSION = "5.1.2"
BLENDER_BASE = "https://download.blender.org/release/Blender5.1/"
BLENDER_BUILDS = {  # (sys.platform prefix, machine) -> archive, sha256 (from blender-5.1.2.sha256 on download.blender.org)
    "windows": {"file": f"blender-{BLENDER_VERSION}-windows-x64.zip", "bytes": 414378949,
                "sha256": "345bedea7b0acf7cc9666423d8553f9129622aea34ded65c23e8cb70f83f14ff"},
    "linux": {"file": f"blender-{BLENDER_VERSION}-linux-x64.tar.xz", "bytes": 396237880,
              "sha256": "aaccb355f50183979b698bcce7467103a76261b5fa59f4972295842662a285fb"},
}
ALLOWED_HOSTS = ("storage.googleapis.com", "download.blender.org")
REQUIRED_MODULES = {"cv2": "opencv-python", "mediapipe": "mediapipe", "numpy": "numpy"}


_ALLOW_ANY_URL = False  # only test_setup_tools.py turns this on (file: URLs, no network)


class SetupError(Exception):
    pass


def say(msg=""):
    print(msg, flush=True)


# ---- download with checks ----------------------------------------------------------------------------------------
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest, sha256, expected_bytes=None, retries=3, timeout=30, allow_any_url=False):
    """Download url to dest (atomic: dest only appears when complete and the SHA-256 matches).
    Only https URLs on ALLOWED_HOSTS are accepted unless allow_any_url (used by the tests with file: URLs)."""
    if not (allow_any_url or _ALLOW_ANY_URL):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
            raise SetupError(f"refusing to download from {url}: only https on {', '.join(ALLOWED_HOSTS)}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            h = hashlib.sha256()
            done = 0
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as out:
                total = int(resp.headers.get("Content-Length") or expected_bytes or 0)
                shown = -1
                for block in iter(lambda: resp.read(CHUNK), b""):
                    out.write(block)
                    h.update(block)
                    done += len(block)
                    if total and sys.stdout.isatty():
                        pct = int(done * 100 / total)
                        if pct != shown and pct % 5 == 0:
                            shown = pct
                            print(f"\r    {dest.name}: {pct:3d}%  ({done / 1e6:.0f} of {total / 1e6:.0f} MB)", end="", flush=True)
            if sys.stdout.isatty():
                print()
            if expected_bytes and done != expected_bytes:
                raise SetupError(f"size {done} bytes, expected {expected_bytes}")
            if h.hexdigest() != sha256:
                raise SetupError(f"SHA-256 mismatch for {dest.name}: got {h.hexdigest()}, expected {sha256}")
            os.replace(part, dest)
            return dest
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, SetupError) as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            if isinstance(exc, SetupError) and "mismatch" in str(exc):
                break  # retrying a bad file will not fix it
            if attempt < retries:
                wait = 2 ** attempt
                say(f"    attempt {attempt} failed ({exc}); retrying in {wait}s")
                time.sleep(wait)
    raise SetupError(f"could not download {url}: {last_error}")


def safe_extract(archive, target):
    """Unpack a .zip or .tar.xz into target, refusing any member that would land outside it."""
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)

    def inside(name):
        dest = (target / name).resolve()
        return dest == target or target in dest.parents

    if str(archive).endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            bad = [n for n in z.namelist() if not inside(n)]
            if bad:
                raise SetupError(f"unsafe path in archive: {bad[0]}")
            z.extractall(target)
    else:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if not inside(m.name):
                    raise SetupError(f"unsafe path in archive: {m.name}")
                if m.islnk() and not inside(m.linkname):
                    raise SetupError(f"unsafe link in archive: {m.name}")
            if hasattr(tarfile, "data_filter"):
                t.extractall(target, filter="data")  # also blocks absolute links, devices, odd permissions
            else:
                t.extractall(target)


# ---- step 1: python packages ------------------------------------------------------------------------------------
def missing_modules():
    return [pkg for mod, pkg in REQUIRED_MODULES.items() if importlib.util.find_spec(mod) is None]


def step_python(args):
    say("[1/3] Python packages")
    if sys.version_info < (3, 9):
        raise SetupError(f"Python {sys.version.split()[0]} is too old: use Python 3.9 or newer (tested with 3.13)")
    missing = missing_modules()
    if not missing:
        say(f"    already installed: {', '.join(REQUIRED_MODULES.values())}")
        return {"status": "present", "python": sys.version.split()[0]}
    if not REQUIREMENTS.exists():
        raise SetupError(f"{REQUIREMENTS} not found")
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
    say(f"    missing: {', '.join(missing)}")
    say("    running: " + " ".join(cmd))
    if args.dry_run:
        return {"status": "would install"}
    if sys.prefix == sys.base_prefix and not args.yes and sys.stdin.isatty():
        say("    note: this is your system Python, not a virtual environment.")
        if input("    install the packages into it? [y/N] ").strip().lower() not in ("y", "yes"):
            say("    skipped; create a venv (python -m venv .venv) or rerun with --yes")
            return {"status": "skipped by user"}
    if subprocess.call(cmd) != 0:
        raise SetupError("pip install failed (see the messages above)")
    still = missing_modules()
    if still:
        raise SetupError(f"pip finished but these cannot be imported: {', '.join(still)}")
    return {"status": "installed", "python": sys.version.split()[0]}


# ---- step 2: the MediaPipe model ----------------------------------------------------------------------------------
def step_model(args, tools):
    say("[2/3] MediaPipe hand model")
    dest = tools / "models" / MODEL["name"]
    if dest.exists() and sha256_of(dest) == MODEL["sha256"]:
        say(f"    already there: {dest}")
        return {"status": "present", "path": str(dest), "sha256": MODEL["sha256"]}
    legacy = ROOT / MODEL["name"]
    if legacy.exists() and sha256_of(legacy) == MODEL["sha256"]:
        say(f"    reusing the copy in the project root: {legacy} -> {dest}")
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy, dest)
        return {"status": "copied", "path": str(dest), "sha256": MODEL["sha256"]}
    say(f"    downloading {MODEL['url']} ({MODEL['bytes'] / 1e6:.1f} MB)")
    if args.dry_run:
        return {"status": "would download"}
    download(MODEL["url"], dest, MODEL["sha256"], MODEL["bytes"])
    say(f"    saved and verified: {dest}")
    return {"status": "downloaded", "path": str(dest), "sha256": MODEL["sha256"], "source": MODEL["url"]}


# ---- step 3: Blender -----------------------------------------------------------------------------------------------
def blender_name():
    return "blender.exe" if sys.platform.startswith("win") else "blender"


def blender_candidates(tools):
    found = []
    on_path = shutil.which("blender")
    if on_path:
        found.append(Path(on_path))
    found += sorted(tools.glob(f"blender-*/**/{blender_name()}"))
    if sys.platform.startswith("win"):
        for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
            if base:
                found += sorted(Path(base, "Blender Foundation").glob(f"Blender */{blender_name()}"))
    elif sys.platform == "darwin":
        found.append(Path("/Applications/Blender.app/Contents/MacOS/Blender"))
    else:
        found += [Path("/usr/bin/blender"), Path("/usr/local/bin/blender"), Path("/snap/bin/blender")]
    seen, out = set(), []
    for p in found:
        if p.is_file() and str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


def blender_version(exe):
    try:
        out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"Blender (\d+\.\d+\.\d+)", out)
    return m.group(1) if m else None


def blender_build():
    machine = platform.machine().lower()
    if sys.platform.startswith("win") and machine in ("amd64", "x86_64"):
        return BLENDER_BUILDS["windows"]
    if sys.platform.startswith("linux") and machine in ("x86_64", "amd64"):
        return BLENDER_BUILDS["linux"]
    return None


def free_gb(path):
    probe = Path(path)
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def step_blender(args, tools):
    say("[3/3] Blender")
    for exe in blender_candidates(tools):
        version = blender_version(exe)
        note = ""
        if version and not version.startswith("5.1."):
            note = f"  (the project is tested with 5.1.x; this is {version})"
        say(f"    found: {exe}  version {version or 'unknown'}{note}")
        return {"status": "present", "path": str(exe), "version": version}
    build = blender_build()
    if build is None:
        say("    no Blender found, and there is no automatic download for this system.")
        say("    install Blender 5.1 from https://www.blender.org/download/ (the project is tested with 5.1.2).")
        return {"status": "manual"}
    say(f"    no Blender found. Download Blender {BLENDER_VERSION} portable ({build['bytes'] / 1e6:.0f} MB) from download.blender.org?")
    if args.dry_run:
        say(f"    would download {BLENDER_BASE + build['file']} into {tools}")
        return {"status": "would download"}
    if not args.yes:
        if not sys.stdin.isatty() or input("    download it now? [y/N] ").strip().lower() not in ("y", "yes"):
            say("    skipped (rerun with --yes to download it, or install Blender yourself)")
            return {"status": "skipped"}
    need_gb = (build["bytes"] + 1.2e9) / 1e9  # the archive plus the unpacked build (about 1 GB, measured on 5.1.2)
    if free_gb(tools) < need_gb:
        raise SetupError(f"need about {need_gb:.1f} GB free in {tools} (archive plus unpacked Blender), "
                         f"have {free_gb(tools):.1f} GB")
    archive = tools / "downloads" / build["file"]
    if not (archive.exists() and sha256_of(archive) == build["sha256"]):
        download(BLENDER_BASE + build["file"], archive, build["sha256"], build["bytes"], timeout=60)
    say("    verified; unpacking (this takes a minute)")
    with tempfile.TemporaryDirectory(dir=tools, prefix=".unpack-") as tmp:
        safe_extract(archive, tmp)
        top = [p for p in Path(tmp).iterdir()]
        source = top[0] if len(top) == 1 and top[0].is_dir() else Path(tmp)
        final = tools / source.name if source != Path(tmp) else tools / f"blender-{BLENDER_VERSION}"
        if final.exists():
            raise SetupError(f"{final} already exists but holds no {blender_name()}; delete it and rerun")
        shutil.move(str(source), str(final))
    exes = sorted(final.glob(blender_name()))
    if not exes:
        raise SetupError(f"unpacked, but {blender_name()} is not in {final}")
    if not args.keep_archive:
        archive.unlink(missing_ok=True)
    version = blender_version(exes[0])
    say(f"    installed: {exes[0]}  version {version}")
    return {"status": "downloaded", "path": str(exes[0]), "version": version,
            "source": BLENDER_BASE + build["file"], "sha256": build["sha256"]}


# ---- main -----------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Install the third-party tools this project needs.")
    ap.add_argument("--tools-dir", default=str(DEFAULT_TOOLS), help="where downloads go (default: <project>/tools)")
    ap.add_argument("--dry-run", action="store_true", help="show what would be done, change nothing")
    ap.add_argument("--yes", "-y", action="store_true", help="answer yes to every question")
    ap.add_argument("--skip-python", action="store_true", help="do not pip install requirements.txt")
    ap.add_argument("--skip-model", action="store_true", help="do not fetch the MediaPipe model")
    ap.add_argument("--skip-blender", action="store_true", help="do not look for or download Blender")
    ap.add_argument("--keep-archive", action="store_true", help="keep the downloaded Blender archive")
    args = ap.parse_args(argv)
    tools = Path(args.tools_dir).resolve()
    if not args.dry_run:
        tools.mkdir(parents=True, exist_ok=True)
    say(f"project: {ROOT}\ntools:   {tools}{'   (dry run: nothing is changed)' if args.dry_run else ''}\n")

    report, failed = {}, False
    steps = (("python", args.skip_python, lambda: step_python(args)),
             ("model", args.skip_model, lambda: step_model(args, tools)),
             ("blender", args.skip_blender, lambda: step_blender(args, tools)))
    for name, skip, run in steps:
        if skip:
            say(f"[skipped] {name}")
            continue
        try:
            report[name] = run()
        except (SetupError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
            say(f"    FAILED: {exc}")
            report[name] = {"status": "failed", "error": str(exc)}
            failed = True
        say()
    if not args.dry_run:
        (tools / "tools.json").write_text(json.dumps(
            {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": report}, indent=2), encoding="utf-8")

    blender = report.get("blender", {}).get("path")
    say("Summary: " + ", ".join(f"{k}: {v['status']}" for k, v in report.items()))
    if not failed and not args.dry_run:
        say("\nNext:")
        if blender:
            say(f'  open v5/untitled_two_hands_swapped.blend with "{blender}", then run the text block v5_run_receiver')
        say("  python v5/sender.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
