"""Offline tests for setup_tools.py: no network, no pip, nothing outside a temporary folder.

    python v5/test_setup_tools.py
"""
import contextlib
import hashlib
import io
import os
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup_tools as st  # noqa: E402

REAL_MODEL, REAL_ROOT = dict(st.MODEL), st.ROOT  # the tests below swap in fake ones

failures = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  {detail}"))
    if not ok:
        failures.append(name)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def raises(fn, *a, **k):
    try:
        quiet(fn, *a, **k)
    except st.SetupError as exc:
        return str(exc)
    return None


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # ---- download -------------------------------------------------------------------------------------------------
    payload = b"hello hand" * 1000
    src = tmp / "src.bin"
    src.write_bytes(payload)
    url = src.as_uri()
    st._ALLOW_ANY_URL = True
    out = quiet(st.download, url, tmp / "dl" / "a.bin", sha(payload), len(payload))
    check("download_verifies_and_writes_the_file", out.read_bytes() == payload)
    check("download_leaves_no_part_file", not list((tmp / "dl").glob("*.part")))
    msg = raises(st.download, url, tmp / "dl" / "b.bin", "0" * 64, retries=3)
    check("wrong_hash_is_rejected_without_retrying_and_leaves_nothing",
          msg is not None and "mismatch" in msg and not (tmp / "dl" / "b.bin").exists()
          and not list((tmp / "dl").glob("*.part")), str(msg))
    msg = raises(st.download, url, tmp / "dl" / "c.bin", sha(payload), len(payload) + 1)
    check("wrong_size_is_rejected", msg is not None and not (tmp / "dl" / "c.bin").exists(), str(msg))
    st._ALLOW_ANY_URL = False
    check("file_url_refused_by_default", raises(st.download, url, tmp / "dl" / "d.bin", sha(payload)) is not None)
    check("http_refused", raises(st.download, "http://download.blender.org/x", tmp / "e", "0" * 64) is not None)
    check("unknown_host_refused", raises(st.download, "https://example.com/x", tmp / "e", "0" * 64) is not None)

    # ---- safe extraction ------------------------------------------------------------------------------------------
    good = tmp / "good.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("blender-x/blender.exe", "x")
        z.writestr("blender-x/readme.txt", "y")
    quiet(st.safe_extract, good, tmp / "out_good")
    check("zip_extracts_normally", (tmp / "out_good" / "blender-x" / "blender.exe").exists())
    evil = tmp / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../escaped.txt", "bad")
    check("zip_slip_is_refused", raises(st.safe_extract, evil, tmp / "out_evil") is not None
          and not (tmp / "escaped.txt").exists())
    tgood = tmp / "good.tar.xz"
    with tarfile.open(tgood, "w:xz") as t:
        info = tarfile.TarInfo("blender-x/blender")
        info.size = 1
        t.addfile(info, io.BytesIO(b"x"))
    quiet(st.safe_extract, tgood, tmp / "out_tar")
    check("tar_extracts_normally", (tmp / "out_tar" / "blender-x" / "blender").exists())
    tevil = tmp / "evil.tar.xz"
    with tarfile.open(tevil, "w:xz") as t:
        info = tarfile.TarInfo("../escaped_tar.txt")
        info.size = 1
        t.addfile(info, io.BytesIO(b"x"))
    check("tar_path_escape_is_refused", raises(st.safe_extract, tevil, tmp / "out_tar2") is not None
          and not (tmp / "escaped_tar.txt").exists())

    # ---- the model step -------------------------------------------------------------------------------------------
    fake_root = tmp / "project"
    fake_root.mkdir()
    model_bytes = b"model" * 100
    st.ROOT = fake_root
    st.MODEL = dict(st.MODEL, sha256=sha(model_bytes), bytes=len(model_bytes), url=(tmp / "model.bin").as_uri())
    (tmp / "model.bin").write_bytes(model_bytes)

    class Args:
        dry_run = False
        yes = True
        keep_archive = False

    tools = tmp / "tools"
    st._ALLOW_ANY_URL = True
    r = quiet(st.step_model, Args, tools)
    check("model_missing_everywhere_is_downloaded_and_verified",
          r["status"] == "downloaded" and (tools / "models" / st.MODEL["name"]).read_bytes() == model_bytes, str(r))
    r = quiet(st.step_model, Args, tools)
    check("model_already_there_is_left_alone", r["status"] == "present", str(r))

    (tools / "models" / st.MODEL["name"]).write_bytes(b"corrupt")
    r = quiet(st.step_model, Args, tools)
    check("corrupt_model_is_replaced", r["status"] == "downloaded"
          and (tools / "models" / st.MODEL["name"]).read_bytes() == model_bytes, str(r))

    tools2 = tmp / "tools2"
    (fake_root / st.MODEL["name"]).write_bytes(model_bytes)
    st._ALLOW_ANY_URL = False  # a copy in the project root must be reused without any download
    r = quiet(st.step_model, Args, tools2)
    check("model_in_project_root_is_copied_not_downloaded",
          r["status"] == "copied" and (tools2 / "models" / st.MODEL["name"]).read_bytes() == model_bytes, str(r))

    class DryArgs(Args):
        dry_run = True
    tools3 = tmp / "tools3"
    (fake_root / st.MODEL["name"]).unlink()
    r = quiet(st.step_model, DryArgs, tools3)
    check("dry_run_downloads_nothing", r["status"] == "would download" and not tools3.exists(), str(r))

    # ---- the Blender step -----------------------------------------------------------------------------------------
    blender_zip = tmp / "blender-test.zip"
    with zipfile.ZipFile(blender_zip, "w") as z:
        z.writestr(f"blender-test/{st.blender_name()}", "not a real blender")
    bz = blender_zip.read_bytes()
    st.blender_build = lambda: {"file": "blender-test.zip", "bytes": len(bz), "sha256": sha(bz)}
    st.BLENDER_BASE = tmp.as_uri() + "/"
    st.blender_candidates = lambda tools: []
    st._ALLOW_ANY_URL = True
    real_free = st.free_gb
    st.free_gb = lambda path: 0.1
    tools_full = tmp / "tools_full"
    tools_full.mkdir()
    msg = raises(st.step_blender, Args, tools_full)
    check("blender_download_refused_when_the_disk_is_too_full", msg is not None and "GB free" in msg
          and not (tools_full / "downloads").exists(), str(msg))
    st.free_gb = lambda path: 100.0
    tools4 = tmp / "tools4"
    tools4.mkdir()
    r = quiet(st.step_blender, Args, tools4)
    check("blender_is_downloaded_unpacked_and_located",
          r["status"] == "downloaded" and Path(r["path"]).name == st.blender_name() and Path(r["path"]).is_file(), str(r))
    check("blender_archive_is_removed_after_install", not (tools4 / "downloads" / "blender-test.zip").exists())
    check("no_unpack_leftovers", not list(tools4.glob(".unpack-*")))
    r = quiet(st.step_blender, DryArgs, tmp / "tools5")
    check("blender_dry_run_changes_nothing", r["status"] == "would download" and not (tmp / "tools5").exists(), str(r))

    st.blender_build = lambda: {"file": "blender-test.zip", "bytes": len(bz), "sha256": "0" * 64}
    tools6 = tmp / "tools6"
    tools6.mkdir()
    try:
        quiet(st.step_blender, Args, tools6)
        bad_hash_blocked = False
    except st.SetupError:
        bad_hash_blocked = True
    check("blender_with_wrong_checksum_is_not_installed",
          bad_hash_blocked and not list(tools6.glob("blender-*")), "")

    found = tmp / "installed" / st.blender_name()
    found.parent.mkdir()
    found.write_text("x")
    st.blender_candidates = lambda tools: [found]
    r = quiet(st.step_blender, Args, tmp / "tools7")
    check("existing_blender_is_used_and_nothing_downloaded", r["status"] == "present" and r["path"] == str(found), str(r))

    # ---- whole script ---------------------------------------------------------------------------------------------
    st._ALLOW_ANY_URL = False
    target = tmp / "never_created"
    code = quiet(st.main, ["--dry-run", "--tools-dir", str(target), "--skip-python", "--skip-blender"])
    check("main_dry_run_succeeds_and_creates_nothing", code == 0 and not target.exists(), f"code {code}")

    # the pinned values are the real ones: a typo here would break every new user
    check("pinned_sha256_values_look_valid",
          all(len(v) == 64 and set(v) <= set("0123456789abcdef")
              for v in [REAL_MODEL["sha256"]] + [b["sha256"] for b in st.BLENDER_BUILDS.values()]))
    real_copy = REAL_ROOT / REAL_MODEL["name"]
    if real_copy.exists():  # the model you already use must match the pin, or every new user's download would fail
        check("pinned_model_hash_matches_the_model_in_the_project", st.sha256_of(real_copy) == REAL_MODEL["sha256"])

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all setup tool tests passed")
