# v5: hand tracking to Blender

This README is the detailed reference (one section per feature, with measurements). For the whole
picture in one read (how the parts fit, setup, the order things were built in and why, a settings
cheat sheet, troubleshooting, ideas for next steps) start with [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

v5 is the maintained pipeline. The older scripts in the parent folder (`hand_full_control*.py`,
`v1.py`, `src/`, `shared_memory/`) are not used by it and were left untouched.

## Setup on a new machine

```
python v5/setup_tools.py              # installs what is missing; asks before the 395 MB Blender download
python v5/setup_tools.py --dry-run    # shows what it would do and changes nothing
python v5/setup_tools.py --yes        # no questions
```

It does three things, each skipped if already done (safe to rerun): `pip install -r requirements.txt`
(mediapipe, opencv-python, numpy); fetches the MediaPipe `hand_landmarker.task` model into `tools/models/`
(reusing a copy in the project root if you have one); and, only if no Blender is found (PATH, the usual
install folders, `tools/`), downloads the portable Blender 5.1.2 from download.blender.org into `tools/`
(Windows and Linux; on macOS install Blender yourself). Downloads are HTTPS from `storage.googleapis.com` and
`download.blender.org` only, checked against pinned SHA-256 values, written as `.part` files and renamed
when complete, with retries. Blender needs about 1.4 GB free while it installs (the archive, then the unpacked
build of about 1 GB). Flags: `--skip-python`, `--skip-model`, `--skip-blender`, `--tools-dir`, `--keep-archive`.
`tools/tools.json` records what was installed. `test_setup_tools.py` tests all of this offline.

## Run

1. Open **`v5/untitled_rig_fitted.blend`** (one hand) or **`v5/untitled_two_hands_swapped.blend`** (two hands) in Blender: the rig with a skeleton fitted to the mesh (see "Skeleton fitted to the mesh"
   below). `untitled_rig_fixed_v2.blend` is the same rig with the original skeleton and weights, and the
   `untitled_rig_fixed.blend` is the oldest fixed copy (some tests still read it); the other older
   two-hand and `_v3` files were deleted as superseded. In the Scripting tab,
   open the text block `v5_run_receiver` and click Run Script. It runs `blender_receiver.py` from
   next to the .blend, so keep both in the same folder.
2. `python v5/sender.py` (ESC quits). Useful flags: `--hands auto|1|2`, `--dry-run`, `--record FILE`, `--hfov DEG`,
   `--min-cutoff`, `--beta`, `--d-cutoff`, `--max-speed`, `--no-preview`.
3. No webcam: `python v5/replay.py FILE [--loop] [--speed 1.0]` replays a recording.

Running the receiver also adds a **Hand** tab to the 3D viewport sidebar (press N): Start / Stop, a
live status (tracking or lost, frames per second, latency, which way the rig is mirrored) and the
mirror choice (Automatic, Mirrored, Not mirrored). It comes from `blender_panel.py`, which the
receiver loads from next to itself; no change to the .blend is needed, and the receiver works
without the panel.

Blender prints what it decided, for example `[hand] rig mirrored (finger bends)`, plus `max_backlog`,
latency and tick time every 5 s. The sender prints `fps`, rejected frames, `receiver=acked` and
per-stage ms.

## Skeleton fitted to the mesh

The skeleton that came with the mesh did not sit in it. `fit_skeleton.py` builds a skeleton from the mesh:
it renders the mesh from 13 directions around the palm, MediaPipe (the model the tracker uses) finds the
21 hand landmarks in the 9 pictures where it sees a hand, and the landmarks are triangulated into 3D
(`hand_detect.py` runs MediaPipe in a separate process, since Blender's Python has none). Each finger
joint is then moved to the middle of its own finger's cross-section, the armature origin is moved to the
wrist (the receiver pins the WRIST empty there), and the bones, IK constraints and empties are kept,
with new positions. Rolls, IK limits, empties and skin weights are redone for the new bones, and the
receiver now scales from the rig's own wrist-to-middle-knuckle bone (`Rig.span`), not a fixed constant.

```
blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/fit_skeleton.py -- v5 NEW.blend [python=python] [overwrite=1]
```

`python=` is the interpreter that has MediaPipe (the one you run `sender.py` with). It writes a new file and
never replaces one. What the fit changed (`test_fit_headless.py`, on the original skeleton):

| | original skeleton | fitted |
|---|---|---|
| distance of its joints from where the mesh has them | 0.034 on average, 0.149 at most (a finger is 0.24 long and 0.04 thick) | 0 |
| wrist | 0.149 below the wrist, outside the mesh | at the wrist crease, inside the mesh |
| finger bone shares (proximal / middle / distal) | 0.40-0.48 / 0.24-0.32 / 0.26-0.33 | 0.45-0.47 / 0.24-0.31 / 0.25-0.31 |

The bone *lengths* of the original skeleton were already about right; its joints were not where the
mesh bends (the index knuckle and middle joint were about one finger-thickness off). Bending skin about a
pivot that far from the real joint is what stretched and folded it in strong poses. Rendered in a fist,
a pointing hand, a pinch and a spread hand (`pose_gallery.py`), the thumb is no longer a stretched
ribbon, the fist is a compact fist, and the wrist is clean. Measured on your real recording the fitted
skeleton keeps the mesh volume closer to rest (lowest 0.995 against 0.964) but has not reduced mesh
edge stretching (about 3,700 edges stretched by more than 0.01 over 12 poses at power 20, like the old
skeleton; more at softer weights): linear-blend skinning stretches the outside of a bent joint
whatever the skeleton, and the mesh is very dense, so these counts are high for any rig.

## The hand mesh (skinning)

The files above carry the mesh re-skinned by `skin_mesh.py`. The mesh in the original file had broken
weights, which made the skeleton look right and the mesh look wrong:

| Problem in the original weights | Effect |
|---|---|
| 5% of the vertices had no bone | those points never followed the fingers |
| the ring fingertip's weights were in a group named `ring_dip_to_ring_tip"` (stray quote) and a leftover `Bone.017` | the ring fingertip bone moved nothing: the tip was dragged out into a thin ribbon |
| `wrist_to_pinky_mcp` owned most of the palm, `wrist_to_index_mcp` and the thumb knuckle bone owned nothing | the palm wobbled with the pinky |
| two Armature modifiers, one with no object | harmless, removed |

Measured on the real recording (12 poses, `test_skin_headless.py`): the ring fingertip's skin ended
0.324 armature units away from its bone, now 0.009; the worst edge growth fell from 0.326 to 0.040
(a tear or ribbon is a large growth); the lowest mesh volume against rest is 0.964 (Preserve Volume
on). Re-skin any file yourself (writes a new file, never replaces one):

```
blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/skin_mesh.py -- v5 NEW.blend [power=10] [volume=1|0]
```

Each vertex is weighted to the bones nearest to it, `(nearest distance / this distance) ** power`,
the four strongest, normalised. `power` 3 gives softer joints, 10 crisper ones. The five
wrist-to-knuckle palm bones share the palm softly between them. The mesh is hidden in your file;
show it with the eye icon (or H / Alt+H). With `--hands 2` a copy of the mesh is skinned for each hand.

**Rigid palm** (`RIGID_PALM` in `blender_receiver.py`, on by default): the five wrist-to-knuckle
bones and the wrist-to-thumb-base bone stay in the palm's rest shape; only the fingers follow the
live landmarks. A real palm is one stiff piece, and letting these bones follow landmark noise tore
slits in the skin across the palm. The skeleton's palm no longer copies the live palm's proportions
(the knuckle bend error against your raw fingers went from 1.8 to 2.6 degrees).

## Finger collision (fingers do not pass through each other)

On by default (`COLLISION` in `blender_receiver.py`). The rig's proportions differ from your hand's
and the palm is rigid, so a retargeted hand often has fingers lying inside each other (the thumb
through the index, the middle through the ring, fingertips sunk into the palm). Each finger segment
is now a capsule as thick as the mesh around its bone, and after retargeting:

- neighbouring fingers (index-middle, middle-ring, ring-pinky, thumb-index) may not overlap more than
  they do in the rest pose (the rest pose is the contact the mesh was made with, so it is never
  pushed); `COLLISION_MARGIN` 0.7 lets soft fingers press together by a third of their thickness;
- fingertips may not sink into the palm (`PALM_COLLISION`);
- overlapping fingers are pushed apart (the knuckles stay put, the bones keep their lengths, four
  passes) and the joint limits are applied again to any finger that was moved.

The thickness comes from `skin_mesh.py`, which stores it on every bone (`v5_radius`, the 75th
percentile of the skin's distance to the bone); without it a fraction of the segment length is
used. It costs about 0.2 ms per frame. Measured on your recording and the skinned mesh
(`test_collision_headless.py`):

| | collision off | collision on |
|---|---|---|
| segment pairs overlapping | 144 | 19 |
| fingertips inside the palm | 12 | 2 |
| mesh vertices closer than 0.35 radius to a neighbouring finger's bone (12 poses) | 37,732 | 6,158 |
| of those, ring and middle finger | 15,243 | 0 |
| how far collision moves a bone direction | | 1.3 degrees on average, 2.9 at the 90th percentile |

The thumb against the index finger is the weakest pair (thumb skin still reaches into the index in
some poses: about 1,800 and 800 vertices). Trade-off: a higher radius percentile (`skin_mesh.py
RADIUS_PERCENTILE`, 90 removes 94% of the overlap) or a margin nearer 1 keeps fingers further apart
and moves the pose more (about 6 degrees at the 90th percentile at 90). Because collision moves
fingers on purpose, `analyze_recording_headless.py` runs with it off (`collision=1` turns it on) and
scores how well the rig follows your fingers.

## Two hands

`v5/untitled_two_hands_swapped.blend` is the fitted rig twice: the first hand (`Armature`, the empties named
as the landmarks) at the origin is driven by your **left hand**, and a copy of the armature, its 21 landmark
empties and the skinned mesh (`Armature.001`, `WRIST.001`, ...) 0.8 units to its +X side is driven by your
**right hand**, so in Blender's front view your right hand is on the right. (Build another layout with
`build_two_hands.py`, e.g. `spacing=-0.8` for the copy on the -X side.) The receiver mirrors a rig to fit, from the finger bends. Every
IK constraint points at its own hand's empties, and the mesh copy is shown or hidden like the original
(it has about 600,000 vertices). Open it, run `v5_run_receiver`, then simply:

```
python v5/sender.py
```

The sender asks Blender how many hand rigs it has (Blender's reply says so) and tracks that many
hands: it starts with one, and when the reply says two it rebuilds MediaPipe for two hands and
prints `Blender has 2 hand rigs: now tracking 2 hands`. `--hands 2` forces two, `--hands 1` forces
one; a mismatch with the file is explained on the console. The preview window shows
`hands seen N, tracking M` and labels each tracked hand `left hand` / `right hand` (dots: green, orange).
If `seen` is 2 but `tracking` is 1, the sender has not heard from Blender yet. If `seen` stays 1
with two hands in view, MediaPipe is not detecting the second hand: show it clearly, or lower
`--detect-confidence` (default 0.5 with two hands, 0.7 with one).

**Your right hand drives `Armature.001` (the rig on the right) and your left hand `Armature` (the one on the
left)** (`slots.py`; if you ever want them the other way round, swap `LEFT, RIGHT = 0, 1` there and
`HAND_NAMES` in `blender_receiver.py`; a test checks the two agree). MediaPipe labels
every hand "Right" or "Left" (the sender flips the picture like a mirror, which is what MediaPipe
expects, so the label is your real hand), and a new hand takes the rig its label names; with no usable
label the side of the picture decides (right hand on the right). After that the hand keeps its rig by
where it is: it follows the nearest remembered position, allowing for its motion, so a flickering label
cannot swap two hands, and only a label that has said the opposite for 8 frames in a row moves a hand
to the other rig. A rig remembers its hand for about half a second if it drops out. Each hand has its own outlier gate, filter,
scale, mirror decision and relax-to-rest, so one hand leaving only relaxes its own rig. The panel
shows a block and a mirror choice for each hand. `--hands 1` (the default) and a one-hand .blend
behave exactly as before: the receiver uses a second rig only if `Armature.001` exists.

Make your own two-hand file from any fixed copy (writes a new file, never replaces one):

```
blender -b --disable-autoexec v5/untitled_rig_fitted.blend --python v5/build_two_hands.py -- v5 NEW.blend [spacing=0.8]
```

Recordings keep a `slot` per packet. `analyze_recording_headless.py ... slot=1` scores hand 2, and
`bake_recording.py` bakes every hand a take has (`hands=1` or `hands=2` picks one).

## No mirror, and which way the hand faces (`UNMIRROR_HAND`, `HAND_YAW_DEG`)

Two steps in `blender_receiver.py`, both applied to every hand before anything else:

1. **No mirror** (`UNMIRROR_HAND = True`). The sender flips the camera picture before MediaPipe sees it (MediaPipe's
   left/right labels expect that), so every landmark arrives as a mirror image of the hand: left and right are
   right, but depth is reversed and a right hand arrives with a left hand's shape. The receiver reflects the depth
   (Blender's Y) back, for the hand and for its translation, so Blender shows the hand you really hold up: a right
   hand stays a right hand, and you and Blender's front view both look along +Y with X to your right and Z up.
2. **Turned to face the front view** (`HAND_YAW_DEG = 180`). Every hand is then turned 180 degrees about **its own
   long axis** (from the wrist through the middle knuckle; with the fingers up that is the vertical Z axis), so palm
   and back swap sides: hold your palm toward the webcam and Blender's front view sees the palm. Only the
   orientation turns; where the hand sits does not.

Why about its own axis: a turn about the fixed world Z axis sends every sideways tilt the other way (tilt your
hand to the right and Blender's tilts left). A turn about the hand's own axis goes with the hand, so every movement
keeps its direction: tilt right and it tilts right, push forward and it goes forward, and the position of the hand
(`TRANSLATE_HAND`: right = +X, toward the webcam = +Y, up = +Z) is unaffected by the turn.
`HAND_TURN_ABOUT = "world"` gives the world-axis turn; `HAND_YAW_DEG = 0` turns nothing (palm toward the webcam
then faces +Y, away from the front view); `UNMIRROR_HAND = False` gives the old mirror view.

One thing a hand that shows its palm to you cannot avoid: it curls toward you. With the palm facing the front view
the fingers curl toward the viewer, while your own fingers curl away from you toward the webcam. Use `HAND_YAW_DEG = 0`
if you would rather have your own point of view, where the curl matches.

`test_features_headless.py` builds a right and a left hand the way the sender delivers them (palm to the camera,
fingers up) and checks on the real rig: with the turn the palm normal faces the front view (-Y), the right hand's
thumb is on the right of the front view and the left hand's on the left, the right hand needs no mirroring and the
left does, and tilting the real hand right / left / forward / back moves the fingers the same way in Blender (while
a turn about the world axis reverses the sideways tilts). `test_pipeline.py` checks the turn is rigid, keeps the
chirality and follows any rotation of the hand, and that sweeping, pushing and lifting the hand move it right,
forward and up.

## Which hand? Either.

The receiver decides whether the rig must be mirrored from **the finger bends themselves**, not from
MediaPipe's handedness label: for a given hand shape the fingers curl a fixed way relative to the
palm, and the rig's rest pose gives the same measure. The label is only a first guess until the
fingers have bent, and the first clearly bent frame corrects a wrong guess. Tested on a real
recording and on its exact mirror image (the other hand), with the label set to Left, Right and
absent: identical results every time. To check your own hands, record each and analyze it:

```
python v5/sender.py --record right.jsonl
blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/analyze_recording_headless.py -- v5 right.jsonl check=1
```

The analyzer reports, per joint, how far the rig's bend is from your real finger's bend (after the
straight dead zone below), counts "stuck" joints (your finger bent over 40 degrees while the rig's
is under 15) and shows how bent the rig is where your finger is straight. `MIRROR_OVERRIDE` at
the top of `blender_receiver.py` forces the decision if you ever need to.

## Why fingers could stay straight while you curled yours (found and fixed)

Reproduced on the real recording `landmarks.json` (105 frames of curling):

| Cause | Effect | Fix |
|---|---|---|
| Hand shape guessed from MediaPipe's label | wrong guess left 59% of curled joints stuck | decided from the finger bends |
| Rig's IK limits too tight on the side axis (real fingers curl in tilted planes) | 14% stuck even with the right mirror | side range widened (`IK_SIDE_SLACK_DEG`); the receiver enforces the exact limits |
| A .blend saved while the receiver had the rig mirrored holds a mirrored transform | "mirror" meant the opposite; fingers curled the wrong way | the rest transform is un-mirrored when read (`Rig`) |
| Sender filter lag (1.5 / 20 / 1.0) | rig lagged curls by 5 degrees, 39 at worst | 2.0 / 80 / 5.0: 1.9 and 16 (smooths a still hand a little less) |

After the fixes, on that recording (both hands): 0 of 166 curled joints stuck, bend error 1.8 degrees
at the knuckle, 3.0 at the middle joint, 2.8 at the tip, measured against your raw, unfiltered
fingers (the remaining error is mostly the filter's lag).

## Straight fingers stay straight (`STRAIGHT_ZONE_DEG`)

MediaPipe never reports a flat finger as perfectly straight: on the real recording every PIP and DIP
of a finger held straight reads about 5 degrees bent (live mean 6.5 and 4.8 degrees). The rig copied
that, so three joints of 5 degrees left the fingertip visibly curled. The rig itself was not adding
any curl (it followed the live bend to within a degree). The receiver now applies a dead zone to
PIP and DIP: a joint bend under 6 degrees is shown as 0, it fades in smoothly up to 20 degrees, and
above that it passes through unchanged. It applies to the joint's whole bend, in the finger's plane
and out of it. Result on the recording (both hands): straight joints show 0.4 degrees at PIP and
0.2 at DIP on average (they showed 5.9 and 4.2); curls over 20 degrees are unchanged.

The price: genuine bends under about 15 degrees are understated (a relaxed, slightly bent finger
looks a little straighter). Change `STRAIGHT_ZONE_DEG` at the top of `blender_receiver.py`: raise
the first number for stricter straightening, `(0, 0)` or removing a joint switches it off. No
change to the .blend is needed: the loader runs `blender_receiver.py` from next to it.

## Thumb, lost hand, and baking a take

- **Thumb limits** (`THUMB_BEND_MAX_DEG`): the thumb's knuckle bends at most 70 degrees and its tip
  joint 90; a segment bent further is swung back. The first thumb joint (CMC) moves freely. The thumb
  has no dead zone and its bend plane is not constrained.
- **Lost hand** (`LOST_HOLD_S`, `LOST_EASE_S`): when the sender reports no hand, the rig holds its
  last pose for 0.5 s and then relaxes to its rest pose over 1 s. The hand keeps its last position
  and orientation. `LOST_EASE_S = 0` holds the last pose until the hand returns. It only reacts to
  the sender's "lost" message, not to a sender that has died.
- **Bake a take into keyframes** so it plays without the sender or webcam:

```
blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/bake_recording.py -- v5 TAKE.jsonl OUT.blend [bones=1] [start=1]
```

  Every recorded frame goes through the real receiver and is keyed at the scene's frame rate on the
  empties and the Armature object (plays back through the IK like a live session). `bones=1` also
  bakes plain bone keyframes and removes the IK constraints in OUT, for export. It writes a new file
  and refuses to replace an existing one or the file it started from.

## Retargeting: why the empties sit on the bone tips

A bone has a fixed length and can only rotate, but a tracked hand has other proportions and its
measured segment lengths change from frame to frame, so copying landmark positions left empties
floating out of reach. The receiver (`RETARGET`) places each empty where its bone tip can be: the
live **direction** of the segment, the rig's own bone **length**, and (`CLAMP_TO_LIMITS`) the finger
joint limits applied to the directions. Each finger's bend plane is the live finger's own, so a
finger curling toward the thumb is followed. On a hand built differently from the rig, with noisy
segment lengths, random palm orientation and tilted finger planes, the average distance between an
empty and its bone tip is 0.00 milli-units (it was 33) and the bone-direction error 0.09 degrees.

The IK limits stored in the rig are only a wider safety net (`IK_SLACK_DEG`, `IK_SIDE_SLACK_DEG` in
`rig_tools.py`); the receiver's clamp is the exact anatomical limit. `FLEXION_RANGE`,
`MCP_ABDUCTION_DEG` and `SIDE_MARGIN_DEG` exist in both files; a test fails if they differ.

## What the fixed copy contains

Built by `build_rig_copy.py` from the pristine original (the source is only read):

- the meaningless IK `subtarget` cleared on all 20 constraints
- the 12 non-thumb finger bones rolled so one local axis is the finger's hinge (positions unchanged)
- IK limits on those bones: no twist, hinge range PIP 0-110, DIP 0-90, MCP -20..90 degrees, widened
  by a safety margin
- the superseded text blocks removed (`IK TESTING`, `edit mode`, `scale`, `small`, `Text`,
  `Text.001`); `emptyness creation`, `IK CONSTRAIN` and `bone mesh` kept
- text blocks `v5_run_receiver` and `v5_rig_tools`

Rebuild: `blender -b --disable-autoexec untitled.blend1 --python v5/build_rig_copy.py -- v5 v5/NEW_NAME.blend`
(use a new file name: Blender writes over a file you have open and save).

## Tests

```
python v5/test_pipeline.py
python v5/test_setup_tools.py
blender -b --disable-autoexec untitled.blend --python v5/test_rig_headless.py -- v5
blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/verify_copy_headless.py -- v5
blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/test_mimic_headless.py -- v5
blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/test_features_headless.py -- v5
blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/test_two_hands_headless.py -- v5
blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/test_skin_headless.py -- v5
blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/test_collision_headless.py -- v5
blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/test_fit_headless.py -- v5
blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/analyze_recording_headless.py -- v5 landmarks.json check=1
```

The Blender tests load the .blend in memory and never save it.

## Files

| File | Role |
|---|---|
| `sender.py` | webcam, MediaPipe world landmarks, outlier gate, One Euro filter, UDP |
| `blender_receiver.py` | Blender timer: drain socket, scale, shape decision, palm tracking, retarget, empties |
| `protocol.py` | packet format and landmark names |
| `one_euro.py`, `outlier.py` | filtering |
| `recording.py`, `replay.py` | record and replay |
| `rig_tools.py` | rig validation, IK cleanup, bone roll, joint limits; the only bone-to-empty mapping |
| `setup_tools.py` (+ `requirements.txt` in the project root) | one-command setup: pip packages, MediaPipe model and Blender into `tools/` |
| `build_rig_copy.py` | builds a fixed rig copy |
| `build_two_hands.py` | adds a second hand (armature, empties, mesh) to a fixed copy |
| `fit_skeleton.py`, `hand_detect.py` | fit the skeleton to the mesh with MediaPipe on renders of it (a new file) |
| `pose_gallery.py` | render the mesh in a set of extreme poses (with the joints drawn on, `skeleton=1`) |
| `skin_mesh.py` | re-skins the hand mesh to the 20 bones and stores each bone's skin thickness (a new file) |
| `slots.py` | keeps each detected hand in the same slot (sender) |
| `blender_panel.py` | sidebar panel: start/stop, status, mirror choice (loaded by the receiver) |
| `bake_recording.py` | bakes a recorded take into keyframes (new file) |
| `frames_io.py` | loads recordings for the offline tools |
| `analyze_recording_headless.py` | scores how well the rig follows a recording (your real fingers) |
| tests | `test_pipeline.py`, `test_rig_headless.py`, `verify_copy_headless.py`, `test_mimic_headless.py`, `test_features_headless.py`, `test_two_hands_headless.py`, `test_skin_headless.py`, `test_collision_headless.py`, `test_fit_headless.py` |

## Known limits

- The test data is one real recording of one hand plus synthetic hands; I have not watched a live
  webcam session.
- The two-hand .blend files must come from `build_two_hands.py` as it is now: older ones
  (since deleted) had the armatures hidden and the mesh shown, which also stopped
  `bake_recording.py bones=1` from baking. Use `untitled_two_hands_swapped.blend`.
- The fit comes from MediaPipe's idea of where the joints are on a render of the mesh (9 of 13 views
  detect a hand, landmarks agree to within 0.009 of the picture width). Check it in your Blender with the
  bones shown over the mesh; to redo or adjust it, run `fit_skeleton.py` again on another file.
- Still at the edges of the model: a hand's twist about its own fingers is locked (the IK keeps
  segments from rolling), the thumb has bend limits only, and linear-blend skinning stretches the outside of a
  strongly bent joint (the mesh volume is preserved).
- `test_skin_headless.py` first checks that the weights in the file are broken, so run it on the v2 file
  or the original, not on a v3 file. A thin light line can show along the pinky side of the hand in
  side views; the mesh edges there grow under 5 mm, so it looks like a gap in the scanned mesh itself.
- The thumb only has bend limits (no bend plane), and finger segments get no twist beyond what the palm gives.
- The panel and the bake tool were tested headless; I have not clicked the panel in an open Blender window.
- Two hands: a new hand takes the rig MediaPipe's label names, and after that it keeps its rig by
  position and motion (see "Two hands"), so two hands crossing at the same spot and height can still
  swap slots if the label also flickers for 8 frames. The second rig
  sits at a fixed place; with `TRANSLATE_HAND` off neither rig follows where its hand is in the room.
  I have not run two hands from a live webcam. The sender's two-hand loop is tested with a fake
  camera and a fake MediaPipe (both hands sent and drawn, the switch to two hands, the warnings), but
  whether MediaPipe finds both of your real hands is only visible live.
- Global hand translation (`TRANSLATE_HAND`) is off by default. Its depth is the least accurate axis.
- MediaPipe's hand model gives no per-landmark confidence, so the outlier gate is the substitute.
- Filter settings trade smoothness for lag; raise `--min-cutoff` / lower `--beta` for a steadier hand.
- Camera and driver latency before `cap.read()` returns is not measured.
