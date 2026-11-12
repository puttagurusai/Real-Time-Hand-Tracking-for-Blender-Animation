# Project guide: webcam hands to a rigged Blender hand

Read this first. It gives the whole picture in one pass; [README.md](README.md) is the detailed reference
(every feature, with measurements). Nothing here replaces it.

## 1. What this is

A webcam watches your hands. MediaPipe finds 21 landmarks per hand, a Python **sender** cleans and sends
them over UDP, and a **receiver** running inside Blender turns them into bone motion on a scanned hand
mesh (about 600,000 vertices). Your **left hand drives the left rig and your right hand the right rig**,
and the result is not mirrored: a right hand stays a right hand.

```
webcam -> sender.py (MediaPipe, outlier gate, One Euro filter, hand slots) --UDP 127.0.0.1:5004--> Blender:
          blender_receiver.py (scale, un-mirror, turn, retarget, finger limits, collision) -> empties -> IK -> bones -> mesh
```

| Stage | Where | What it does |
|---|---|---|
| Capture and detect | `sender.py` | OpenCV frame, flipped like a mirror, MediaPipe HandLandmarker, world landmarks (metres) and a Left/Right label |
| Clean | `outlier.py`, `one_euro.py` | reject jumps, smooth with a One Euro filter |
| Who is who | `slots.py` | keeps each hand in its own slot (0 = left hand, 1 = right hand) |
| Transport | `protocol.py` | small JSON packets: `frame`, `lost`, `stop`; Blender answers `ack hands=N` |
| Shape the hand | `blender_receiver.py` | scale from the rig, un-mirror, turn 180 degrees about the hand's own axis, retarget to the rig's bone lengths, joint limits, straight dead zone, finger collision |
| Move the rig | Blender | the receiver moves 21 empties; IK constraints on the bones follow them |
| Look | `skin_mesh.py`, `fit_skeleton.py` | the skeleton is fitted to the mesh and the mesh is re-skinned |

Run order: open the .blend, run the text block `v5_run_receiver`, then `python v5/sender.py`.

## 2. Setup

- **Blender 5.1** (tested headless at `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`). Its own Python runs the receiver.
- **Python 3.13** with `opencv-python`, `mediapipe`, `numpy` for the sender (and for `fit_skeleton.py`'s `python=` option, which runs MediaPipe outside Blender).
- **`hand_landmarker.task`** (MediaPipe model) in the project root, next to the `v5` folder. `sender.py` looks for it there.
- Windows 11 was used. Camera 0 at 640x480, 30 fps by default; UDP port 5004 on 127.0.0.1 (`--port` on the sender, `PORT` in the receiver, they must match).

```
pip install opencv-python mediapipe numpy
```

Files to open in Blender: `v5/untitled_rig_fitted.blend` (one hand) or `v5/untitled_two_hands_swapped.blend`
(two hands; left rig at the origin, right rig 0.8 units to +X). The mesh is hidden: show it with the eye icon.

## 3. How we got here (in order, and why)

1. **Original project** (parent folder: `hand_full_control*.py`, `src/`, `hand.blend`, ...). Landmarks were sent to Blender and empties drove IK bones. It worked loosely; those files were left untouched (`PROJECT_OVERVIEW.txt` describes them).
2. **A clean pipeline** (`v5`): a versioned packet protocol, outlier gate and One Euro filter, a stale-frame check, recordings and replay, and tests that run without a webcam or Blender (a fake `bpy`).
3. **A safe rig copy** (`build_rig_copy.py`, `rig_tools.py`): the original rig had meaningless IK targets and no bend limits, so a copy gets cleaned IK, finger bones rolled so each has a hinge axis, and joint limits. The source file is only ever read.
4. **Right hand to the right rig, either hand works**: whether the rig must be mirrored is decided from the finger bends, not from MediaPipe's label (a wrong label left most curled joints stuck). Filter settings were retuned to cut lag.
5. **Straight fingers stay straight**: MediaPipe reads a flat finger as about 5 degrees bent, so PIP and DIP get a dead zone (`STRAIGHT_ZONE_DEG`).
6. **Retargeting and limits**: each empty is placed where its bone tip can reach (live direction, the rig's own bone length), thumb limits, and a relax-to-rest when the hand is lost.
7. **Bake a take** to keyframes (`bake_recording.py`) so it plays without the sender or webcam.
8. **Two hands** (`build_two_hands.py`, `slots.py`): a second armature, empties and mesh copy; each hand has its own gate, filter, scale, mirror decision and relax. Left/Right label picks a hand's rig, position and motion keep it there. The sender follows Blender's `ack hands=N` and rebuilds MediaPipe for one or two hands.
9. **The mesh looked wrong in strong poses.** Two causes were found and measured: broken skin weights (5% of vertices had no bone, the ring fingertip's weights sat in a mistyped group) and a skeleton whose joints were not where the mesh bends (the bone lengths were about right). Fixes: `skin_mesh.py` (re-skin, rigid wrist cuff), `fit_skeleton.py` (render the mesh from 13 directions, MediaPipe finds the joints, triangulate, centre each joint in its own finger), a rigid palm, and finger collision so fingers do not pass through each other.
10. **Rig placement and signal direction**: the two rigs were swapped in the scene, and the slot mapping reversed so your right hand drives the right rig on screen.
11. **No mirror feeling and facing** (`UNMIRROR_HAND`, `HAND_YAW_DEG`): depth is reflected back so Blender shows the hand you hold up, then each hand is turned 180 degrees about its own long axis so palm toward the webcam shows the palm in Blender's front view. Turning about the hand's own axis (not the world Z axis) keeps every tilt, push and lift in the same direction.
12. **Cleanup**: superseded .blend files, Blender `.blend1` backups and Python caches were deleted.

## 4. Conventions worth knowing

- **Your frame and Blender's front view**: X is your right, Y is toward the webcam, Z is up. Blender's front view looks along +Y.
- **Landmarks** arrive as a mirror image (the picture is flipped before MediaPipe, which is what its labels expect). The receiver reflects Y back (`UNMIRROR_HAND`), then turns the hand (`HAND_YAW_DEG`, `HAND_TURN_ABOUT`).
- **Slots**: 0 is the left hand (rig `Armature`, empties named as the landmarks), 1 is the right hand (`Armature.001`, `WRIST.001`, ...).
- **Scale** comes from the rig's own wrist-to-middle-knuckle bone (`Rig.span`).
- **New files, never overwrites**: every tool that changes a .blend writes a new one and refuses to replace the input. Blender overwrites a file you have open and save, so use new names.

## 5. Settings cheat sheet (top of `blender_receiver.py` unless noted)

| Setting | Default | Effect |
|---|---|---|
| `UNMIRROR_HAND` | `True` | show the hand as it really is; `False` gives the old mirror view |
| `HAND_YAW_DEG`, `HAND_TURN_ABOUT` | `180.0`, `"hand"` | turn about the hand's own axis (`"world"` = vertical axis); `0` turns nothing, which gives your own point of view and the matching curl direction |
| `TRANSLATE_HAND`, `TRANSLATE_GAIN` | `False`, `(1,1,1)` | also move the rig with the hand's position (depth is the weakest axis) |
| `MIRROR_OVERRIDE` | `None` | force mirrored or not mirrored if the automatic choice is ever wrong |
| `STRAIGHT_ZONE_DEG` | PIP and DIP `(6, 20)` | below 6 degrees shown as 0, fully passed above 20 |
| `THUMB_BEND_MAX_DEG` | `mcp 70, ip 90` | thumb bend limits |
| `RIGID_PALM` | `True` | palm bones keep their rest shape |
| `COLLISION`, `PALM_COLLISION`, `COLLISION_MARGIN` | `True`, `True`, `0.7` | fingers pushed apart instead of overlapping; 1 keeps them further apart |
| `LOST_HOLD_S`, `LOST_EASE_S` | `0.5`, `1.0` | hold the last pose, then relax to rest (`LOST_EASE_S = 0` holds until the hand returns) |
| `MAX_FRAME_AGE_S` | `0.5` | old frames are not applied |
| `PORT`, `TIMER_INTERVAL` | `5004`, 1/60 s | UDP port; the receiver drains the socket every tick |
| Sender `--hands` | `auto` | `auto` follows Blender, or `1` / `2` |
| Sender `--min-cutoff / --beta / --d-cutoff / --max-speed` | `2.0 / 80 / 5.0 / 4.0` | filter: steadier hand vs. less lag |
| Sender `--detect-confidence` | 0.7 (one hand), 0.5 (two) | lower it if the second hand is not found |
| Sender `--hfov`, `--camera`, `--width`, `--height`, `--fps` | 60, 0, 640, 480, 30 | camera and field of view |
| Sender `--record FILE`, `--dry-run`, `--no-preview`, `--no-mirror` | | recording, test without Blender, no window, do not flip the picture |

The Blender panel (sidebar, press N, tab **Hand**) gives Start / Stop, a live status per hand and a mirror
choice per hand.

## 6. Verified, and not verified

- **Verified**: 36 Python tests (`python v5/test_pipeline.py`, fake `bpy`, fake camera and MediaPipe) and the headless Blender suites listed in the README's Tests section, on the real rig and on a real recording (`landmarks.json`, 105 frames of one hand). Checks include palm and thumb direction, tilt directions, un-mirror, two-hand slots, skin quality, collision and the fit.
- **Not verified**: a live webcam session (one or two hands), whether MediaPipe finds both real hands, and clicking the panel in an open Blender window.
- **Known caveats**: a palm shown to the viewer curls toward the viewer (set `HAND_YAW_DEG = 0` for your own point of view); linear-blend skinning still stretches the outside of a strongly bent joint; twist along fingers is locked; the thumb has bend limits only. More in README "Known limits".

## 7. Troubleshooting

| Symptom | Likely cause | Check and fix |
|---|---|---|
| Nothing moves | receiver not running, or ports differ | run `v5_run_receiver` first; sender and receiver both on 5004; the sender prints `receiver=acked` when Blender answers |
| Only one hand moves | the .blend has one rig, or MediaPipe sees one hand | preview shows `hands seen N, tracking M`; open `untitled_two_hands_swapped.blend`; `--hands 2`; lower `--detect-confidence` |
| `seen 2` but `tracking 1` | sender has not heard from Blender yet | start the receiver; wait for `Blender has 2 hand rigs` |
| Right hand moves the left rig | slot mapping edited | `LEFT, RIGHT = 0, 1` in `slots.py` and `HAND_NAMES` in the receiver must agree (a test checks) |
| Feels mirrored | `UNMIRROR_HAND` off | set it to `True` |
| Fingers curl the opposite way to mine | palm-forward view curls toward the viewer | `HAND_YAW_DEG = 0` |
| A finger will not curl | wrong mirror decision | status line shows `mirrored (finger bends)`; use the panel's mirror choice or `MIRROR_OVERRIDE`; analyze a recording (README "Which hand? Either.") |
| Fingers curled when I hold them flat | MediaPipe noise | widen `STRAIGHT_ZONE_DEG` |
| Mesh not visible | it is hidden in the file | eye icon, or H / Alt+H |
| Slow viewport | the mesh has about 600,000 vertices | hide the mesh while testing, or run the tracking with the mesh hidden |
| Two hands swap rigs | hands crossing at the same place and the label flickering | keep them apart; labels need 8 consecutive frames to move a hand |
| Saved file looks wrong after a session | Blender overwrote an open file | save under a new name |

## 8. Where to go next (ideas, not done)

1. Panel switch for "face me / my view" instead of editing constants.
2. A calibration pose (flat hand, fist) to learn each finger's range.
3. Wrist, forearm and elbow bones.
4. Corrective shapes at the knuckles and a lighter proxy mesh while tracking.
5. A second camera for depth and hidden fingers.
6. Gestures (pinch, fist, point) that trigger Blender actions.
7. Record and bake from the panel; export baked takes to FBX or glTF.
8. One-click setup (fit, skin, two-hand build) as panel buttons; packaging as an add-on.

## 9. Which file is which

- `v5/untitled_rig_fitted.blend`, `v5/untitled_two_hands_swapped.blend`: use these.
- `v5/untitled_rig_fixed_v2.blend`: original skeleton and original weights; the input for the build, fit, skin and most test scripts.
- `v5/untitled_rig_fixed.blend`: oldest fixed copy; still read by a few tests.
- Parent folder (`hand.blend`, `untitled.blend`, scans, scripts, zips): your originals, untouched.
- Scripts and tests: see the README's Files table.
