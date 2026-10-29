# Real-time source-tube pose qualification

`src/demo2gripper/test_tube_pose_tracker.py` turns a sequence of aligned RGB-D
frames into qualified tube poses. It does not open a camera, command a robot, or
claim that F2G-Pose is a tracker.

The source-stage data path is:

```text
qualified reviewed seed RGB + SAM3 mask + signed endpoints
  -> optical-flow/ECC registration from the reviewed RGB into live frame 0
  -> optical-flow propagation + ECC agreement
  -> foreground-depth/background-ring gate
  -> one F2G proposal
  -> reviewed open/closed endpoint sign on the first pose
  -> temporal sign continuity + canonical cylinder roll
  -> SE(3) window mean and measured dispersion
  -> ObjectPoseEstimate, or HOLD with no stale estimate
```

`mask_iou` is the measured flow-versus-ECC overlap (or flow-versus-external-mask
overlap). It is not ground-truth IoU and is never used as F2G model confidence.
The F2G response keeps `model_confidence=None`.

Use the scene adapter directly in the live configuration:

```json
{
  "factory": "demo2gripper.test_tube_pose_tracker:create_static_source_tube_scene_interpreter",
  "config": {
    "camera_serial": "254622073804",
    "reviewed_seed_rgb": "runs/reviewed/tube_seed_rgb.png",
    "reviewed_initial_mask": "runs/reviewed/tube_mask.png",
    "reviewed_seed_qualification_receipt": "runs/reviewed/source_seed_receipt.json",
    "camera_extrinsic_qualification_receipt": "runs/qualified/camera_receipt.json",
    "intrinsics": "runs/qualified/K.npy",
    "reviewed_open_uv": [311.0, 126.0],
    "reviewed_closed_uv": [326.0, 351.0],
    "reviewed_endpoints_evidence_id": "operator-review-20261024",
    "tracking_id": "source-tube-session-1",
    "mask_tracker": {
      "min_internal_iou": 0.65,
      "max_occluder_overlap_fraction": 0.2
    },
    "foreground_depth": {
      "max_mask_near_background_fraction": 0.5
    },
    "pose_qualifier": {
      "window_size": 7,
      "min_window_frames": 3,
      "max_translation_jump_m": 0.03,
      "max_orientation_jump_deg": 25.0
    }
  }
}
```

The seed receipt uses schema
`demo2gripper.test_tube_source_seed_qualification.v1`. It must be accepted for
runtime and bind the camera serial, workspace calibration ID, current camera
extrinsic qualification ID, SHA-256 values of both the reviewed RGB and mask,
the intrinsics artifact SHA-256, both signed endpoint pixels, and their
evidence ID. The factory checks the claimed camera/workspace IDs against the
actual camera qualification receipt. Every live frame must retain that camera
serial and exact K. The first frame is always registered from the reviewed RGB;
equal image dimensions are never treated as proof that old mask pixels still
identify the tube.

The field layout is provided in
`examples/test_tube_source_seed_qualification.template.json`. It is an inert
template with `runtime_qualified: false`; filling artifact paths does not turn
it into a qualification receipt.

An external segmentation refresh may be supplied with
`external_mask_pattern`, for example `/tmp/tube_masks/{sequence:06d}.png`. The
refresh is accepted only when it overlaps the optical-flow prediction. SAM3 is
therefore used for initialization or explicit refresh, not on every control
frame.

The adapter covers the pre-contact source stages. A complete task interpreter
can be composed through `downstream_factory`; without one, later visual
predicates stay false and the controller holds. Near grasp occlusion, execution
must switch to tactile and then robot-relative held-tube kinematics. Runtime
qualification receipts are still required by `test_tube_live_io`; this module
does not mint them.

The reviewed `T_tube_tcp` relation must use this same canonical tube frame:
local `+Z` points from the reviewed closed end to the open end, while roll about
that axis is deterministically canonicalized. A relation recorded in another
object-frame convention cannot be composed without conversion.

Run the CPU checks with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/test_test_tube_pose_tracker.py tests/test_f2g_pose_service.py
.venv/bin/python -m ruff check \
  src/demo2gripper/test_tube_pose_tracker.py \
  tests/test_test_tube_pose_tracker.py
```

