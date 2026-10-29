# Resident downstream scene interpreter

`demo2gripper.test_tube_downstream_scene` fills the perception gap between the
source-tube F2G proposal and the rack insertion stages. It is a long-lived
component of the same seven-stage process; an operator does not invoke it one
stage at a time.

## Pose ownership after contact

| Stage | Metric pose source | Front RGB role |
| --- | --- | --- |
| Extract | current TCP composed with the contact-grounded `T_tube_tcp` | confirm that the visible tube axis agrees; debounce relative lift clearance |
| Micro-loosen / pivot | current TCP pivot point plus the intersection of the calibrated pivot plane and the back-projected RGB line plane | provide the live tube-axis line and the calibrated target-angle gate |
| Reclamp | same live pivot reconstruction | confirm target angle while tactile confirms bilateral contact and stopped rotation |
| Approach / align / insert | current TCP composed with the frozen post-pivot `T_tcp_tube` | reject a kinematic pose that disagrees with the visible tube |
| Release | measured socket pose, inserted kinematic tube pose, and a stable fresh RGB line window | confirm that the tube remains at the qualified support pose |

The pivot reconstruction needs both planes. A 2-D line by itself leaves one
3-D degree of freedom and is never promoted to a metric tube pose. Likewise,
the socket always comes from a measured, runtime-qualified `T_base_socket`;
the image never creates or updates a metric socket frame.

At tactile-confirmed reclamp the interpreter and live adapter both freeze:

```text
T_tcp_tube_held = inverse(T_base_tcp) @ T_base_tube_rgb_pivot
```

Every later observation uses the current measured TCP:

```text
T_base_tube = T_base_tcp_current @ T_tcp_tube_held
```

This is the term that was missing when scene observations received only an RGB
frame and a state name. `SceneInterpreter.observe` now also receives the fresh
`T_base_tcp`; using the preceding frame's TCP while the arm moves is forbidden.

For the normalized top-down demo relation, `T_tube_tcp_demo_grasp` stays null
in configuration. A configured legacy relation is also only a planning
candidate because the final Servo/recenter correction may change it. At the
final stationary, bilateral-contact source observation, before the first
extraction motion, the live adapter freezes and injects the actual pair:

```python
scene.set_grasp_relation(
    T_tube_tcp_actual,
    T_base_tube_at_contact=T_base_tube_from_the_same_observation,
)
```

The interpreter copies both transforms atomically, updates the inverse and the
pivot point in the tube frame, and stores the lower-tip height at contact. The
first post-lift frame can therefore satisfy the 30 mm clearance predicate; it
does not reset the baseline after the lift. A repeated identical injection is
idempotent, while rebinding either transform is rejected. Until this handoff,
EXTRACT and pivot poses and predicates fail closed. A late injection after a
downstream observation is rejected rather than inferring contact after motion.

## Chaining with the source tracker

Use the downstream factory inside the source tracker configuration:

```json
{
  "factory": "demo2gripper.test_tube_pose_tracker:create_static_source_tube_scene_interpreter",
  "config": {
    "camera_serial": "254622073804",
    "reviewed_seed_rgb": "/qualified/source_seed_rgb.png",
    "reviewed_initial_mask": "/qualified/source_mask.npy",
    "reviewed_seed_qualification_receipt": "/qualified/source_seed_receipt.json",
    "camera_extrinsic_qualification_receipt": "/qualified/camera_receipt.json",
    "intrinsics": "/qualified/K.npy",
    "reviewed_open_uv": [310, 241],
    "reviewed_closed_uv": [228, 367],
    "reviewed_endpoints_evidence_id": "reviewed-current-camera-source-ends",
    "downstream_factory": "demo2gripper.test_tube_downstream_scene:create_qualified_held_tube_scene_interpreter",
    "downstream_config": {
      "camera_extrinsic": "/qualified/T_base_camera.npy",
      "camera_extrinsic_qualification_receipt": "/qualified/camera_receipt.json",
      "intrinsics": "/qualified/K.npy",
      "camera_serial": "254622073804",
      "static_socket_pose": "/qualified/T_base_socket.npy",
      "static_socket_pose_qualification_receipt": "/qualified/socket_receipt.json",
      "scene_qualification_receipt": "/qualified/scene_receipt.json",
      "rgb_tracker_qualification_report": "/qualified/rgb_tracker_report.json",
      "pivot_reference": "configs/test_tube_front_rgb_pivot_reference_20261023.json",
      "T_tube_tcp_demo_grasp": null,
      "T_socket_tube_entry": null,
      "reviewed_seed_line_uv": null,
      "axis_roi_uvxy": [0, 0, 640, 480],
      "pivot_plane_normal_tcp": [1, 0, 0],
      "pivot_point_tcp": [0, 0, 0],
      "jaw_axis_tcp": [1, 0, 0],
      "socket_outward_axis_socket": [0, 0, 1],
      "policy": {
        "tube_length_m": 0.11,
        "insertion_target_depth_m": 0.02
      }
    }
  }
}
```

The null socket-entry transform and line above are placeholders and
deliberately fail static preflight. `T_tube_tcp_demo_grasp` may remain null in
normalized top-down mode; both normalized and legacy modes still require the
runtime contact injection. The complete machine-readable placeholder is
`examples/test_tube_downstream_scene.config.template.json`.

The factory validates, before hardware opens:

- a fresh camera extrinsic artifact and qualification receipt;
- a measured static socket artifact and qualification receipt;
- one shared `workspace_calibration_id` for camera, socket, and scene;
- exact artifact SHA-256 bindings;
- an eligible three-trial robot-held RGB tracker report;
- the same camera serial, intrinsics, and pivot-reference hash used during RGB
  qualification;
- a socket-entry transform, ROI, and reviewed line seed. A legacy demonstrated
  transform may be supplied as a planning candidate, but the runtime contact
  injection remains mandatory in either mode.

The current camera was moved after the older eye-to-hand result. That older
`T_base_camera` must not be put in these fields. Until a fresh extrinsic and a
measured socket receipt exist, base-frame alignment and support predicates stay
blocked.

## Runtime failure behavior

The interpreter returns no held pose and keeps stage predicates false when the
runtime grasp handoff is missing, the line is missing or ambiguous, the line
disagrees with the metric projection, the pivot planes are degenerate, the
current TCP is absent, the camera identity changes, or a qualification is
missing. The resident controller then pauses the arm and reacquires evidence.
It does not substitute a plan-only demo relation, neighbouring D435 depth, a
stale robot pose, or a manually watched transition.

