# One-command test-tube inference stack

`scripts/run_test_tube_live_stack.py` is the operator entry for the complete
resident task. The default mode is inert. It validates every artifact and
writes `stack_report.json`; it does not start F2G, ROS, a camera, tactile
sensors, Hand-E, or FR3.

The resident process must be able to import the RealSense SDK, Hand-E serial
transport, ROS Python bindings, and the proprietary GF220 SDK in the same
Python environment. On this workstation the project `.venv` is prepared with
the compatible public packages through `.[live]` and the local
`pyvitaisdk-1.0.11-cp312-cp312-linux_x86_64.whl`. The vendor wheel constrains
NumPy to 1.26, SciPy to 1.15, and OpenCV to 4.11; do not upgrade those packages
independently in the live environment.

```bash
.venv/bin/python -m pip install -e '.[dev,live]'
.venv/bin/python -m pip install /path/to/pyvitaisdk-1.0.11-cp312-cp312-linux_x86_64.whl
```

```bash
cd /home/vitai/lzr/tactile_agent
.venv/bin/python scripts/run_test_tube_live_stack.py \
  --stack configs/test_tube_live_stack.5090.template.json \
  --output-dir runs/preflight/test_tube_live_stack
```

Add `--runtime-check` to read the F2G `/health` endpoint and the ROS graph. This
still sends no robot or gripper command. The checked-in 5090 manifest records
the verified remote F2G/RADIO revisions and the loopback SSH tunnel command.
GPU 6 was free during deployment verification; it is an operator-selected
default, not a permanent reservation.

After copying both templates to non-template files and replacing every
qualification placeholder, one command starts the managed SSH/F2G boundary,
starts or attaches MoveIt Servo, and delegates the whole seven-stage task to
the resident controller:

```bash
.venv/bin/python scripts/run_test_tube_live_stack.py \
  --stack configs/test_tube_live_stack.5090.json \
  --output-dir runs/live/test_tube_closed_loop_$(date +%Y%m%d_%H%M%S) \
  --execute \
  --accept-arm-only-collision-model
```

There are no per-stage operator commands. The managed processes are stopped
when the task finishes or fails. Static blockers prevent even infrastructure
startup, so the launcher cannot turn nulls into qualification receipts.
The resident task child sources the configured ROS and Franka workspace setup
scripts and then executes the same venv Python that launched the wrapper; the
Servo process and the controller therefore see the same ROS graph without
dropping Python dependencies.

`--accept-arm-only-collision-model` is a process-local acknowledgement, not a
qualification receipt. When the FR3+Hand-E collision/payload receipt is absent,
the flag removes only that exact missing-receipt blocker for this invocation.
Camera and TCP gates remain independent. Both the stack and resident reports
record that Hand-E, adapter, fingers, tactile attachments, payload mass, and
centre of gravity are outside the qualified collision/inertial model. The live
adapter then clamps MoveIt velocity/acceleration scales to `0.03`, Servo linear
speed to `0.03 m/s`, and Servo angular speed to `0.30 rad/s`. A present but
invalid collision receipt cannot be waived by the flag.

## Runtime-grounded human relation

The current demonstration does not need an independently reconstructed demo
6D tube pose. It contributes the reviewed axial fraction
`f = 0.664323374340949` and the thumb/index direction used only to choose the
symmetric jaw-axis sign. Every live observation instantiates:

```text
p_grasp = p_tube_center + (f - 0.5) * measured_tube_length * signed_tube_axis
tcp +Z  = base -Z
tcp +X  = perpendicular(tcp +Z, signed_tube_axis)
```

This is the correct representation for the requested top grasp. A constant
demo `T_tube_tcp` would rotate the tool away from base `-Z` whenever the live
tube orientation differed from the demonstration. At contact, the adapter
freezes the concrete measured `inv(T_base_tube) @ T_base_tcp`; extraction and
later kinematic tracking therefore do not require an unoccluded F2G result.
That same concrete relation is injected into the resident downstream scene
interpreter. The null `T_tube_tcp_demo_grasp` in the normalized template is
intentional; inserting an unmeasured candidate transform there is forbidden.

The normalized relation still requires a physical tube length with provenance.
The current 103.1 mm value from the adjusted-camera depth stripe is an
extrapolated plan-only estimate, so the template leaves `tube_length_m` null.

## What F2G does in this stack

F2G supplies a single-frame RGB-D pose and size proposal. It does not authorize
motion by itself. The source tracker requires:

- one qualified current-camera seed RGB, reviewed tube mask identity, and
  reviewed open/closed endpoints;
- a foreground depth cluster separated from the surrounding background;
- projected F2G axis and centre agreement with the warped reviewed endpoints;
- F2G axis and centre-line agreement with a PCA line fitted to the selected
  foreground 3D points;
- signed-axis canonicalisation, temporal filtering, bounded uncertainty,
  latency, jump, and staleness.

The cross-check only rejects proposals; it never edits an F2G result into a
passing pose. This matters because the current old-frame remote probe was about
80 mm and 47 degrees away from the independent tube evidence even though a
stable temporal filter could eventually accept repeated wrong outputs.

## One resident perception chain

The checked-in task template now names both concrete factories. The outer
factory tracks the source tube from reviewed RGB-D evidence and qualifies each
F2G proposal. Its `downstream_factory` remains alive for extraction, tactile
pivoting, rack alignment, insertion, and release:

```text
create_static_source_tube_scene_interpreter
  -> create_qualified_held_tube_scene_interpreter
```

The downstream interpreter receives the current measured TCP on every frame.
It combines robot-relative held-tube kinematics with the qualified front-RGB
axis, tactile pivot/reclamp events, and the measured socket transform. No
operator advances a stage. All artifact paths in the template that depend on
the current physical setup remain null, so static preflight stops before F2G,
ROS, cameras, or serial devices start.

## Remaining physical inputs

The launcher currently and intentionally reports blockers for:

- a fresh fixed-camera extrinsic and receipt for D435 serial `254622073804`;
  the camera moved about 5.45 degrees, and the recovered transform is plan-only
  with up to about 15.6 mm cross-reference deviation;
- a caliper-measured tube length and provenance receipt;
- a reviewed current-scene seed mask/endpoints plus a task qualification run
  for the F2G/geometric source tracker. The seed receipt must hash the seed
  RGB/mask/endpoints and bind their camera serial, K, workspace, and fresh
  camera-extrinsic qualification;
- current-camera intrinsics, a reviewed held-tube line/ROI, and an eligible
  robot-held RGB tracker report bound to the same camera and pivot reference;
- a measured rack socket frame, `T_socket_tube_entry`, and insertion depth,
  with camera/socket/scene receipts sharing one workspace calibration ID;
- a positive held-tube/pivot and post-release scene interpreter qualification;
- positive GF220 contact, pivot-angle, slip, resistance, and jam examples for
  this tube/rack setup;
- a measured Hand-E TCP plus gripper collision geometry, payload, and centre of
  mass;
- MoveIt Servo installation and a fake/real low-speed lifecycle test, including
  its Jazzy standalone auto-run state, pause around coarse MoveIt motion, TWIST
  reselection, status faults, and stop;
- aggregate source/held/socket pose receipts binding all producers to the same
  physical session.

Until those artifacts exist, `--execute` exits before starting F2G or ROS.
