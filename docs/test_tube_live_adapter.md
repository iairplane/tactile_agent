# Test-tube live adapter boundary

This document defines the remaining boundary between the reviewed test-tube
task runtime and real FR3/Hand-E/sensor I/O. It does not authorize execution.

## Continuous control loop

The live adapter must run one continuous loop:

```text
observe fresh scene, robot, gripper, RGB and tactile receipts
  -> build qualified state predicates
  -> decide_test_tube_transition
  -> persist a dispatch-intent receipt
  -> dispatch one bounded semantic action
  -> persist completion/cancel/failure receipt
  -> observe again
```

The adapter must never expose manual per-stage entry points as the normal task
path. A stage changes only through the compiled predicate guards.

## Required adapters

### Scene provider

Returns a timestamped, uncertainty-carrying observation containing:

- target identity separated from nearby tubes;
- metric `T_base_tube`;
- metric `T_base_socket` and outward socket axis;
- tube tip and socket entry;
- measured insertion depth/remaining depth;
- collision-scene identifier and source observation receipt.

The transparent tube and socket cannot use neighbouring background depth as
their centreline. A fiducial, opaque geometric anchor, reviewed stereo/multi-view
estimate, touch-off, or direct physical rack measurement must supply the
missing metric constraint.

### Motion adapter

Provides `plan`, `execute`, `cancel`, and `state` with an in-process ROS action
goal handle. Every result records:

- requested configured-EE target or bounded waypoint sequence;
- IK/collision result and Cartesian fraction;
- joint-path audit;
- goal ID and deadline;
- completion or cancel acknowledgement;
- measured final TCP and error.

`test_tube_live_io.RosMoveItServoRuntime` keeps TF, joint-state, and Servo
handles alive for the complete task. Coarse free-space targets use the audited
`moveit_pose_goal.py` path. Every bounded correction uses an active MoveIt
Servo `TwistStamped` loop. Missing `moveit_servo`, a missing Servo node, or a
missing topic subscriber stops adapter construction. Construction also calls
`/servo_node/switch_command_type` and requires an affirmative `TWIST` response;
having a topic subscriber alone is insufficient. `/servo_node/status` is kept
alive for the run and any non-`NO_WARNING` collision, singularity, or joint
limit status is latched as an arm fault. Servo corrections are never replaced
with repeated blocking MoveIt goals. Coarse MoveIt goals and Servo share the
same trajectory controller, so the runtime requires `/servo_node/pause_servo`,
confirms Servo is paused before a coarse goal, and resumes plus reselects TWIST
mode only after that goal returns.

The repository-local Servo config and one-process stack entry are documented in
`docs/fr3_moveit_servo.md`.

### Gripper adapter

Provides `status`, bounded absolute-width moves, and `stop`, with width,
motor/current or object state, command deadline, and completion receipt. The
existing Hand-E CLI supplies absolute `0..50 mm` moves and status, but still
needs an explicit stop primitive or a proven long-lived-session equivalent.

### Sensor providers

- `D435LiveSource` supplies serial-pinned, aligned, fresh RGB-D frames with
  intrinsics and hardware/SDK/host timestamps.
- `GF220LiveSource` supplies synchronized, fresh WARPED/DIFF/FORCE6D windows.
- `TubeAxisCandidateTracker` supplies TRACKED/HOLD/RESEED evidence.
- `TestTubeContactGate` supplies only CONTACT_PRESENT/CONTACT_ABSENT/INVALID;
  it never claims a stable grasp.

### Predicate builder

The builder is the only component allowed to promote raw observations into
state-machine predicates. It must bind each predicate to producer receipt IDs,
timestamps, qualification IDs, and the current task/session. Examples:

- `grasp_established` = qualified contact + valid retained Hand-E width/state +
  (bilateral contact or visual tube-between-fingers confirmation);
- `front_rgb_calibrated_angle_reached` requires the promoted robot-held tracker
  qualification, never the human-demo replay alone;
- `tube_stays_supported` requires a fresh scene observation after release;
- shadow-only slip or jam observations remain logs and cannot enter the promoted
  predicate set.

## HOLD semantics

For stale/invalid evidence, overload, provider failure, action deadline, or
budget exhaustion, HOLD must perform and persist this order:

1. cancel the active arm goal;
2. confirm cancellation or controller terminal state;
3. stop the active Hand-E command;
4. cease all further dispatch;
5. persist final robot, gripper, sensor, and goal-state receipts.

Opening the gripper or executing a retreat is a task recovery action, not an
emergency stop. The physical E-stop remains the final independent authority.

## Implemented resident adapter

The `ClosedLoopIO` implementation is
`demo2gripper.test_tube_live_io:create_live_io`. It owns one long-lived session
for:

- `D435LiveSource` and aligned RGB-D;
- `GF220LiveSource` and fresh paired tactile windows;
- `SerialModbusRTU` and Hand-E feedback/commands;
- ROS TF plus joint states, coarse MoveIt, and MoveIt Servo;
- two explicit pure predicate boundaries: a scene interpreter and a tactile
  interpreter.

Three different pose producers are deliberately kept separate:

1. **Source tube:** F2G-Pose proposes `T_camera_tube`; the qualified scene
   interpreter resolves the open/closed-end sign, cylinder roll, tracking ID,
   and temporal uncertainty before the pose can enter control.
2. **Held tube:** extraction first uses the qualified demonstration
   `T_tube_tcp`. During pivot, the scene interpreter must recover the tube axis
   from qualified front RGB/tactile evidence. Once reclamping is stable, the
   measured `T_tube_tcp_held` is frozen and subsequent poses come from current
   robot kinematics. F2G is not used to guess the gripper-occluded tube.
3. **Rack socket:** a qualified, measured static `T_base_socket` is emitted on
   every observation. It does not pretend to be an RGB-D estimate.

Non-RGB-D poses carry `requires_rgbd_quality_gates=false`; their own producer
qualification and uncertainty remain visible in `ObjectPoseEstimate`.

The source grasp may use a qualified full `T_tube_tcp` or the preferred
runtime-grounded normalized relation. In normalized mode the controller uses
the reviewed axial fraction, a measured tube length, exact base `-Z` approach,
and a jaw axis perpendicular to the signed live tube axis. At contact the
adapter freezes the resulting concrete `T_tube_tcp` for extraction.

The unfilled configuration is
`configs/test_tube_closed_loop.live.template.json`. Its nulls are intentional.
List all missing evidence without opening hardware:

```bash
python scripts/preflight_test_tube_live_io.py \
  --config configs/test_tube_closed_loop.live.template.json
```

After every null is replaced by a qualified artifact and MoveIt Servo is
installed/launched, inspect the live ROS boundary as a read-only check:

```bash
python scripts/preflight_test_tube_live_io.py \
  --config configs/test_tube_closed_loop.live.json \
  --check-servo-runtime
```

The stack wrapper performs static/read-only runtime checks and then runs the
complete state machine; the operator does not invoke individual stages. See
`docs/test_tube_live_stack.md`.

```bash
python scripts/run_test_tube_closed_loop.py \
  --config configs/test_tube_closed_loop.live.json \
  --output-dir runs/live/test_tube_closed_loop \
  --execute \
  --live-adapter demo2gripper.test_tube_live_io:create_live_io \
  --accept-arm-only-collision-model
```

The adapter additionally requires qualification receipts with schema
`demo2gripper.test_tube_live_io_qualification.v1`. A receipt has `kind`,
`qualification_id`, and `runtime_qualified=true`. Scene and socket receipts
also carry measured `position_std_m` and `orientation_std_deg`; the same is
required for the fixed-camera extrinsic, Hand-E TCP, and demonstration
`T_tube_tcp` relation so composed held poses retain uncertainty provenance. The aggregate
pose receipt keeps the runner's public backend descriptor auditable, while
separate source-F2G, held-tube, and static-socket receipts identify which
producer actually supplied each pose.

The live template deliberately contains `null` metric geometry. Run the
dedicated preflight command above to obtain every blocker at once. The main
runner also recognizes this template before `parse_geometry`, writes a
`blocked_live_execution` report with `hardware_opened=false`, and exits with
status 2 in dry-run, replay, and execute modes. It never constructs the live
adapter for an unparsed template.

On this host, the qualified FR3 workspace is the repository-local
`third_party/franka_ros2_ws/install_ros/setup.bash`: sourcing it exposes
`franka_fr3_moveit_config`. The similarly named external
`/home/vitai/lzr/franka_ros2_ws/install/setup.bash` does not exist, and the
repository's older `install/setup.bash` does not expose that package.

## Remaining physical qualifications

The adapter fails before opening devices until:

- exact live tube/socket geometry and insertion depth exist;
- a unique `T_tube_tcp` and Hand-E widths/close budget are selected;
- Hand-E/tube/rack collision objects and motion bounds pass plan-only checks;
- RGB, contact, jam, insertion-progress, and post-release-support predicate
  producers have been qualified;
- MoveIt Servo is installed, active, and subscribed to the configured twist
  topic;
- FR3 and D435 data paths pass a fresh health check.

