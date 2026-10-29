# Continuous test-tube task runner

## Current demo07 execution

The current workstation entry is `scripts/run_test_tube_live_stack.py` with
`configs/test_tube_live_stack.5090.json`; it selects
`configs/test_tube_closed_loop.demo07.json`. This process owns observation,
planning, stage transitions, Hand-E feedback, and recovery. Reading its logs
does not advance a stage. Do not run the individual grasp helpers concurrently.

After a pre-close `SIDE_GRASP` hold, `--recover-from-run <previous-run-dir>`
allows the same entry to retreat through the previous run's measured approach
poses before acquiring a new scene. This option requires a matching task and a
trace with no close or contact; current FK must still match its final pose,
and current tactile/gripper feedback must confirm open and untouched. The
script rejects a moved arm or held object instead of reusing that path.

For this task, the final 20 mm may be committed while two distinct, fresh,
qualified source estimates agree and the gripper is open. The endpoint remains
the demonstrated object-relative grasp target; no extra downward offset is
added. The same speed limits and 8 mm maximum Servo increments still apply.
During this finite approach, transparent-depth loss alone does not cancel the
endpoint if current SAM/RGB identity, mask, and signed endpoints still agree
with the last metric-qualified frame. A missing image/service or changed image
geometry pauses motion. Fresh contradictory 3D evidence, unexpected contact,
hardware faults, or an exhausted 8-second approach budget prevent continuation.

Measured TCP convergence triggers one continuous close automatically. The
first fresh optical deformation on either finger pad stops that close; FORCE6D
is corroborating load/slip evidence rather than a prerequisite for light first
contact. Two fresh stationary windows must retain bilateral contact patches
away from the pad-image boundary before extraction. The first arrived window
stops the arm and latches its measured pose while image
identity is still valid; the next stationary confirmation uses TCP and tactile
feedback even if the fingers fully hide the tube. No further arm motion is
issued after that arrival latch. The
original source pose and timestamp survive that bounded approach and closure;
they are not relabelled as a fresh measurement. The adapter freezes the actual
grasp relation only after stopped, retained bilateral tactile contact. Later
extraction, pivot, insertion, and release still require their own predicates.
Insertion commands use an arrival tolerance no greater than 0.5 mm and one
third of their step, so a commanded insertion is not immediately mistaken for
an already-reached target. Ordinary moves retain their 1.5 mm tolerance and
recorded retreat paths pass continuously through intermediate waypoints.
This does not increase the configured motion speeds.

`task/steps/*.json` records measured observations, source rejection reasons,
the chosen action, and dispatch result. `task/run_report.json` distinguishes
`COMPLETE` from `HOLD`; a completed software test or a launched process is not
evidence of a completed physical task.

`scripts/run_test_tube_closed_loop.py` is the single task-level inference and
control entry point for the seven demonstrated stages:

```text
observe live tube/socket poses
  -> approach vertically from above
  -> transfer the full demonstrated T_tube_tcp grasp and close continuously
  -> extract
  -> tactile-supervised micro-loosen / gravity pivot / reclamp
  -> approach the rack from its live 6D frame
  -> align and insert with progress-plus-resistance jam recovery
  -> verify support, release, and retreat
```

The process is intended to stay alive for the whole task. It repeatedly
observes scene, robot TCP, Hand-E state, front RGB and tactile state; computes
one bounded action; persists an intent receipt; dispatches it through one
long-lived adapter; and observes again. Stage transitions come from
`decide_test_tube_transition`, not an operator watching the robot.

## Current demonstration readiness

The recording at `runs/live/test_tube_demo_20261023_02` supplies a reviewed
normalized grasp relation: axial fraction `0.664323374340949`, a signed
closed-to-open tube convention, top-down approach, and the thumb/index
direction used only for the parallel-jaw sign. This no longer requires a
same-frame demo 6D tube pose. Runtime still needs a measured tube length and a
qualified live tube pose whose origin is the geometric centre. The current
length estimate from transparent depth is plan-only, so the live template
leaves it null. The rack-entry origin, axis, and insertion depth are also not
measured. An identity transform must not be substituted for missing geometry.

The checked example is explicitly synthetic:

- `examples/test_tube_closed_loop.synthetic.config.json` labels both identity
  relations `synthetic_fixture_only` and sets
  `execution_scope=synthetic_replay_only`;
- `examples/test_tube_closed_loop.synthetic.replay.json` expands a deterministic
  nominal observation trace covering all seven stages;
- the CLI refuses `--execute` for that scope even if hardware flags are passed.

## Demonstration and top approach

For the preferred normalized mode, every fresh signed live pose instantiates

```text
p_grasp = p_tube_center + (f - 0.5) * measured_length * signed_tube_axis
tcp +Z  = base -Z
tcp +X  = perpendicular(tcp +Z, signed_tube_axis)
```

This keeps top approach as a hard constraint even if the live tube rotates.
The pregrasp translates along base `+Z`; the final motion comes along base
`-Z`. The human free-space trajectory is not replayed. Legacy fully qualified
`T_tube_tcp_demo_grasp` inputs remain supported for replay and other tasks.

A transparent cylinder has an unobservable roll gauge. The adapter maps the
estimated long axis to tube `+Z`, removes arbitrary roll with
`canonicalize_cylindrical_tube_pose`, and must disambiguate the axis sign. Tube
`+Z` means closed end to open end. The sign must come from a rim/open-end
detector or a previously signed temporally continuous track. PCA or a 6D model's
unsigned long axis is not enough.

## Real-time correction

Before contact, each fresh 6D estimate regenerates the desired TCP pose. The
controller compares it with measured robot FK and limits every correction by
translation and rotation step, total target drift, inter-frame jump, pose age,
inference latency, uncertainty, mask quality and depth-cluster quality. The
controller never sends image pixels directly to the robot.

Pose warm-up before motion keeps collecting frames. During a Servo stage, a
temporarily rejected mask, depth cluster, pose jump, or stale estimate sends a
zero-velocity arm command and retries acquisition in the same state. The
state deadline bounds this retry; a persistent loss, robot fault, tactile
fault, or exhausted recovery budget still enters terminal HOLD.

Near the fingers, the gripper can hide the tube. When this happens inside a
small already-aligned switch window, the controller freezes the last qualified
pose and transfers supervision to tactile. It does not keep visually servoing
on a corrupted mask. Occlusion before the switch window stops the task.

Hand-E receives one continuous close command. The loop only monitors it until
fresh optical tactile contact or Hand-E object feedback requests a stop. Hand-E
object status is corroborating-only and never authorizes extraction. If two
stationary windows have no bilateral optical contact, or their patch centres
remain at the pad boundary, the same resident policy reopens to at least
49.5 mm, descends exactly 5 mm along the already committed approach direction,
and closes continuously again. It may do this twice, with a hard 10 mm total
travel budget; it neither asks an operator nor assumes a base-frame down axis.
Fresh centred bilateral optical contact permits extraction. Persistent one-pad contact permits a
bounded open/recenter/reobserve recovery; it never permits extraction. Because
GF220 A/B has not been calibrated to TCP jaw sign, the controller does not map
"left sensor" directly to a correction direction. It opens 1--2 mm, tries a
1 mm TCP `+X` jaw-axis probe, recloses continuously, and compares first-contact
width/bilateral contact. If that is worse it crosses the original point once
with a 2 mm `-X` move. One reversal and total travel are budgeted; a motion
adapter without qualified bounded TCP servo support fails closed.

After pivot and stable reclamp, the controller measures and freezes the new
`T_tube_tcp_held = inv(T_base_tube) @ T_base_tcp`. Rack targets therefore use
the actual post-pivot held relation rather than the original grasp relation.

## Pose backend boundary

F2G-Pose is not imported by the pure controller. The transport contracts in
`pose_observation.py` and `f2g_pose_service.py` still need a task adapter that
maps a qualified service result into `ObjectPoseEstimate`. A live adapter must expose
`ClosedLoopIO` and a task qualification receipt that asserts these capabilities:

- metric full SE(3), continuous source-tube tracking;
- open/closed-end sign disambiguation and roll canonicalization;
- in-hand tube tracking after extraction;
- metric socket SE(3);
- timestamped latency/uncertainty, mask identity and temporal IoU;
- foreground depth support from a clustered masked-depth estimate.

The current SAM3 probe illustrates why the last item matters. A transparent
tube mask may contain many nominally valid depth pixels while most of them are
background at 1.2–2.0 m. The contract records valid depth count, foreground
support count/fraction, background fraction, and whether masked-depth
clustering qualified the selected foreground. A large raw count cannot pass a
background-dominated observation.

Temporal consistency alone cannot make a systematic F2G error safe. Before a
source proposal enters the controller, the tracker also compares its projected
axis/centre with warped reviewed endpoints and its 3D axis/centre-line with a
line fitted to the selected foreground depth cluster. Disagreement rejects the
proposal; the geometric evidence never overwrites it into a passing pose.

The existing RGB-D axis/anchor path is exposed as
`rgbd_axis_anchor`. It can seed a reviewed image axis or translation, but it
lacks full SE(3), in-hand tracking and socket pose. The runner reports those
missing capabilities and refuses live motion.

## Why the old process needed an operator

The previous tools produced snapshots or ran individual bounded commands. They
did not provide all of the following in one resident process:

- a task-qualified live 6D tube/socket estimator through gripper occlusion;
- an exact Hand-E TCP/collision model and post-pivot held-object transform;
- an in-process MoveIt goal handle with reliable cancel acknowledgement;
- calibrated tactile angle, slip and insertion-jam predicates;
- a metric transparent rack-entry frame and measured insertion depth.

Knowing camera pixels and robot FK is therefore insufficient. The missing term
is a fresh, identifiable metric transform between them, plus uncertainty and a
cancellable execution loop. This runner supplies the loop and all transition
contracts; it fails closed where a qualified producer is still missing.

## Modes

Dry-run is the default and opens no device:

```bash
python scripts/run_test_tube_closed_loop.py \
  --config /path/to/config.json \
  --output-dir runs/test_tube_closed_loop_plan
```

Replay drives the same controller and writes one atomic receipt per step:

```bash
python scripts/run_test_tube_closed_loop.py \
  --config /path/to/config.json \
  --replay /path/to/replay.json \
  --output-dir runs/test_tube_closed_loop_replay
```

The included fixture can be checked without a camera, ROS, or robot:

```bash
/home/vitai/miniconda3/envs/fhh/bin/python \
  scripts/run_test_tube_closed_loop.py \
  --config examples/test_tube_closed_loop.synthetic.config.json \
  --output-dir /tmp/test_tube_closed_loop_dry

/home/vitai/miniconda3/envs/fhh/bin/python \
  scripts/run_test_tube_closed_loop.py \
  --config examples/test_tube_closed_loop.synthetic.config.json \
  --replay examples/test_tube_closed_loop.synthetic.replay.json \
  --output-dir /tmp/test_tube_closed_loop_replay
```

The first report ends as `synthetic_replay_ready`. The second consumes 19
observations, reaches `COMPLETE`, and keeps
`hardware_commands_dispatched=false` throughout.

Live mode requires measured task geometry, a qualified backend receipt,
realtime kernel, explicit
collision-model acknowledgement, and an external `module:function` adapter.
No F2G-Pose result is promoted merely because a checkout imports successfully.

