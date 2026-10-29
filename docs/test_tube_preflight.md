# Test-tube task: offline compiler and preflight boundary

The reviewed demonstration in
`runs/live/test_tube_demo_20261023_02/analysis/stage_plan_refined.json` defines the task
intent. It does **not** define an executable robot trajectory. The offline
compiler converts that review into a fixed state machine and stops before any
ROS, MoveIt, Hand-E, or tactile-device command can be sent.

## Responsibility split

| Authority | Responsibility |
|---|---|
| Agentic planning | Free-space approach to the tube, extraction clearance, socket standoff, jam-recovery re-observation/realignment, and retreat. MoveIt will eventually plan these paths from the measured robot state. |
| Object-relative pose | Side-grasp `T_tube_tcp`, the tube-axis target after pivot, tube-to-socket axis alignment, and insertion along the live socket axis. The human hand path is never replayed. |
| Tactile supervision | Session-bootstrap contact present/lost selects continue/reclamp. Slip and insertion-jam features are recorded in shadow mode for this run. Tactile predicates do not generate free-space poses. |
| Perception supervision | Target identity, metric scene frames, tube-between-fingers confirmation, front-RGB tube angle, source clearance, socket alignment, insertion progress, and support after release. |

The seven human stages compile into smaller runtime states:

```text
OBSERVE
  -> APPROACH_TUBE
  -> SIDE_GRASP
  -> EXTRACT
  -> (PIVOT_MICRO_LOOSEN <-> PIVOT_EVALUATE)
  -> RECLAMP
  -> APPROACH_SOCKET
  -> ALIGN_SOCKET
  -> (INSERT_MICROSTEP <-> INSERT_EVALUATE)
  -> RELEASE
  -> RETREAT
  -> COMPLETE
```

`HOLD` is terminal. `JAM_RECOVERY_RETRACT` returns to `ALIGN_SOCKET` after a
bounded retract and a fresh scene observation.

## Predicates and recovery precedence

Recovery events are evaluated before normal progress:

1. Invalid/stale evidence or overload stops in `HOLD`.
2. During micro-loosen/pivot, `contact_lost` immediately enters `RECLAMP`.
   Bilateral symmetry is not required; any qualified finger contact is
   sufficient to keep monitoring a side-held tube. `slip_detected` joins this
   branch only after a labelled runtime qualification; it is shadow-only now.
3. With contact maintained, this run uses only a calibrated front-RGB image-axis
   trigger learned from the same fixed-camera demonstration to finish the pivot
   and enter `RECLAMP`. Tactile angle is excluded from the current plan. It may
   become an alternative only after a separate robot-contact angle calibration.
4. The insertion contract advances only in bounded steps along the live socket
   axis. A future jam predicate needs both low axial progress over the
   configured window and a calibrated resistance/shear rise. One force spike
   alone is not a jam. The current jam predicate is shadow-only, so these
   physical insert/recovery transitions remain disabled.
5. A jam stops the push, retracts along the socket axis, re-observes, realigns,
   and retries within a fixed budget. Exhausting the budget enters `HOLD`.

`contact_present` alone never authorizes extraction. `SIDE_GRASP -> EXTRACT`
requires a separate `grasp_established` predicate combining valid gripper
state, an in-range retained width, contact, and either bilateral contact or a
visual confirmation that the target tube lies between the fingers. Any-pad
contact remains useful for contact-loss monitoring after the grasp.

These gates are intentionally practical: contact reacts after two qualified
windows, the visual angle target uses four frames, and a future jam gate uses two
windows. Contact uses the implemented
[`test_tube_contact_bootstrap.py`](../src/demo2gripper/test_tube_contact_bootstrap.py)
policy on the first robot session:

1. Create one runtime `session_id`. Before the `SIDE_GRASP` close, capture at
   least three non-overlapping unloaded dual-FORCE6D windows with five or more
   unique samples per finger.
2. Apply the reviewed side-grasp relation and one bounded close with the tube
   between the fingers.
3. After that close, capture at least three non-overlapping positive-contact
   windows from the same session, sensor serials, and calibration IDs.
4. Call `bootstrap_test_tube_contact`, then activate `TestTubeContactGate` for
   this session. It uses a pooled MAD noise estimate, requires a repeatable
   response above both an absolute floor and an eight-noise margin, and applies
   on/off hysteresis to the maximum raw SDK Fz change across qualified pads.
5. Continue when contact is present; return to `RECLAMP` immediately when an
   established contact is lost. Invalid, stale, replayed, or provenance-changed
   windows or a session mismatch produce `HOLD_INVALID`.

The fitted values are raw sensor units and are never reused across sessions.
This bootstrap qualifies contact presence only; it does not qualify physical
force, tube angle, slip, or insertion jam.

`slip_detected` and `insertion_jam` remain shadow-only because this recording
contains no robot-finger/tube contact and no controlled obstruction. They are
logged but cannot trigger a motion transition. This does not block free-space
MoveIt plan-only checks; it keeps the corresponding physical pivot/insertion
transitions disabled until labelled calibration trials exist.

## Compile the current dry run

```bash
cd /home/vitai/lzr/tactile_agent
python scripts/compile_test_tube_preflight.py \
  --stage-review runs/live/test_tube_demo_20261023_02/analysis/stage_plan_refined.json \
  --config configs/test_tube_preflight_20261023.json \
  --output outputs/test_tube_preflight_20261023/compiled_plan.json \
  --overwrite \
  --require-offline-contract-ready
```

This command always remains offline. It reads JSON, validates the task
contract, hashes available evidence, and writes blockers.
`--require-offline-contract-ready` returns zero for the current reviewed
contract. `--require-plan-only-ready` deliberately returns status 2 until fresh
metric `T_base_tube`, `T_base_socket`, and insertion depth are available.

The generated contract always contains:

```json
{
  "execution_authorized": false,
  "hardware_execution_enabled": false,
  "hardware_boundary": {
    "dispatch_supported": false,
    "robot_or_gripper_imports": false,
    "subprocess_commands_present": false
  }
}
```

The compiled contract also carries a producer manifest for every promoted
predicate.  The hardware-independent runtime in
[`test_tube_runtime.py`](../src/demo2gripper/test_tube_runtime.py) accepts only
timestamped producer receipts for the current task/session, persists dispatch
intent before an action, and persists the action result afterward.  Runtime
budgets are owned by that loop rather than accepted from sensor providers.
[`test_tube_predicates.py`](../src/demo2gripper/test_tube_predicates.py) is the
typed receipt-to-predicate boundary; shadow observations cannot be promoted by
supplying similarly named JSON fields. Live operation accepts that provider
type only. Offline `JsonReplay` must also be its own semantic action executor,
so replay evidence cannot be connected to hardware dispatch.

The live RGB input boundary is implemented by
[`d435_live.py`](../src/demo2gripper/d435_live.py).  It keeps one serial-pinned
pipeline alive, aligns depth to color, carries intrinsics and hardware/SDK/host
timestamps, and rejects stale or replayed frames.  The tracker is still barred
from controlling the pivot until
[`test_tube_rgb_qualification.py`](../src/demo2gripper/test_tube_rgb_qualification.py)
passes at least three distinct robot-held trials, including labelled occlusion
and distractor cases. Distinctness is bound to capture-session IDs and complete
recording-content fingerprints; trial IDs, annotations, ROIs and replay ranges
do not contribute. The compiled evidence hashes both this evaluator and
[`qualify_test_tube_rgb_tracker.py`](../scripts/qualify_test_tube_rgb_tracker.py),
which performs replay and provenance validation.

It also evaluates three deterministic semantic traces with the same pure state
transition function used by the contract:

- the nominal sequence from `OBSERVE` through `COMPLETE`;
- `contact_lost` during pivot, followed automatically by
  `RECLAMP_RECOVERY -> PIVOT_EVALUATE -> RECLAMP`;
- a shadow-only insertion-jam observation that is logged but is not promoted to
  a physical recovery predicate.

Every trace records predicates, selected action, and next state, with
`manual_stage_calls_required=false` and `hardware_commands_dispatched=false`.
They prove deterministic stage dispatch order; they do not claim that missing
live metric geometry or runtime gates have been qualified.

## What still has to be measured at execution start

The three reviewed geometry artifacts make the offline review contract
complete. They do **not** make the MoveIt plan-only boundary ready, because
they deliberately keep exact executable transforms null rather than inventing
values from transparent-object RGB-D:

- instantiate and collision-filter one `T_tube_tcp` from the reviewed
  cylindrical side-pinch candidate family after a fresh metric tube observation;
- reacquire a metric `T_base_tube` with the moved source tube separated from the
  distractor tube;
- measure a metric rack-entry origin, outward axis, and insertion depth;
- qualification of the live front-RGB tube-axis extractor against the reviewed
  f330 reference (`135.50°` in the image `+v`-down convention); the reference
  and deterministic calibration ID are already recorded. Human-demo replay
  tracked 61/61 frames with `0.479°` MAE and fired at f326 inside f330±5, but
  remains `runtime_gate_eligible=false` until robot-held trials pass;
- a fresh session contact bootstrap at `SIDE_GRASP` (the policy and thresholds
  are compiled; its three unloaded and three positive FORCE6D windows must come
  from that live session);
- labelled slip and jam calibration if those observations are to control motion.
  They remain shadow-only and do not block plan-only.

The current compiled report therefore has
`offline_task_contract_ready=true`, `moveit_plan_only_ready=false`, and
`hardware_execution_ready=false`.

Loading that report through the unified runtime is also tested.  The current
blocked replay stops at `HOLD` from `OBSERVE` with `step_count=0` and
`hardware_commands_dispatched_by_core=false`; no semantic executor call occurs.

The authoritative hardware blockers also include selection of an exact
`T_tube_tcp`, grounded Hand-E widths/close budget, a validated Hand-E/tube/rack
collision scene, the live hardware adapter and predicate producers, support
verification after release, and fresh FR3/D435 data-path health. Motion values
in the offline contract are review bounds until those checks validate them.

The demonstration's raw tactile streams are useful for visual review, but raw
UVC images without robot-mounted tube contact cannot qualify these runtime
predicates. A later tactile calibration may fill the thresholds without
changing the stage machine or authority split.

