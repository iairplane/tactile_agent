# Test-tube runtime boundary

`demo2gripper.test_tube_runtime` is the hardware-independent orchestration
layer for the test-tube task. It consumes the state machine produced by
`compile_test_tube_preflight`, asks an injected `PredicateProvider` for one
fresh evidence snapshot, calls `decide_test_tube_transition`, and hands one
semantic action to an injected `SemanticActionExecutor`.

The core and its replay CLI import no ROS, MoveIt, libfranka, Hand-E, camera,
or tactile SDK. `scripts/replay_test_tube_runtime.py` has no hardware mode.
A future live adapter may implement the two protocols, including a provider
backed by the serial-pinned, long-lived `D435LiveSource`, but that adapter is
outside this module.

## Start gate

The runtime refuses to consume the first predicate unless all of these are
true in the compiled plan:

- `execution_authorized` and `hardware_execution_enabled`;
- every offline, metric-geometry, MoveIt plan-only, runtime-transition, and
  hardware-execution preflight gate;
- every explicit `execution_readiness` item, including the long-lived front
  RGB source and healthy live stream;
- qualified contact, angle, and insertion-jam gates.

It also requires a fresh `demo2gripper.test_tube_runtime_geometry.v1` bundle
with `T_base_tube`, `T_base_socket`, the selected `T_tube_tcp_side_grasp`, the
insertion axis and measured depth, collision-scene provenance, and the live
contact/vision/jam profile identifiers. A replay JSON cannot promote a plan
that lacks any of those facts.

The current compiler intentionally writes `execution_authorized=false`,
`hardware_execution_enabled=false`, and `hardware_execution_ready=false`.
Consequently, the current compiled task ends in `blocked_preflight/HOLD` at
this boundary. The synthetic ready-plan fixture in the unit tests exercises
the complete state machine without weakening that real boundary. A future
audited authorization step must produce the qualified compiled plan; editing
only nested preflight booleans is insufficient.

## Receipts and budgets

The runtime accepts only predicate snapshots carrying a
`demo2gripper.test_tube_predicate_receipt.v2` receipt. The receipt is generated
by `TestTubePredicateBuilder` (normally through `TestTubePredicateProvider`) and
is checked again at the orchestration boundary. In particular, the runtime:

- accepts `TestTubePredicateProvider` for live production and `JsonReplay` for
  offline verification, rejecting arbitrary protocol implementations before
  their `observe` method is called; a `JsonReplay` provider must also be the
  action executor, so replay evidence cannot be paired with a hardware adapter;
- requires the compiled `predicate_producer_manifest` to match the canonical
  builder manifest;
- recomputes `evidence_id` as the SHA-256 digest of the complete builder
  receipt and checks the receipt's manifest digest;
- matches task, state, timestamp, promoted predicates, and the state-specific
  required producer set;
- binds tactile and insertion receipts to the live geometry's contact and jam
  profile IDs;
- pins the builder session and provenance IDs on the first accepted snapshot
  and rejects either changing during the run.

Arbitrary provider dictionaries, unmanifested predicates, predicates emitted
in the wrong state, missing producer receipts, and edited replay evidence all
lead to a persisted, non-dispatched `HOLD`. Shadow observations remain inside
the receipt and cannot become control predicates.

Every step uses a two-phase receipt in `steps/NNNN.json`:

1. evidence, decision, and dispatch intent are atomically persisted;
2. only then is the injected semantic executor entered;
3. its result and the applied next state atomically replace the pending
   intent.

An abrupt failure during an action therefore leaves a visible pending intent
in `runtime_report.json`. Provider failures are recorded as not dispatched and
go directly to `HOLD`.

The runtime owns all budget predicates. Providers cannot assert remaining
budget. It counts bounded close probes, pivot microsteps, recovery reclamps,
jam recoveries, insertion microsteps, and total runtime steps. Exhaustion leads
to a persisted terminal `HOLD`.

## Replay

```bash
python scripts/replay_test_tube_runtime.py \
  --plan /path/to/compiled_plan.json \
  --replay /path/to/runtime_replay.json \
  --output-dir /path/to/replay_receipts
```

The replay root schema is `demo2gripper.test_tube_runtime_replay.v1`. Each
step fixes its state, evidence ID, timestamp, promoted predicates, complete
predicate-builder receipt, expected event/action/next state, and semantic
action result. JSON replay therefore crosses the same manifest, provenance,
producer-requirement, and receipt-digest boundary as a live provider. State,
timestamps, evidence IDs, expected decisions, and complete trace consumption
are all checked deterministically.
