# Live test-tube tactile interpreter

`demo2gripper.test_tube_tactile_interpreter:create_test_tube_tactile_interpreter`
is the concrete `TactileInterpreter` factory consumed by `test_tube_live_io`.
It keeps one contact gate and one tactile-controller memory for the full run;
it does not open a GF220, Hand-E, or robot handle.

Use the live-config fragment in
`examples/test_tube_tactile_interpreter.config.json`. The referenced
qualification bundle is deliberately a draft. It must be produced from the
same live GF220 session because the positive-contact profile is bound to the
two current unloaded-baseline calibration IDs and to strictly later sample
timestamps. Changing either baseline invalidates the bundle.

## State mapping

| Closed-loop states | Tactile mode | Promoted output |
| --- | --- | --- |
| `OBSERVE`, `APPROACH_TUBE`, `SIDE_GRASP` | `grasp` | contact, bilateral contact, single-pad side |
| `EXTRACT`, `APPROACH_SOCKET`, `ALIGN_SOCKET`, `HOLD` | `hold` | retained contact and qualified slip |
| `PIVOT_MICRO_LOOSEN`, `PIVOT_EVALUATE` | `pivot` | calibrated tube-axis target |
| `RECLAMP`, `RECLAMP_RECOVERY` | `reclamp` | axis target plus stationary-axis debounce |
| insert and jam-recovery states | `insert` | resistance-high and confirmed jam |
| release, retreat and complete states | `release` | contact absence after opening |

The controller still logs unqualified slip, axis and jam **candidates** as
shadow evidence. Only predicates promoted by an accepted runtime profile enter
`TactileFeedback`.

## Fail-closed boundary

The factory returns an interpreter with `valid=false` when the bundle is
missing, draft, rejected, structurally invalid, or tied to different sensor or
calibration IDs. A qualified interpreter also returns `valid=false` in these
stage-local cases:

- either pad lacks a positive-contact response qualification;
- pivot/reclamp lacks the pad-axis or rotation-stop qualification;
- insertion lacks an accepted jam profile or FORCE6D wrench transform;
- a window is stale, replayed, unsynchronized, or fails adapter provenance.

That `valid=false` is consumed by the closed-loop policy as HOLD. It is not
replaced with a guessed force threshold, assumed pad axis, or guessed jam
limit.

## Evidence

Every call writes
`<run-output>/tactile_interpreter/NNNNNN.json` and atomically refreshes
`latest.json`. The receipt preserves:

- source controller evidence ID and invalid reason;
- active serials, contact presence and bilateral status;
- each pad's contact area, centre, axis and residual statistic;
- slip candidate and promoted/shadow status;
- target-axis error, inter-window rotation delta and stop debounce count;
- insertion progress, resistance, threshold and jam candidate;
- the final `TactileFeedback` and `hardware_commands_dispatched=false`.

The template does not qualify the hardware. Its zero values and `draft` status
are placeholders and are intentionally rejected by the factory.
