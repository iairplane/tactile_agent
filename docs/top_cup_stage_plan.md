# Top-cup stage-plan compiler

`scripts/compile_top_cup_stage_plan.py` combines three reviewed JSON artifacts:

- `demo2gripper.cup_rim_relation.v1`: the frozen pre-contact cup-rim frame and the
  demonstrated pinch endpoint relative to that rim;
- `demo2gripper.image_only_semantic_stages.v1`: the reviewed one-shot phase split;
- `demo2gripper.tactile_supervision_config.v1`: the tactile predicates and qualification
  state.

It writes `demo2gripper.top_cup_stage_plan.v3` with this fixed order:

```text
OBSERVE -> GROUND_TOP_RIM -> APPROACH -> RIM_TOUCH_ALIGN -> GENTLE_PINCH
-> SEPARATE -> VERIFY_ONE -> CLEAR_LIFT -> TRANSPORT -> PLACE -> RELEASE -> RETREAT
```

Only `RIM_TOUCH_ALIGN` consumes the demonstrated `T_rim_human_pinch_tcp` endpoint.
`APPROACH`, `CLEAR_LIFT`, `TRANSPORT`, `PLACE`, and `RETREAT` explicitly request new
online agent/MoveIt plans. `GENTLE_PINCH` is a discrete tactile probe at a fixed rim pose;
`SEPARATE` follows live cup-axis task geometry. The pre-lift gate accepts only a stable
`INNER_OUTER_RIM_MATCH` tactile result: one finger must match its calibrated
inner-rim template and the other must match its calibrated outer-rim template.
The two templates may have different image areas, shapes, positions, and expected loads;
there is no left/right image-symmetry gate. `VERIFY_ONE` is retained as the historical stage name,
but now verifies tactile hold continuity after the micro-lift; it does not count cups from
front RGB-D. No demonstrated trajectory is replayed.

The task-specific state machine is
[`src/demo2gripper/cup_tactile_gate.py`](../src/demo2gripper/cup_tactile_gate.py). It maps
`NO_CONTACT` to a bounded downward probe, `MULTI_RIM_OR_TOO_DEEP` to unload then a
release-authorized upward correction, `ROLE_MISMATCH` to unload then a release-authorized
lateral/angular correction, `INNER_OUTER_RIM_MATCH` to lift permission, and `UNKNOWN`
to stop. The tactile policy must assign exactly one `inner` and one `outer` role to the
current left/right probe hypothesis; that mapping is frozen within a probe context.

Example:

```bash
python scripts/compile_top_cup_stage_plan.py \
  --cup-rim-relation runs/.../cup_rim_relation.json \
  --semantic-stages runs/.../analysis_image_stages/semantic_stages.json \
  --tactile-policy configs/nested_cup_tactile_shadow.example.json \
  --output runs/.../top_cup_stage_plan.json
```

The compiler always emits `review_only`, `execution_authorized=false`, and
`hardware_execution_enabled=false`. The example tactile policy is also shadow-only.
The result is an auditable stage contract, not an input to the current robot runner.
