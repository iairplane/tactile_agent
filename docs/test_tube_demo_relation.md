# Test-tube demonstration relation recovery

`scripts/recover_test_tube_demo_relation.py` is the offline boundary between
the human RGB-D demonstration and the continuous test-tube controller. It does
not open any hardware. It writes one receipt that distinguishes observable
constraints, a diagnostic reconstruction, and an execution-qualified relation.

## Relation used by the task

The human demonstration supplies the grasp location and topology:

- tube `+Z` is signed from the closed end to the open end;
- the TCP origin is on the tube centreline at the demonstrated axial fraction;
- the Hand-E jaw axis is perpendicular to the tube axis;
- the thumb/index direction selects only the sign of that jaw axis, which is a
  gauge for a symmetric parallel-jaw gripper.

The operator's current requirement supplies the approach constraint. Let
`d = [0, 0, -1]` in the FR3 base and let `a` be the signed tube axis. The
canonical TCP tool direction and jaw closing direction are

```text
z_tcp = d
x_tcp = normalize(z_tcp cross a)
y_tcp = z_tcp cross x_tcp
```

The sign of `x_tcp` is chosen to agree with the demonstrated thumb/index
direction. If `d` and `a` are parallel, the projected thumb/index direction
supplies the free jaw yaw. This keeps the Hand-E tool axis and the final motion
strictly downward while keeping its closing axis perpendicular to the tube.
It resolves the otherwise free cylindrical roll deterministically.
The free-space pregrasp is still vertically above the target, so the final path
arrives from base `-Z`.

At runtime, each fresh tube pose instantiates the same relation:

```text
T_base_tcp = T_base_tube_live @ T_tube_tcp_demo_grasp
```

## What the current recording recovers

For `test_tube_demo_20261023_02`, frame 246 is bracketed by RTMPose frames 245
and 250. The script interpolates only between those observations and samples
aligned depth at each fingertip. The resulting midpoint and depth patch
statistics are stored with the receipt. The legacy image review already gives
the signed tube endpoints and axial fraction `0.6643 +/- 0.0898`.

The transparent tube surface itself has no trustworthy D435 depth. The script
therefore also fits a **diagnostic** metric candidate from endpoint rays, the
fingertip midpoint, and the nominal length prior. That fit is useful for an
overlay and for checking gross consistency. It is never promoted to execution:
the midpoint of anatomical fingertip keypoints is not proven to lie on the
tube centreline, and the `0.11 +/- 0.02 m` length is a visual estimate.

Run the offline recovery with:

```bash
cd /home/vitai/lzr/tactile_agent
.venv/bin/python scripts/recover_test_tube_demo_relation.py \
  --demo-dir runs/live/test_tube_demo_20261023_02 \
  --overwrite
```

Outputs:

- `analysis/tube_demo_relation_receipt.json`: constraints, uncertainty,
  candidate, exact preflight blockers, and optional runner fragment;
- `analysis/tube_demo_relation_receipt_review.png`: one-frame visual review;
- `analysis/tube_demo_relation_review.template.json`: operator review template;
- `analysis/tube_demo_pose_receipt.template.json`: independent metric-pose
  receipt template.

The current output intentionally has `T_tube_tcp_demo_grasp: null` and
`runtime_qualified: false`.

## One-time qualification

A full relation is generated only when both optional inputs pass:

```bash
.venv/bin/python scripts/recover_test_tube_demo_relation.py \
  --demo-dir runs/live/test_tube_demo_20261023_02 \
  --human-review /path/to/completed_relation_review.json \
  --tube-pose-receipt /path/to/qualified_same_frame_pose.json \
  --output /path/to/qualified_relation.json
```

The human review confirms the image labels, fingertip keypoints, and top
approach. It cannot create metric geometry. The independent pose receipt must
contain all of the following:

- a same-frame `T_camera_tube` whose origin is the tube geometric centre;
- signed `+Z` from closed end to open end;
- measured tube length and uncertainty at most 5 mm;
- position standard deviation at most 10 mm and orientation standard deviation
  at most 8 degrees;
- non-empty provenance and `runtime_qualified: true`.

When these checks pass, `runner_geometry_fragment` contains the exact
`T_tube_tcp_demo_grasp`, relation ID, source, and qualification flag accepted by
the closed-loop runner. A partial receipt raises instead of substituting an
identity transform.

## Pivot and occlusion

F2G or another live pose estimator can update the tube pose before contact. It
does not need to remain visible inside the fingers. During the demonstrated
micro-release/pivot, tactile keeps contact and detects axial slip while front
RGB measures the relative tube-axis change until that visual primitive has been
calibrated to tactile. At tactile-confirmed reclamp, the controller freezes the
measured held relation:

```text
T_tube_tcp_held = inverse(T_base_tube_last_qualified) @ T_base_tcp_measured
```

This update preserves the demonstrated axial grasp coordinate unless the
tactile slip primitive says it changed. Force direction alone is not treated
as a tube angle, and an occluded F2G prediction is not required after contact.
