# Robot-held test-tube RGB tracker qualification

`scripts/qualify_test_tube_rgb_tracker.py` replays lossless RGB from three
robot-held test-tube trials. It is read-only: it does not open the D435, ROS,
MoveIt, Hand-E, or the FR3. Capture can use the existing long-lived
`D435LiveSource` recording path, but qualification always runs later from the
saved frames and independent JSON labels.

Every new capture made by `scripts/record_front_rgbd_tactile.py` writes a
random `recording_session_id` into its atomically published manifest. Copy that
exact value into each trial's `recording_session_id` annotation. Record three
separate physical trials; copying one directory or relabelling one recording
does not create another trial.

Copy
`examples/test_tube_rgb_qualification_annotations.example.json`, then replace
the three recording paths and reviewed labels. Each trial contains:

The formal input contract is
`examples/test_tube_rgb_qualification_annotations.schema.json`.

- one pivot interval, a reviewed initial tube-edge seed, and the bounded ROI;
- the `recording_session_id` copied from that recording's manifest;
- visibility intervals used as the tracked-fraction denominator;
- an independent tube-axis label at least every five visible frames;
- the reviewed reclamp frame and the earliest frame at which reclamping is
  acceptable;
- one partial-occlusion interval and one distinct distractor-axis interval,
  each replayed from its own reviewed seed.

Run:

```bash
.venv/bin/python scripts/qualify_test_tube_rgb_tracker.py \
  --annotations /path/to/reviewed_annotations.json \
  --reference-config configs/test_tube_front_rgb_pivot_reference_20261023.json \
  --output /path/to/test_tube_rgb_tracker_qualification.json
```

Exit code `0` means all three distinct recordings passed every fixed criterion.
Exit code `2` means the report was written but promotion remains false. Input
contract errors fail without producing an eligible report.

The report checks, per trial:

- at least 95% tracking while the tube is labelled visible;
- axis MAE at most 3 degrees and maximum error at most 5 degrees;
- the first reclamp decision within 3 frames of the reviewed crossing;
- no reclamp before the independently labelled acceptable interval;
- fail-closed output for every reviewed occlusion and ambiguity frame;
- sufficient labels with no interval larger than five frames.

`runtime_gate_eligible` becomes true only when at least three distinct capture
session IDs and three distinct full-recording content fingerprints pass, all
provided trials pass, and trial identifiers are unique. The content fingerprint
is computed from the complete RGB recording, complete hardware timestamp array,
per-frame camera timestamp/frame-number provenance, camera serial, intrinsics,
frame count, and manifest schema. It excludes `trial_id`, all reviewed labels,
the ROI, replay intervals, and selected evidence frames. Therefore changing
annotations, pointing three trials at one directory, copying one recording, or
editing only its session ID cannot satisfy the three-trial rule.

Each report records both the full manifest hash and the content fingerprint.
Treat the capture directory as immutable after review; any content change
requires a new qualification report.

The utility does not edit the learned reference. After promotion, preflight
must pin the qualification report path and SHA-256; the human-held demonstration
replay remains insufficient. Runtime still needs the same fixed D435 serial,
intrinsics and mount, a fresh reviewed seed, and fresh tracker evidence.
