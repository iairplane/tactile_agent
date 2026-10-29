# F2G-Pose two-host inference boundary

This adapter keeps CUDA perception on `5090-lzr` and the time-sensitive FR3
controller on the robot workstation. It is a perception transport only. It
does not import ROS, open a Franka connection, or dispatch motion.

```text
robot workstation                              5090-lzr
-----------------                              -------
fresh aligned D435 RGB-D
target mask + K
client timestamp/frame sequence  -- HTTP -->   one resident F2G estimator
freshness/replay/latency guard    <-- JSON --  pose-only + input diagnostics
T_base_camera @ T_camera_object
temporal/task/tactile qualification
```

The model request uses compressed NPZ because images are arrays. The reply is
plain JSON with the fixed schema
`demo2gripper.pose_observation.v1`; its machine-readable definition is
[`examples/f2g_pose_observation.schema.json`](../examples/f2g_pose_observation.schema.json).
The camera transform is `T_camera_object`, so
`p_camera = T_camera_object @ p_object`, in metres and camera optical axes.

## What the reply proves

Every reply binds these values to one echoed client request:

- client-clock timestamp, sequence and `frame_id`;
- `T_camera_object` and metric object size, or an explicit rejection;
- mask area, bounding box, source and source score;
- all valid mask-depth samples, one-dimensional depth clusters, the selected
  foreground cluster, its size/depth/span, outside-mask-ring depth, and the
  fraction of mask samples matching that neighbouring background;
- preprocessing, model and total server latency;
- backend name, device, source revision and checkpoint SHA-256;
- cylinder-axis sign and roll qualification fields.

F2G-Pose does not publish a calibrated confidence. The service therefore
records `model_confidence: null`; it never invents one. For a test tube it also
publishes `axis_sign_status: unresolved` and
`cylindrical_roll_status: unresolved`. The task controller must identify the
open end from a rim/end observation or temporal task evidence and must either
canonicalize roll or avoid using it.

## Foreground-depth gate

The official F2G preprocessing accepts 50 valid mask-depth pixels. That count
is insufficient for the current transparent tube: in recorded frame `000004`,
many pixels inside the silhouette measure the background seen through or
around the tube.

Before F2G runs, this adapter:

1. clusters valid depths inside the target mask;
2. measures a dilated ring immediately outside the mask;
3. finds compact, sizeable mask-depth clusters;
4. requires a selected cluster to be separated from the ring background;
5. rejects the input when too much mask depth matches that background;
6. zeros all depth except the selected cluster before model preprocessing.

Thus hundreds of valid pixels can still produce
`mask_depth_dominated_by_neighbouring_background`. A rejected observation is
diagnostic evidence, never a control target. Thresholds are visible server
arguments, while every reply records the measured values and exact rejection
reasons. Qualify the thresholds on recorded data before live use.

## GPU-host service

Install the official F2G-Pose checkout, weights, RADIO dependency, and its CUDA
extensions in an isolated environment on `5090-lzr`. The RTX 5090 needs a
PyTorch/CUDA build and locally compiled extensions that support its GPU
architecture; do not copy CUDA extension binaries from another machine.

Sync this repository without embedding credentials, then start the server on
the GPU host:

```bash
ssh 5090-lzr
cd ~/code/tactile_agent
source /path/to/f2g-pose-env/bin/activate
export D2G_POSE_TOKEN='set-the-same-random-token-on-both-hosts'
python scripts/serve_f2g_pose.py \
  --host 127.0.0.1 --port 8765 \
  --checkpoint /path/to/f2g-pose.pt \
  --f2g-root /path/to/F2G-Pose \
  --device cuda:0
```

The default loopback bind keeps the service off the general LAN. On the robot
workstation, create an SSH tunnel using the existing SSH config entry:

```bash
ssh -N -L 8765:127.0.0.1:8765 5090-lzr
```

In another robot-workstation terminal, use the same token and inspect health:

```bash
export D2G_POSE_TOKEN='set-the-same-random-token-on-both-hosts'
curl -H "Authorization: Bearer $D2G_POSE_TOKEN" \
  http://127.0.0.1:8765/health
```

No password is stored by these scripts. SSH authentication remains in the
user's SSH agent/config.

## Offline CPU transport check

This proves serialization, HTTP, diagnostics, and client guards without CUDA
or robot access:

```bash
python scripts/serve_f2g_pose.py \
  --backend fake_pose --allow-fake-backend \
  --host 127.0.0.1 --port 8765
```

Query one recorded frame (depth is already metres in this example):

```bash
python scripts/query_f2g_pose_service.py \
  --endpoint http://127.0.0.1:8765 \
  --rgb /path/to/rgb.npy \
  --depth /path/to/depth_m.npy \
  --mask /path/to/mask.npy \
  --intrinsics /path/to/K.npy \
  --allow-fake-backend \
  --output /tmp/fake_pose_observation.json
```

For millimetre depth PNG, add `--depth-scale 0.001`. The fake backend is named
`fake_pose`; the normal client rejects it unless `--allow-fake-backend` is
explicitly present.

## Live provider call

The long-lived D435/controller process constructs one provider and reuses it:

```python
from demo2gripper.f2g_pose_service import PoseServiceClient, RemotePoseProvider

provider = RemotePoseProvider(
    PoseServiceClient("http://127.0.0.1:8765", timeout_s=0.25)
)
observation = provider.estimate(
    frame_id=f"{frame.serial}:{frame.color_frame_number}",
    capture_timestamp_s=frame.host_received_monotonic_s,
    rgb=frame.rgb,
    depth_m=frame.depth_m,
    K=frame.intrinsics.matrix,
    mask=target_mask,
    mask_source="qualified_target_tracker.v1",
    mask_source_score=target_mask_score,
)
T_camera_object, size_m = observation.require_pose()
T_base_object = T_base_camera @ T_camera_object
```

The provider has no retry loop. It rejects timeout, excess round-trip/server
latency, old capture timestamps, response/request mismatches, repeated or
regressed sequences, disallowed backends, failed mask/depth gates, and rejected
observations. The service also rejects a second request while the single GPU
estimator is busy, so timed-out calls cannot form an unbounded inference queue.
Motion code should HOLD on those exceptions instead of reusing the previous
pose.

## Remaining task-layer work

This service makes object pose available at runtime, but it does not make a
single F2G frame safe for direct Cartesian control. The complete tube loop must
still provide:

- a persistent target mask/identity stream for every RGB-D frame;
- a temporal SE(3) filter with jump rejection and task-level qualification;
- tube open-end sign resolution and cylindrical-roll handling;
- fresh `T_base_camera`, measured `T_base_tcp`, and their timestamp alignment;
- bounded Cartesian servo commands with collision checks and cancellation;
- visual correction only while the fixed camera still sees the tube;
- a transition to tactile contact/slip/tilt/jam predicates after gripper
  occlusion makes visual pose unreliable.

The current fixed-camera transparent tube can legitimately fail the depth gate.
In that case the controller needs a qualified opaque/fiducial anchor,
multi-view/stereo estimate, or a touch/tactile refinement. Lowering the gate to
accept background depth would make the apparent pose real-time while leaving
it physically wrong.
