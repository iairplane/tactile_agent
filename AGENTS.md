# AGENTS.md

## Mission

Bring this repository up on a new Ubuntu workstation connected to the Franka at
`172.16.0.2`, then continue the pipeline from live RGB-D perception to a real-robot
grasp. Work in three explicit levels:

1. Core geometry and synthetic smoke test pass.
2. Live RGB-D produces `T_camera_object.npy` and a reviewed Franka plan JSON.
3. A compatible Franka driver executes the reviewed plan at low speed.

Do not report level 3 complete when the repository has only generated a plan. The
current repository does not contain a real-robot executor.

## Current system boundary

The implemented path is:

```text
calibrated RGB-D human demonstration
  -> SAM 3 target discovery
  -> SAM 2.1 hand/object video masks
  -> RTMPose-m Hand5 joints
  -> WAPR or FoundationPose T_camera_object
  -> object-frame pinch or power-grasp intent
  -> GraspGenX Panda candidates
  -> selected T_object_gripper_tcp
  -> Isaac validation or Franka target-plan JSON
```

What is already implemented:

- Core object-frame intent inference and candidate ranking.
- Disk adapters for SAM 3, SAM 2.1, RTMPose, WAPR, FoundationPose and GraspGenX.
- Isaac Lab `PREGRASP -> APPROACH -> CLOSE -> LIFT -> HOLD` validation.
- Stock Panda Hand target composition in `src/demo2gripper/franka.py`.
- Plan generation in `scripts/prepare_franka_panda_plan.py`.

What is not implemented:

- A live RGB-D camera driver.
- Hand-eye calibration acquisition.
- Panda IK/collision/force-closure filtering before final candidate selection.
- A libfranka/franka_ros2/MoveIt executor or gripper command sender.
- Splat2Act/3DGS scene reconstruction.

The existing Panda conversion assumes a stock Franka Hand, command frame
`panda_hand`, `0.1034 m` hand-to-TCP offset and a fixed `Rz(-90 deg)` frame rotation.
Do not use that mapping for a Robotiq or another end effector; measure its TCP and
implement the corresponding gripper contract first.

## Repository map

- `src/demo2gripper/`: geometry, intent, selection, I/O and Franka target transforms.
- `scripts/`: backend adapters, observation construction, visualization and validation.
- `configs/backends.lock.json`: upstream repositories and revisions used by this project.
- `docs/backends.md`: backend-specific installation and commands from the original host.
- `docs/perception_io.md`: RGB-D and model-output disk contracts.
- `docs/isaaclab.md`: the verified Isaac Sim 5.1 / Isaac Lab path.
- `tests/`: CPU unit tests for transforms, intent and candidate conversion/selection.
- `examples/`: small tracked inputs only.

`third_party/`, model weights, datasets, `runs/`, videos and the original `/mnt/sdb`
artifacts are intentionally absent from Git. `scripts/clone_backends.py` only clones
and checks out source revisions; it does not install dependencies, download weights or
accept model licenses.

## Scope and working rules

- Keep changes surgical. Fix failures that occur on the supported path; do not add
  compatibility frameworks, feature flags or defensive wrappers for hypothetical hosts.
- Keep the core, perception, pose, grasp and robot-driver environments separate. They
  have incompatible Python, CUDA and ROS requirements and exchange only files.
- Treat absolute `/home/lzr/...`, `/mnt/sdb/...` paths and physical GPU 7 in old run
  commands as evidence from the original machine, not configuration for the new host.
- Do not commit model weights, datasets, generated videos, ROS logs, build trees or
  machine-specific calibration files.
- Preserve `configs/backends.lock.json` unless intentionally upgrading a backend and
  rerunning its real output check.
- GT/oracle object poses may be used only for an explicitly labelled isolated simulator
  diagnostic. They must not enter the perception-driven or real-robot control input.
- Before a check, know which failure it can distinguish and what action follows. Do not
  rerun settled checks without new evidence.

## Coordinate and data contracts

All transforms use right-handed coordinates, metres, column vectors and
`T_parent_child`:

```text
p_parent = T_parent_child @ [p_child.x, p_child.y, p_child.z, 1]
T_base_object = T_base_camera @ T_camera_object
T_base_tcp = T_base_object @ T_object_gripper_tcp
```

Required live perception inputs:

```text
rgb/NNNNNN.png
depth/NNNNNN.npy       # float depth in metres, aligned to RGB
K.npy                  # OpenCV 3x3 intrinsics
```

Core observation and grasp contracts:

- `T_camera_object.npy`: `(4,4)` or `(F,4,4)`, object to optical camera.
- `hand_joints_camera_m.npy`: `(F,21,3)` in the same camera frame.
- `hand_joint_confidence.npy`: `(F,21)`.
- `timestamps_s.npy`: `(F,)`, strictly increasing.
- `candidates.npz/poses`: `(K,4,4)` `T_object_gripper_tcp` transforms.
- `selected_grasp.json`: selected object-frame TCP, widths and approach distance.

The canonical TCP has local `+X` along the jaw closing axis and local `+Z` pointing
toward the object. Pregrasp retreats along local `-Z`; lift moves along base `+Z`.

## New-host bring-up order

### 1. Inventory before choosing packages

Record the following in the work report, not in a committed machine-specific file:

```bash
lsb_release -a
uname -a
python3 --version
nvidia-smi -L
nvidia-smi
echo "${ROS_DISTRO:-unset}"
ip -br link
ip -br addr
ip route get 172.16.0.2
```

Also establish from Franka Desk:

- Robot model: Panda or FR3.
- Robot System version and whether FCI is installed/enabled.
- Stock Franka Hand, Robotiq or another end effector.
- Whether the camera is fixed in the workspace or eye-in-hand.
- RGB-D camera model and whether its SDK supplies depth aligned to RGB.

These facts determine the driver branch and calibration equation. Do not choose the
latest libfranka merely because it is latest; select the version compatible with the
robot's System version using the official Franka compatibility documentation.

### 2. Core CPU environment

The core requires Python 3.11 or newer:

```bash
cd /path/to/tactile_agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,mesh,visualization]'
pytest -q
demo2gripper make-synthetic --output-dir runs/demo
demo2gripper run --config configs/pipeline.example.json
```

Expected result: all tests pass and the synthetic run writes
`runs/demo/grasp_intent.json` and `runs/demo/selected_grasp.json`. This verifies only
the CPU geometry/data path, not CUDA perception or robot control.

### 3. GPU selection on the new host

The original host reserved physical GPU 7. Select an available physical GPU on the new
host instead:

```bash
export D2G_GPU=0
CUDA_VISIBLE_DEVICES="$D2G_GPU" <command> --device cuda:0
```

After `CUDA_VISIBLE_DEVICES` isolates one card, PyTorch, ONNX Runtime and Isaac see it as
`cuda:0`; do not pass the original physical number as the in-process device.

### 4. Locked backend sources and isolated environments

List and clone only the backends needed for the next milestone:

```bash
source .venv/bin/activate
python scripts/clone_backends.py --list
python scripts/clone_backends.py \
  --root third_party/locked \
  --names segmentation hand_video_segmentation hand_pose known_object_pose grasp_candidates
```

Add `known_object_pose_fallback` only if FoundationPose is required. Use the actual
directories under `third_party/locked/`; older docs show original-host directory names.

Maintain these environment boundaries:

| Environment | Verified baseline | Purpose |
|---|---|---|
| `.venv` | Python >=3.11 | Core geometry, plan generation, tests |
| SAM 3 | Locked upstream revision | Text-prompted object mask |
| perception | Python 3.10, SAM 2.1, rtmlib 0.0.16, ONNX Runtime GPU 1.23.2 | Hand/object masks and Hand5 joints |
| WAPR | Python 3.10, torch 2.7.1 + CUDA 12.8 baseline | Primary known-CAD pose |
| FoundationPose | Python 3.11, torch 2.7.0 + CUDA 12.8 baseline | Pose fallback |
| GraspGenX | Python 3.10, torch 2.7.1 + CUDA 12.8 baseline | Panda grasp candidates |
| Franka/ROS | OS packages plus a colcon workspace | Driver, MoveIt and gripper actions |
| Isaac/RoboDojo | Existing verified Isaac Sim 5.1 stack, optional on the robot host | Simulation only |

Install each locked checkout according to its upstream instructions and
`docs/backends.md`. Download checkpoints separately after accepting their license terms.
If the new GPU architecture differs, rebuild FoundationPose CUDA extensions locally;
do not copy the original host's `sm_120` binaries. WAPR may use `WAPR_BACKEND=torch`
without TensorRT.

### 5. Franka network and compatible driver

The robot address is `172.16.0.2`. The directly connected host interface should have an
address such as `172.16.0.1/24`:

```bash
ip -br link
sudo ip link set dev <ROBOT_NIC> up
sudo ip addr replace 172.16.0.1/24 dev <ROBOT_NIC>
ip route get 172.16.0.2
ping -c 3 172.16.0.2
```

The route must use `<ROBOT_NIC>` rather than the workstation's default gateway. Do not
change the default route.

Install a libfranka version compatible with the observed Robot System version. Prefer
the official `franka_ros2` + MoveIt 2 route when the host OS/ROS and robot are supported;
do not implement a new raw torque controller for this pipeline. Useful primary sources:

- <https://frankarobotics.github.io/docs/compatibility.html>
- <https://github.com/frankarobotics/libfranka>
- <https://github.com/frankarobotics/franka_ros2>

First prove read-only robot-state communication with the matching official example.
Then prove the vendor's standard joint-position/MoveIt example independently of this
repository. Keep the exact libfranka, franka_ros2, ROS, Ubuntu and Robot System versions
in the run report.

### 6. Camera and hand-eye calibration

Capture synchronized RGB plus depth already aligned to RGB, save depth in metres, and
export the actual `K.npy`. Do not use the Imitrob approximate pixel shift on live camera
data.

For a fixed external camera, calibrate and save:

```text
calibration/T_base_camera.npy
```

For an eye-in-hand camera, combine the calibrated hand-to-camera transform with the
current robot FK at the capture timestamp to produce a fresh `T_base_camera` for every
observation. The current plan script accepts the resulting matrix; it does not perform
that time synchronization itself.

Run the mask and pose adapters, then inspect their saved mask/pose overlay before using
the pose. The live output required by the plan stage is `T_camera_object.npy`.

### 7. Generate the current real-robot plan boundary

```bash
source .venv/bin/activate
python scripts/prepare_franka_panda_plan.py \
  --grasp-json runs/real/selected_grasp.json \
  --t-base-camera calibration/T_base_camera.npy \
  --t-camera-object runs/live/T_camera_object.npy \
  --frame-index -1 \
  --lift-height-m 0.03 \
  --output runs/live/franka_plan.json
```

Review `T_base_object`, the canonical TCP targets and the `panda_hand` targets in the
output. `execution_order` is descriptive metadata; `force_grasp` is not currently an
implemented controller command. A selected grasp is not proof of Panda reachability.

Before real execution, insert robot-side filtering in this order:

```text
human-intent/GraspGenX top K
  -> Panda IK
  -> self/environment collision
  -> approach trajectory
  -> gripper/contact feasibility
  -> final candidate
```

### 8. Implement and stage the missing executor

Use the compatible ROS 2/MoveIt stack to consume `panda_hand_targets` from the reviewed
plan JSON. Send pregrasp, grasp and lift as planned Cartesian/joint trajectories and use
the official Franka gripper action for open/grasp. Keep perception and robot execution
as separate processes joined by the plan file until the first real grasp succeeds.

Stage the first motion with the operator present, the emergency stop reachable, a clear
workspace, low speed and conservative contact/grasp settings:

1. Read robot state only.
2. Plan in RViz without execution.
3. Execute pregrasp only.
4. Execute approach without closing.
5. Test gripper open/close away from the object.
6. Execute grasp with a short lift.

Do not automatically unlock joints, enable FCI or start physical motion. Those are
operator actions. Once the operator explicitly authorizes a stage, execute it and save
the planned and measured poses plus the outcome.

## Definition of done

- **Level 1:** core environment installs, tests pass, and the synthetic intent/candidate
  files are produced.
- **Level 2:** a real aligned RGB-D frame produces a visually checked object mask,
  `T_camera_object.npy`, and `franka_plan.json` using a measured `T_base_camera`.
- **Level 3:** the compatible driver reports robot state, MoveIt finds collision-free
  trajectories, the stock hand executes open/grasp, and the object is lifted and held.

Report the first failing boundary directly. Do not replace a failed physical result with
a generated visualization or simulator-oracle result.
