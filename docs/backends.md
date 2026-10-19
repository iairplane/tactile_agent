# 后端安装与输出适配

版本以 [`../configs/backends.lock.json`](../configs/backends.lock.json) 为准。本文只使用上游已经公开的命令和字段；上游没有稳定批量导出格式的地方，明确标出需要添加的导出钩子。

本项目固定只使用 **GPU 7**。所有 CUDA 推理、GraspGenX 与 Isaac Lab 命令必须保留 `CUDA_VISIBLE_DEVICES=7` 前缀，此时进程内设备是 `cuda:0`。SAM 3、SAM 2.1/RTMPose、WAPR、FoundationPose fallback、GraspGenX 和 Isaac Lab 保持环境隔离；交换文件统一写到 `runs/<sequence>/`。

## 1. SAM 3 / 3.1：文本目标 mask

当前 Isaac 对象感知使用 [facebookresearch/sam3](https://github.com/facebookresearch/sam3) 官方图像 API。`segment_sam3.py` 调 `build_sam3_image_model` 和 `Sam3Processor`，以固定类别文本对每帧产生候选，并保存最高分目标的 mask、box 和 score。脚本没有 GT box/mask 输入。

本机实际证据使用 SAM 3 commit `6fe87d64`、已有 `sam3.pt` 和提示 `pudding box`：

```bash
cd /home/lzr/code/demo2gripper
CUDA_VISIBLE_DEVICES=7 \
  /mnt/sdb/lzr/univtac/venvs/cap-x-agent0/bin/python scripts/segment_sam3.py \
  --rgb-dir /mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/input/rgb \
  --checkpoint /mnt/sdb/lzr/univtac/models/sam3/sam3.pt \
  --model-root /mnt/sdb/lzr/univtac/worktrees/cap-x-agent0/capx/third_party/sam3 \
  --text-prompt 'pudding box' \
  --output-dir /mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/sam3 \
  --device cuda:0
```

SAM 3.1 是 `frontier_segmentation_optional` 锁定的最新目标。它使用同一上游仓库和适配器，但 Hugging Face checkpoint 是 gated；接受 SAM License 并登录后才能切换。当前产物应标记为 SAM 3，不能标记为 SAM 3.1。

### SAM 2.1：人手/部件视频 mask

SAM 2.1 保留给人手示范中的交互式多目标传播。`segment_sam2.py` 使用首帧点/框为手、目标本体和独立运动部件分配不同 `obj_id`，再调用 `propagate_in_video`。其安装、提示 JSON 与完整磁盘格式见 [`perception_io.md`](perception_io.md)；它不是本次 Isaac pudding-box 目标 mask 的来源。

## 2. RTMPose-m Hand5：默认手部 21 点 + RGB-D

默认手后端是 OpenMMLab 的 RTMPose-m Hand5 ONNX，使用 [rtmlib](https://github.com/Tau-J/rtmlib) `0.0.16` 和 `onnxruntime-gpu` `1.23.2`。官方模型来自：

```text
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip
```

本机模型入口为 `/mnt/sdb/lzr/demo2gripper/models/rtmpose-m-hand5.onnx`。适配器不再运行第二个 hand detector：它从每帧 SAM 手 mask 的非零范围得到 bbox，RTMPose 在该框内输出 `(21,2)`，然后在 mask 内为每个关节采样对齐深度并用真实 `K` 反投影。因此 `hand_joints_camera_m.npy` 是逐关节 RGB-D 抬升，不是 MANO 单目 3D 重建。

实际入口是：

```bash
cd /home/lzr/code/demo2gripper
D2G_NVIDIA_SITE=/mnt/sdb/lzr/univtac/conda_envs/UniVTAC/lib/python3.10/site-packages/nvidia
CUDA_VISIBLE_DEVICES=7 \
LD_LIBRARY_PATH="${D2G_NVIDIA_SITE}/cudnn/lib:${D2G_NVIDIA_SITE}/cublas/lib:${D2G_NVIDIA_SITE}/cuda_runtime/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  /mnt/sdb/lzr/demo2gripper/envs/perception/bin/python \
  scripts/track_rtmpose_rgbd.py \
  --rgb-dir /mnt/sdb/lzr/demo2gripper_runs/dexycb_pudding_view06/input/rgb \
  --depth-dir /mnt/sdb/lzr/demo2gripper_runs/dexycb_pudding_view06/input/depth \
  --intrinsics /mnt/sdb/lzr/demo2gripper_runs/dexycb_pudding_view06/input/K.npy \
  --hand-mask-dir /mnt/sdb/lzr/demo2gripper_runs/dexycb_pudding_view06/segmentation/masks/1 \
  --model /mnt/sdb/lzr/demo2gripper/models/rtmpose-m-hand5.onnx \
  --output-dir /mnt/sdb/lzr/demo2gripper_runs/dexycb_pudding_view06/hand_pose \
  --backend onnxruntime --device cuda --fps 10
```

输入 RGB、米制 depth `.npy` 与手 mask 必须同 stem。输出包括原始 2D 关节/SimCC 分数、米制 3D 关节、`[0,1]` 关节 confidence、时间戳和逐帧 manifest；完整合同见 [`perception_io.md`](perception_io.md)。

### 可选：AnyHand-WiLoR / MANO

`hand_reconstruction_optional` 锁定 AnyHand-WiLoR，供确实需要 MANO pose、shape 或 mesh 的实验使用，不是默认抓取链路。该方案需要按上游条款手工取得 `MANO_RIGHT.pkl` 和模型权重；`track_wilor_rgbd.py` 仍保留其磁盘适配器。默认 RTMPose 路径不下载 MANO，也不依赖 MANO 几何。

## 3. WAPR：已知 CAD 物体位姿

默认后端是 [WangYuLin-SEU/WAPR](https://github.com/WangYuLin-SEU/WAPR)，固定提交
`f165cec8ebdd48ae6f46d06a46c40d796ab80b71`。它从 RGB-D、OpenCV 内参、目标 mask 和米制 CAD 估计 object-to-camera 位姿；默认 recipe 使用 12 个初始姿态、WAPR/SAPR refinement 和 WBPS 排序。源码为 LGPL-2.1-only，四份一方权重 `wapr_w_mask.pth`、`wapr_wo_mask.pth`、`sapr.pth`、`wbps.pth` 为 CC BY-ND 4.0。

本机 Python 3.10 环境位于 `/mnt/sdb/lzr/demo2gripper_envs/wapr`。设置 `WAPR_BACKEND=torch` 可直接用 PyTorch 执行同一模型，不要求 TensorRT；TensorRT 只用于加速。主入口为：

```bash
cd /home/lzr/code/demo2gripper
CUDA_VISIBLE_DEVICES=7 \
WAPR_CACHE_DIR=/mnt/sdb/lzr/demo2gripper_cache/wapr \
WAPR_BACKEND=torch \
  /mnt/sdb/lzr/demo2gripper_envs/wapr/bin/python scripts/estimate_wapr_pose.py \
  --rgb /mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/input/rgb/000000.png \
  --depth /mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/input/depth/000000.npy \
  --intrinsics /mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/input/K.npy \
  --object-mask /mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/sam3/masks/1/000000.png \
  --mesh /mnt/sdb/lzr/demo2gripper_data/assets/008_pudding_box/google_16k/textured.obj \
  --mesh-scale 1.0 \
  --output-dir /mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/wapr \
  --device cuda:0
```

输出 `T_camera_object.npy`、`wapr_manifest.json` 和 `pose_overlay.png`。`T_camera_object` 是米制 CAD frame 到 OpenCV camera frame 的变换；完整合同与世界系合成见 [`perception_io.md`](perception_io.md)。

该命令已在 GPU 7 上实跑。当前结果的 WAPR inference 为 `1.037 s`，世界系误差为 `1.255 mm / 0.498 deg`；其外部估计在 Isaac 中驱动 candidate 7 后通过 IK、双指接触、`74.31 mm` 抬升和最终保持。估计与控制均未读取仿真物体真值，真值仅由独立评估脚本读取。

### FoundationPose fallback

已实跑 fallback 使用 [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) 原版 PyTorch 实现，提交号见锁文件；代码、数据和权重受限非商业研究/评估的 NVIDIA Source Code License。本机环境是 Python 3.11.16、torch `2.7.0+cu128`、torchvision `0.22.0+cu128`、nvdiffrast `0.4.0` 和 PyTorch3D `0.7.9`；CUDA 扩展针对 RTX 5090 的 `sm_120` 构建。上游 refiner 与 scorer 权重目录分别为 `2023-10-28-18-33-37`、`2024-01-11-20-02-45`。

本仓库的 `track_foundationpose_original.py` 用首帧 RGB-D、目标 mask 和 CAD 调 `FoundationPose.register`，后续帧调 `track_one`。实际入口为：

```bash
cd /mnt/sdb/lzr/demo2gripper_third_party/FoundationPose
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=/mnt/sdb/lzr/demo2gripper_third_party/FoundationPose/mycpp/build \
  /mnt/sdb/lzr/demo2gripper_envs/foundationpose/bin/python \
  /home/lzr/code/demo2gripper/scripts/track_foundationpose_original.py \
  --rgb-dir /mnt/sdb/lzr/demo2gripper_runs/dexycb_pudding_view06/input/rgb \
  --depth-dir /mnt/sdb/lzr/demo2gripper_runs/dexycb_pudding_view06/input/depth \
  --intrinsics /mnt/sdb/lzr/demo2gripper_runs/dexycb_pudding_view06/input/K.npy \
  --object-mask-dir /mnt/sdb/lzr/demo2gripper_runs/dexycb_pudding_view06/segmentation/masks/2 \
  --mesh /mnt/sdb/lzr/demo2gripper_data/assets/008_pudding_box/google_16k/textured.obj \
  --mesh-scale 1.0 \
  --output-dir /mnt/sdb/lzr/demo2gripper_runs/dexycb_pudding_view06/foundationpose \
  --device cuda:0 \
  --registration-refine-iterations 5 \
  --tracking-refine-iterations 2
```

输出是 `(F,4,4)` 的 `T_camera_object.npy` 和 `foundationpose_manifest.json`。manifest 记录首帧 `registration_score_raw` 与逐帧耗时；原版 `track_one` 不返回逐帧 confidence，因此适配器不虚构 `confidence.npy`。需要先检查 CAD 投影和轨迹，再决定是否从新的可靠 mask 重启一段序列。完整磁盘合同见 [`perception_io.md`](perception_io.md)。

CAD 单位通过适配器的 `--mesh-scale` 明确转换到米。`object_points_m.npy` 必须从同一份缩放后 CAD 表面采样，并保留完全相同的物体原点和轴。

本仓库的确定性采样器会保留输入 mesh 坐标：

```bash
cd /home/lzr/code/demo2gripper
python -m pip install -e ".[mesh]"
python scripts/sample_object_points.py \
  --mesh assets/object.obj \
  --mesh-unit-scale 1.0 \
  --output runs/real/object_points_m.npy
```

没有 CAD 时，锁文件提供 [Point2Pose](https://github.com/tzuyuan/point-to-pose) 作为可选 RGB-D 在线重建/跟踪后端；它会改变物体几何和坐标基准，不应在首轮已知 CAD 实验中混用。

## 4. GraspGenX：Panda 候选

锁定仓库：[NVlabs/GraspGenX](https://github.com/NVlabs/GraspGenX)。本机实际环境是 Python `3.10.20`、PyTorch `2.7.1+cu128`，仓库位于 `/home/lzr/code/demo2gripper/third_party/GraspGenX`，解释器位于 `/mnt/sdb/lzr/demo2gripper_envs/graspgenx/bin/python`：

```bash
cd /home/lzr/code/demo2gripper/third_party/GraspGenX
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=/home/lzr/code/demo2gripper/third_party/GraspGenX \
GRASPGENX_GRIPPER_CFG_DIR=/mnt/sdb/lzr/demo2gripper_models/gripper_descriptions \
GRASPGENX_CHECKPOINT_DIR=/mnt/sdb/lzr/demo2gripper_models/graspgenx_checkpoints \
  /mnt/sdb/lzr/demo2gripper_envs/graspgenx/bin/python scripts/demo_object_mesh.py \
  --mesh_file /absolute/path/to/object_mesh_m.obj \
  --mesh_scale 1.0 \
  --checkpoints /mnt/sdb/lzr/demo2gripper_models/graspgenx_checkpoints/release \
  --gripper_name franka_panda \
  --grasp_threshold -1.0 \
  --return_topk \
  --topk_num_grasps 100 \
  --no-visualization \
  --output_file /absolute/path/to/runs/real/grasps.yml
```

若 mesh 不是米制，`--mesh_scale` 必须改成实际到米的比例。上游 mesh demo 会先中心化点云，推理后再把位姿变回输入 mesh 坐标。GraspGenX 约定 `+Z` 为接近轴、`+X` 为闭合方向，和本项目合同一致。

官方 `graspgenx.dataset.eval_utils.load_from_isaac_grasp_format` 可从 YAML 读取：

```python
poses, scores = load_from_isaac_grasp_format("runs/real/grasps.yml")
```

该 YAML 的 position/quaternion 是输入 mesh 坐标中的 **gripper-base pose**，并非本项目比较的 finger-center TCP；四元数格式为显式 `w` 加 `xyz`。它也**不含逐候选 jaw width**。使用仓库转换器：

```bash
cd /home/lzr/code/demo2gripper
CUDA_VISIBLE_DEVICES=7 python scripts/convert_graspgenx.py \
  --input runs/real/grasps.yml \
  --output runs/real/candidates.npz \
  --pregrasp-width-m 0.08 \
  --tcp-offset-m 0.1034
```

转换器沿每个候选的局部 `+Z` 平移 `--tcp-offset-m`，得到 `T_object_gripper_tcp`；`widths_m` 记录 `--pregrasp-width-m` 指定的预抓取总开口。0.1034 m 与 0.08 m 是当前 Panda 资产的显式配置，不从 YAML 猜测。若更换 gripper/TCP，必须从对应资产重新测量并同时修改转换器和 Isaac 参数。

## 5. 核心融合

四个时序数组通过：

```bash
cd /home/lzr/code/demo2gripper
python scripts/build_observations.py \
  --t-camera-object runs/real/wapr/T_camera_object.npy \
  --hand-joints-camera-m runs/real/hand_pose/hand_joints_camera_m.npy \
  --confidence runs/real/hand_pose/hand_joint_confidence.npy \
  --timestamps-s runs/real/hand_pose/timestamps_s.npy \
  --output runs/real/observations.npz
```

随后运行：

```bash
cd /home/lzr/code/demo2gripper
demo2gripper learn-intent \
  --observations runs/real/observations.npz \
  --object-points runs/real/object_points_m.npy \
  --output runs/real/grasp_intent.json

demo2gripper select-grasp \
  --intent runs/real/grasp_intent.json \
  --candidates runs/real/candidates.npz \
  --output runs/real/selected_grasp.json
```

这些命令是 CPU 几何计算，不需要 GPU 前缀。
圆柱瓶盖若绕 CAD `+Z` 连续对称，在 `learn-intent` 命令追加
`--continuous-symmetry-axis z`；候选排序会使用相同的 yaw 等价类。

## 6. Isaac Lab

仿真固定使用已安装的 RoboDojo 环境：Isaac Sim 5.1.0.0、Isaac Lab 源码提交
`afca7b09d60d8beb9c1cb28b43066499940b969b`（安装包版本 `0.54.3`）和 Python 3.11.16。
源码位于 `/home/lzr/code/RoboDojo/third_party/IsaacLab`，解释器位于
`/mnt/sdb/lzr/univtac/conda_envs/RoboDojo/bin/python`。不要复用感知或 GraspGenX 环境。

本仓库仿真脚本、资产要求、控制阶段和判定指标见 [`isaaclab.md`](isaaclab.md)。`--object-pose`
只设置仿真物体的初始真值；控制目标使用 `--estimated-object-pose`、
`--estimated-object-pose-json` 或仅调试用的 `--use-oracle-object-pose`，三者必须选一个。
控制专用运行可用 `isaaclab.python.headless.kit`；只要指定 `--rgbd-output-dir` 或视频输出，就必须改用 `isaaclab.python.headless.rendering.kit`，否则相机扩展不会加载。

先用内置动态方块验证完整控制链路：

```bash
OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=7 \
  /mnt/sdb/lzr/univtac/conda_envs/RoboDojo/bin/python \
  /home/lzr/code/demo2gripper/scripts/isaaclab_validate_grasp.py \
  --grasp-json /home/lzr/code/demo2gripper/examples/smoke_selected_grasp.json \
  --robot-usd /home/lzr/code/RoboDojo/Assets/Robots/franka/Franka.usd \
  --smoke-cuboid \
  --object-pose 0.55 0 0.04 0 0 0 1 \
  --use-oracle-object-pose \
  --experience /home/lzr/code/RoboDojo/third_party/IsaacLab/apps/isaaclab.python.headless.kit \
  --device cuda:0 \
  --headless \
  --result-json /home/lzr/code/demo2gripper/runs/smoke/isaaclab_grasp_result.json
```

真实动态刚体 USD 使用外部感知位姿：

```bash
OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=7 \
  /mnt/sdb/lzr/univtac/conda_envs/RoboDojo/bin/python \
  /home/lzr/code/demo2gripper/scripts/isaaclab_validate_grasp.py \
  --grasp-json /home/lzr/code/demo2gripper/runs/real/selected_grasp.json \
  --robot-usd /home/lzr/code/RoboDojo/Assets/Robots/franka/Franka.usd \
  --object-usd /absolute/path/to/object.usd \
  --object-pose 0.55 0 0.05 0 0 0 1 \
  --estimated-object-pose-json /home/lzr/code/demo2gripper/runs/real/estimated_object_pose.json \
  --tcp-offset-m 0.1034 \
  --experience /home/lzr/code/RoboDojo/third_party/IsaacLab/apps/isaaclab.python.headless.kit \
  --device cuda:0 \
  --headless \
  --result-json /home/lzr/code/demo2gripper/runs/real/isaaclab_grasp_result.json
```

`--smoke-cuboid` 与 `--object-usd` 二选一。命令行和位姿 JSON 均采用世界坐标中的
`x y z qx qy qz qw`（`XYZW`）；脚本只在 Isaac 5.1 接口边界转换为内部 `WXYZ`。主流程应通过
标定外参计算 `T_world_object = T_world_camera @ T_camera_object`，再传给
`--estimated-object-pose` 或 `--estimated-object-pose-json`；不能把相机系位姿直接作为世界系输入。
仿真 GT `T_world_object` 只用于 `evaluate_object_pose.py` 评分，不进入 WAPR、坐标合成或外部估计控制。
Isaac 使用相同的 `--tcp-offset-m` 从 canonical TCP 反变换到 hand link；它必须与 YAML 转换值
一致。夹爪从 `pregrasp_width_m` 向 `close_target_width_m` 闭合，双侧手指接触后才进入 LIFT；
到达闭合目标或超时仍无双侧接触则失败。物理 GPU 7 在进程内是 `cuda:0`。
