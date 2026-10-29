# 感知适配器与数据合同

适配器把 SAM 3/3.1、WAPR、SAM 2.1、RTMPose-m Hand5 和 FoundationPose fallback 接到统一的磁盘边界。当前已知 CAD 的对象感知主线是 `Isaac RGB-D -> SAM 3/3.1 文本 mask -> WAPR -> T_camera_object -> T_world_object`。GPU 推理固定使用物理 **7 号卡**；设置 `CUDA_VISIBLE_DEVICES=7` 后，进程内可见设备编号是 `cuda:0`。

## Isaac 单帧对象位姿主线

本次 `008_pudding_box` 的 RGB-D 输入已经由 Isaac Sim 5.1 实际导出到
`/mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/input`。带相机采集必须使用
`isaaclab.python.headless.rendering.kit`：

```bash
OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=7 \
  /mnt/sdb/lzr/univtac/conda_envs/RoboDojo/bin/python \
  /home/lzr/code/demo2gripper/scripts/isaaclab_validate_grasp.py \
  --grasp-json /mnt/sdb/lzr/demo2gripper_runs/dexycb_pudding_view06/isaac_stage0/candidate_7_selected_grasp.json \
  --robot-usd /home/lzr/code/RoboDojo/Assets/Robots/franka/Franka.usd \
  --object-usd /mnt/sdb/lzr/demo2gripper_runs/dexycb_pudding_view06/isaac/008_pudding_box.usd \
  --object-pose 0.55 0.0 0.084 0.0 -0.7071067811865476 0.7071067811865476 0.0 \
  --use-oracle-object-pose \
  --contact-threshold-n 10.0 \
  --rgbd-output-dir /mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/input \
  --video-output /mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/capture_debug.mp4 \
  --experience /home/lzr/code/RoboDojo/third_party/IsaacLab/apps/isaaclab.python.headless.rendering.kit \
  --device cuda:0 --headless \
  --result-json /mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/capture_debug_result.json
```

`--use-oracle-object-pose` 只驱动这次采集命令附带的调试抓取；后续感知不读取该控制目标。感知输入只有
`rgb/000000.png`、米制 `depth/000000.npy`、`K.npy` 和
`calibration/T_world_camera.npy`。仿真真值单独位于
`evaluation/T_world_object.npy`，只传给 `evaluate_object_pose.py` 计算误差，不传给 SAM、WAPR、位姿合成或抓取控制。

本机已经用已有官方 SAM 3 checkpoint 和文本 `pudding box` 运行：

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

实际输出为 `sam3/masks/1/000000.png`、`sam3/manifest.json` 和
`sam3/first_frame_overlay.png`。该命令只有文本提示入口，没有 GT box/mask 入口。当前证据使用 SAM 3；最新的 SAM 3.1 checkpoint 在 Hugging Face 上 gated，接受许可并登录后可沿用相同适配器替换 checkpoint。

## 输入序列

推荐目录如下：

```text
data/demo/
├── rgb/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── depth/
│   ├── 000000.npy
│   ├── 000001.npy
│   └── ...
└── K.npy
```

- RGB 是同分辨率、数字文件名、连续编号的 RGB PNG。SAM 2/3 适配器也接受数字文件名的 RGB JPEG。
- depth 与 RGB 文件 stem 一一对应，形状为 `(H,W)`、浮点数、单位米。`0` 可表示无效深度。
- `K.npy` 是 OpenCV 针孔模型的 `3×3` 内参；深度必须已经对齐到 RGB。
- 同一刚体的 CAD、WAPR 输出、CAD 表面点和 GraspGenX 输入必须共享原点、轴和尺度。

### RealSense 在线采集

使用固定相机的序列号选择设备，避免多台同型号相机在重启或重新插拔后选错：

```bash
/home/vitai/miniconda3/bin/python scripts/capture_realsense_rgbd.py \
  --serial 254622073804 \
  --output-dir runs/real/input \
  --width 640 --height 480 --fps 30 \
  --warmup-frames 30 --frames 300
```

脚本用 RealSense SDK 把 Z16 深度重投影到彩色光学坐标，按设备的深度比例转换为
`float32` 米制 `.npy`，并从实际彩色流导出 `K.npy`。`timestamps_s.npy` 来自彩色帧的
原始设备硬件时钟且严格递增；每帧的彩色/深度时间戳、帧号、有效深度比例和相机畸变
参数另存于 `input_manifest.json`。输出目录必须为空。脚本依赖 `pyrealsense2` 和 OpenCV；
本仓库核心环境不强制安装相机 SDK，以保持感知和机器人环境隔离。采集期间会强制检查彩色和
深度的帧号、设备时间戳都持续递增，并限制两者的硬件时间差；重复或失同步的深度帧不会写入。
脚本先写同目录下的临时目录，全部帧和 manifest 完成后才原子发布输出目录，因此
`input_manifest.json` 的存在是一次完整采集的完成标记。

### 固定外部相机到 FR3 基座的标定

本机使用固定 D435（序列号 `254622073804`）时，标定目标是
`calibration/T_base_camera.npy`：它把彩色光学相机坐标中的点变换到 FR3 基座坐标。
本工作目录已经生成下列标定板；直接打印 PDF，不要对同一个非空输出目录重复运行生成器。
在新的工作目录中若文件不存在，再运行：

```bash
/home/vitai/miniconda3/bin/python scripts/generate_charuco_board.py \
  --output-dir outputs/charuco_7x5_30mm
```

打印 `outputs/charuco_7x5_30mm/charuco_board.pdf` 时选择“实际大小”或 `100%`，关闭
“适合页面”和自动缩放。图案外框应为 `210 mm × 150 mm`，单格边长应为 `30 mm`，
ArUco 标记边长应为 `22 mm`；打印后必须实测。把纸张完整贴在平整、刚性的背板上，
再将背板刚性固定到末端法兰或 Hand-E。相机在全部样本采集和后续使用期间都不能相对
FR3 基座移动，标定板在全部样本之间也不能相对法兰滑动，Hand-E 开口不能改变。

也可以使用刚性普通棋盘格。先沿两个方向分别数**完整方格数**（不是内角点数），并实测
单格边长；代码不会从照片猜物理尺寸。两个方格数必须一奇一偶，否则无标记棋盘格旋转
`180°` 后无法唯一确定同一个物理角点顺序，不能用于本手眼标定。用实测值生成严格的
版本化 metadata，例如：

```bash
SQUARES_X=填写横向完整方格数
SQUARES_Y=填写纵向完整方格数
SQUARE_LENGTH_MM=填写实测单格毫米数
/home/vitai/miniconda3/bin/python scripts/create_checkerboard_metadata.py \
  --squares-x "$SQUARES_X" \
  --squares-y "$SQUARES_Y" \
  --square-length-mm "$SQUARE_LENGTH_MM" \
  --output runs/handeye/checkerboard_metadata.json
```

随后把采集命令的 `--board-metadata` 换成该文件。采集器使用
`findChessboardCornersSBWithMeta` 的原生亚像素角点及黑白 meta 签名，要求全部内角点可见，
并用 IPPE 的两个平面位姿候选检查歧义。板面不要接近平行于成像平面；若提示 planar pose
ambiguous，增加倾斜角度后再采。

操作者每次将机械臂置于一个静止姿态后，采集一个独立目录。脚本只读取 libfranka 状态，
不会解锁、启动控制或发送运动命令：

```bash
/home/vitai/miniconda3/bin/python scripts/capture_handeye_sample.py \
  --serial 254622073804 \
  --board-metadata outputs/charuco_7x5_30mm/charuco_board.json \
  --output-dir runs/handeye/dry_run \
  --dry-run-camera-only

/home/vitai/miniconda3/bin/python scripts/capture_handeye_sample.py \
  --serial 254622073804 \
  --board-metadata outputs/charuco_7x5_30mm/charuco_board.json \
  --output-dir runs/handeye/samples/000 \
  --session-id fr3-d435-fixed-01 \
  --robot-hostname 172.16.0.2
```

当 MoveIt/`ros2_control` 已经连接 FR3 时，不要再启动第二个 libfranka reader。先加载
与正在运行的驱动相同的 ROS overlay，再改用只读 state broadcaster 订阅：

```bash
source /opt/ros/jazzy/setup.bash
source third_party/franka_ros2_ws/install_ros/setup.bash
/home/vitai/miniconda3/bin/python scripts/capture_handeye_sample.py \
  --serial 254622073804 \
  --board-metadata outputs/charuco_7x5_30mm/charuco_board.json \
  --output-dir runs/handeye/samples/000 \
  --session-id fr3-d435-fixed-01 \
  --ros-robot-state-topic
```

该模式只订阅 `/franka_robot_state_broadcaster/robot_state`，并按
`T_base_flange = O_T_EE @ inverse(F_T_EE)` 生成与直接 libfranka 模式相同语义的
`T_base_flange.npy`。broadcaster 必须处于 `active` 并持续发布新状态；仅有一条
transient-local 缓存消息时采集会超时并拒绝落盘。

依次把输出编号改为 `001`、`002`……，采集 `25` 个样本。让标定板覆盖画面中心、四角、
近处和远处，并分别绕至少两个不平行轴改变约 `20–40°`；每次采集前完全停稳，同时确保
至少 `12` 个 ChArUco 角点清晰可见；普通棋盘格则必须全部内角点清晰可见。脚本会拒绝运动中的机械臂、活动错误、失同步 RGB-D、
重复帧、较差重投影和不完整输出。每个样本的 `overlay.png` 必须人工检查：角点和红绿蓝
坐标轴应贴在实体板上。一次数据集的所有命令必须使用同一个 `--session-id`；它声明相机底座
和板到法兰的安装关系没有改变。任一安装被触碰后应停止当前数据集，以新 session 和新目录重采。

保留最后 `5` 个姿态作为独立验证集，再求解：

```bash
/home/vitai/miniconda3/bin/python - <<'PY'
import cv2
assert hasattr(cv2, "calibrateRobotWorldHandEye")
assert hasattr(cv2, "calibrateHandEye")
print(cv2.__version__)
PY

/home/vitai/miniconda3/bin/python scripts/solve_eye_to_hand.py runs/handeye/samples \
  --validation-indices 20 21 22 23 24 \
  --output-dir calibration
```

本机 Conda base 的 Python 同时提供 RealSense SDK、Pillow 和 OpenCV `4.13.0`，后者具备
这两个求解接口；当前项目 `.venv` 中的 OpenCV `5.0.0` 不提供，不能用它运行求解命令。
求解器会先拒绝混用不同 session、相机序列号、内参、畸变或标定板的样本，再同时比较
SHAH、LI 和 PARK，检查运动多样性、
训练闭合误差、保留样本误差与方法间一致性；任一默认阈值超限时不会发布结果。接受后输出
`T_base_camera.npy`、标定板相对法兰的 `T_flange_board.npy` 和完整 `report.json`。只有
`report.json` 中 `acceptance.accepted` 为 `true` 时才能使用该外参。

## SAM 3 / 3.1 文本目标分割

`scripts/segment_sam3.py` 调官方 `build_sam3_image_model` 和 `Sam3Processor`，对每帧使用同一类别文本并保存最高分实例。输出布局为：

```text
runs/real/sam3/
├── masks/1/000000.png ...
├── first_frame_overlay.png
└── manifest.json
```

mask 与原图同分辨率，类型为 `uint8`，背景 `0`、前景 `255`。manifest 保存文本、checkpoint、设备、逐帧 box、score、耗时和相对 mask 路径。它不读取深度或仿真真值；深度在下一步 WAPR 中与 mask 对齐使用。

SAM 3.1 是锁文件中的最新可选 profile，其 Hugging Face checkpoint 需要 gated access。本机现有可复现证据使用 SAM 3 commit `6fe87d64` 和 `sam3.pt`，不要把这次输出标成 SAM 3.1。

## SAM 2.1 手/部件多目标视频分割

安装锁定版本及 checkpoint：

```bash
SAM2_BUILD_CUDA=0 /mnt/sdb/lzr/demo2gripper/envs/perception/bin/python \
  -m pip install --no-deps -e /mnt/sdb/lzr/demo2gripper/backends/sam2
curl -fL \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
  -o /mnt/sdb/lzr/demo2gripper/models/sam2.1_hiera_small.pt
```

首帧提示文件采用一个明确格式。`labels` 中 `1` 是前景点、`0` 是背景点；每个对象使用 points/labels 或 XYXY box 二选一。手、目标主体和会独立运动的部件应使用不同 `obj_id`。

```json
{
  "frame_idx": 0,
  "objects": [
    {
      "obj_id": 1,
      "points": [[420, 265], [381, 244]],
      "labels": [1, 0]
    },
    {
      "obj_id": 2,
      "box": [510, 190, 690, 430]
    },
    {
      "obj_id": 3,
      "box": [545, 170, 650, 225]
    }
  ]
}
```

本次真实序列使用 SAM 2.1 Hiera-S；输出目录必须不存在或为空：

```bash
cd /home/lzr/code/demo2gripper
CUDA_VISIBLE_DEVICES=7 \
  /mnt/sdb/lzr/demo2gripper/envs/perception/bin/python scripts/segment_sam2.py \
  --rgb-dir /mnt/sdb/lzr/demo2gripper_runs/dexycb_pudding_view06/input/rgb \
  --checkpoint /mnt/sdb/lzr/demo2gripper/models/sam2.1_hiera_small.pt \
  --model-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --prompts /mnt/sdb/lzr/demo2gripper_runs/dexycb_pudding_view06/sam2/prompts.json \
  --output-dir /mnt/sdb/lzr/demo2gripper_runs/dexycb_pudding_view06/segmentation \
  --device cuda
```

适配器调用 `build_sam2_video_predictor`、`init_state`、`add_new_points_or_box` 和 `propagate_in_video`。输入会无损暂存为 SAM 2 所需的数字 JPEG 路径，不会重新压缩。输出为：

```text
runs/real/segmentation/
├── masks/
│   ├── 1/000000.png ...
│   ├── 2/000000.png ...
│   └── 3/000000.png ...
└── manifest.json
```

mask 与原图同分辨率，类型为 `uint8`，背景 `0`、前景 `255`。`manifest.json` 保存帧名、对象 ID 和相对路径。默认把视频帧留在 CPU、状态留在 GPU；显存不足时可加 `--offload-state-to-cpu`，速度会下降。

## WAPR 已知 CAD 位姿

主后端是 [WangYuLin-SEU/WAPR](https://github.com/WangYuLin-SEU/WAPR)，固定提交
`f165cec8ebdd48ae6f46d06a46c40d796ab80b71`。输入是单帧 RGB、米制 depth、OpenCV `K`、目标 mask 和米制 CAD；输出是把 CAD 物体坐标变换到 OpenCV 光学相机坐标的 `T_camera_object`。源码许可为 LGPL-2.1-only；`wapr_w_mask.pth`、`wapr_wo_mask.pth`、`sapr.pth`、`wbps.pth` 四份一方权重为 CC BY-ND 4.0。

本机 WAPR 环境位于 `/mnt/sdb/lzr/demo2gripper_envs/wapr`，缓存位于
`/mnt/sdb/lzr/demo2gripper_cache/wapr`。`WAPR_BACKEND=torch` 运行同一组模型而不安装 TensorRT；代价是推理慢于官方 TensorRT FP16 路径。仅获取四份位姿权重：

```bash
WAPR_CACHE_DIR=/mnt/sdb/lzr/demo2gripper_cache/wapr \
  /mnt/sdb/lzr/demo2gripper_envs/wapr/bin/python -c \
  "from wapr.download_assets import check_and_fetch_pack; check_and_fetch_pack('wapr_sapr_wbps')"
```

Isaac pudding-box 输入的主后端命令是：

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

输出合同是：

- `T_camera_object.npy`：`(1,4,4) float32`，米制 object-to-camera 变换；
- `wapr_manifest.json`：输入、坐标约定、模型分数和各阶段耗时；
- `pose_overlay.png`：CAD 轴与包围框投影，供目视检查坐标系。

`score_6d` 是候选排序信号，不是已校准的成功概率。下游必须直接使用返回位姿，不得依据仿真真值选择、反转或修正它。

本次同一 Isaac 帧的实际结果为：`score_6d=1.0035`，PyTorch pose inference `1.037 s`，合成世界位姿后相对隔离 GT 的误差为 `1.255 mm / 0.498 deg`。该位姿随后直接驱动 candidate 7；结果 JSON 明确记录 `target_pose_source=external_estimate`、`pin_object_until_contact=false`，IK、双指接触、抬升和保持均通过，最大抬升 `74.31 mm`。产物位于：

- `/mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/wapr/pose_overlay.png`
- `/mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/wapr/pose_evaluation.json`
- `/mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/wapr/external_pose_grasp.mp4`
- `/mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/wapr/external_pose_grasp_result.json`
- `/mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/sota_pose_pipeline_proof.mp4`：SAM mask、WAPR 投影和外部位姿抓取的同链路三栏视频。

GT 只由 `evaluate_object_pose.py` 在 WAPR 推理完成后读取；上述外部位姿控制运行只读 `estimated_object_pose.json`。

## FoundationPose fallback

原版 [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) 是本机已经实跑的 fallback，锁定提交见 `backends.lock.json`。本机环境为 Python 3.11.16、torch `2.7.0+cu128`、PyTorch3D `0.7.9`、nvdiffrast `0.4.0`；PyTorch3D、nvdiffrast 和上游 `mycpp` 扩展均为 RTX 5090 的 `sm_120` 编译。上游权重按其目录约定放在 `FoundationPose/weights/`。

下面是本次 `view_06`、`008_pudding_box` 的实际命令。必须从 FoundationPose 根目录启动，因为上游按相对路径加载权重；`obj_id=2` 与 SAM 提示文件一致：

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

`--mesh-scale` 是 CAD 单位到米的比例：米制 mesh 用 `1.0`，毫米 mesh 用 `0.001`。适配器用首帧 RGB-D-mask 调 `FoundationPose.register`，后续 RGB-D 帧调 `track_one`，产生：

- `T_camera_object.npy`：`(F,4,4) float32`，把米制 CAD 坐标点变换到 RGB 相机坐标；
- `foundationpose_manifest.json`：首帧原始 registration score、输入路径、位姿约定、每帧 register/track 阶段与耗时。

适配器只在首帧注册，后续逐帧跟踪；它保存所有有限 pose，不按阈值自动丢帧或重新注册。先检查 CAD 投影和轨迹，漂移时从新的可靠 mask 启动另一段序列。对于绕 CAD `+X/+Y/+Z` 连续旋转对称的物体，FoundationPose 的绕轴角可能跳变；在下游 `learn-intent` 使用对应的 `--continuous-symmetry-axis` 做等价类规范化。该轴必须穿过 CAD 原点。

若瓶盖等部件相对主体独立运动，应给它独立 mask 和独立 CAD，再单独运行一次 FoundationPose。若部件始终属于同一刚体，只跟踪完整物体，避免把多个 tracker 的噪声当成关节运动。

## RTMPose-m Hand5 + RGB-D 手位姿

默认手后端是 OpenMMLab 官方 RTMPose-m Hand5 ONNX，由 `rtmlib==0.0.16` 和 `onnxruntime-gpu==1.23.2` 执行。它不需要额外手 detector：每帧直接取 SAM 手 mask 的非零范围作为 top-down bbox。ONNX Runtime 必须能找到 PyTorch 环境中的 CUDA 12、cuBLAS 和 cuDNN 9 动态库；只看到 `CUDAExecutionProvider` 名字并不足以证明实际加载成功。

本次 18 帧、10 FPS 序列的实际命令如下。`CUDA_VISIBLE_DEVICES=7` 后，ORT 的 `--device cuda` 使用进程内 `cuda:0`，即物理 7 号卡：

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
  --backend onnxruntime \
  --device cuda \
  --fps 10
```

关键点顺序是 `wrist, thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4`。脚本在每个 2D 关节周围的手 mask 内采样正深度，以前景四分位值抑制轮廓后的桌面深度，再用 `K.npy` 反投影到相机米制坐标。输出为：

- `hand_keypoints_2d.npy`：`(F,21,2) float32`，RTMPose 像素坐标；
- `hand_keypoint_scores_2d.npy`：`(F,21) float32`，原始 SimCC 峰值，未归一化且可能大于 `1`；
- `hand_joints_camera_m.npy`：`(F,21,3) float32`，逐关节 RGB-D 抬升结果；
- `hand_joint_confidence.npy`：`(F,21) float32`，raw score 裁剪到 `[0,1]`，无有效深度时为 `0`；
- `timestamps_s.npy`：`(F,) float64`，由 `--fps` 生成的后端 fallback；真机融合应使用
  RealSense 输入目录中的设备硬件时间戳；
- `rtmpose_manifest.json`：模型、关键点顺序、score 合同、逐帧 bbox 和有效深度关节数。

本次真实输出的逐帧叠加视频位于 `/mnt/sdb/lzr/demo2gripper_runs/dexycb_pudding_view06/hand_pose/sam2_rtmpose_rgbd.mp4`，同时显示 SAM 手/物体 mask、21 点骨架和有效 RGB-D 关节数。

RGB、depth 和 mask 必须使用相同的数字 stem。若采集时间不等间隔，用设备时间戳替换按 `--fps` 生成的数组。AnyHand-WiLoR 及 `track_wilor_rgbd.py` 只保留为需要 MANO mesh/shape 时的可选路径；默认 21 关键点链路不依赖 MANO。

## 接到核心融合

把物体位姿、手关节与时间戳打包。注意这里的 `--confidence` 必须是 RTMPose 产生的 `(F,21)` 手关节 confidence；WAPR 的 `score_6d` 和 FoundationPose fallback 的 `registration_score_raw` 都不能代替它：

```bash
python scripts/build_observations.py \
  --t-camera-object runs/real/wapr/T_camera_object.npy \
  --hand-joints-camera-m runs/real/hand_pose/hand_joints_camera_m.npy \
  --confidence runs/real/hand_pose/hand_joint_confidence.npy \
  --timestamps-s runs/real/input/timestamps_s.npy \
  --output runs/real/observations.npz
```

WAPR 的结果是相机系 `T_camera_object`，不是 Isaac 使用的世界系目标。用已标定外参直接生成 Isaac 输入文件：

```bash
python scripts/compose_world_object_pose.py \
  --t-camera-object runs/real/wapr/T_camera_object.npy \
  --t-world-camera calibration/T_world_camera.npy \
  --frame-index -1 \
  --output runs/real/estimated_object_pose.json
```

该文件同时保存
`T_world_object = T_world_camera @ T_camera_object` 和外部 `XYZW` 位姿，可直接传给仿真脚本的 `--estimated-object-pose-json`；这里不需要对 `T_camera_object` 求逆。脚本在 Isaac 5.1 接口边界转换为内部 `WXYZ`。`--object-pose` 仅设置仿真中的初始真值。

仿真内有真值时，评分单独运行：

```bash
python scripts/evaluate_object_pose.py \
  --t-camera-object runs/real/wapr/T_camera_object.npy \
  --t-world-camera calibration/T_world_camera.npy \
  --t-world-object-gt evaluation/T_world_object.npy \
  --output runs/real/wapr/pose_metrics.json
```

`evaluation/T_world_object.npy` 只出现在这条评估命令中。估计、位姿合成和外部位姿控制均不读取它。

分割 mask 应抽查首帧、遮挡最重帧和最终接触帧；位姿轨迹应先检查物体坐标轴与 CAD 可视化是否一致。两者成立后再学习物体坐标系内的手—物接触关系。
