# demo2gripper

把一段标定后的 RGB-D 人手示范压缩成**物体坐标系中的平行夹爪抓取意图**，再从 GraspGenX 候选中选出实际 Franka Panda 能执行的 6-DoF 抓取位姿，并交给 Isaac Lab 做接近、闭合、抬升和保持验证。

在另一台 Ubuntu/Franka 主机上部署时，先阅读 [`AGENTS.md`](AGENTS.md)。其中区分了可直接安装的核心环境、需要独立安装的 GPU 后端和当前尚未实现的真机执行器；本文中的 `/home/lzr`、`/mnt/sdb` 与物理 GPU 7 是原机器的复现实录，不是新机器必须沿用的配置。

```text
RGB-D + K + CAD
  ├─ SAM 3 / 3.1 文本提示：目标物体 mask
  │       └─ WAPR：T_camera_object
  └─ SAM 2.1 手 mask → RTMPose-m Hand5：21 个 2D 关节 → RGB-D 逐关节抬升
          ↓ 同一相机坐标系、米制
  捏取：稳定拇指—食指接触段
  整手握持：2D 手势与 CAD 重投影共同运动区域
          ↓ 物体坐标系中的接触区域与闭合轴
  GraspGenX(Panda) 候选 ── 几何硬过滤 + 示范相似度排序
          ↓ T_object_gripper_tcp
  Isaac Lab：PREGRASP → APPROACH → CLOSE → LIFT → HOLD
```

当前实现锁定为：单个主动手、单个刚性目标、已知 CAD、标定且同步的 RGB-D、Franka Panda 平行夹爪。核心 `learn-intent` 处理拇指—食指捏取；`infer_power_grasp_intent.py` 用于电钻这类整手握持。瓶身与瓶盖等相对运动的部件必须分别分割、分别建模和跟踪，不能共用一个刚体位姿。

## 方法选择

| 环节 | 锁定后端 | 在本链路中的职责 |
|---|---|---|
| 目标物体分割 | [SAM 3](https://github.com/facebookresearch/sam3)；最新可选 SAM 3.1 | 用类别文本直接产生目标 mask，不读取真值框或真值 mask |
| 手/部件视频分割 | [SAM 2.1](https://github.com/facebookresearch/sam2) | 首帧点/框提示，随后传播多个稳定 object ID |
| 手 2D/3D 关节 | [RTMPose-m Hand5](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) + [rtmlib](https://github.com/Tau-J/rtmlib) | 用 SAM 手 mask 的 bbox 运行 21 点模型，再从对齐深度逐关节反投影到相机米制坐标 |
| 已知 CAD 物体位姿 | [WAPR](https://github.com/WangYuLin-SEU/WAPR) | 从 RGB-D、目标 mask、内参和米制 CAD 输出 object-to-camera 4×4 位姿；FoundationPose 是已跑通 fallback |
| 夹爪候选 | [GraspGenX](https://github.com/NVlabs/GraspGenX) | 对目标 mesh/点云生成 Panda 6-DoF 候选及模型置信度 |
| 仿真 | RoboDojo Isaac Lab commit `afca7b09` + Isaac Sim 5.1.0.0 | 差分 IK、接触检测、抬升/保持判定 |

版本与提交号固定在 [`configs/backends.lock.json`](configs/backends.lock.json)。WAPR 主后端固定到 `f165cec8`：源码为 LGPL-2.1-only，四份一方权重为 CC BY-ND 4.0；`WAPR_BACKEND=torch` 使用相同模型而不要求 TensorRT。默认手链路的 rtmlib/MMPose 源码使用 Apache-2.0，不依赖 MANO；AnyHand-WiLoR 只作为可选的 MANO 重建方案。原版 FoundationPose 使用限非商业研究/评估的 NVIDIA Source Code License，GraspGenX 权重使用 NVIDIA Open Model License；Isaac Lab 本身开放源码，但完整 PhysX/RTX 流程依赖专有的 Isaac Sim 运行时。

对象分割的本机证据使用已有官方 SAM 3 checkout 和 `sam3.pt`；SAM 3.1 是最新目标，但 Hugging Face checkpoint 需要先接受许可并登录获取。两者都走 `scripts/segment_sam3.py` 的官方图像 API。SAM 2.1 继续服务于人手示范中的交互式多目标视频分割，不是当前 Isaac 目标物体 mask 的来源。

`008_pudding_box` 的单帧 Isaac 闭环已经在物理 GPU 7 上实跑：SAM 3 文本 mask 送入 WAPR 后，世界系物体位姿相对隔离的仿真真值为 `1.255 mm / 0.498 deg`；该外部估计驱动 candidate 7，在不 pin 物体且不使用 oracle 控制目标的条件下通过 IK、双指接触、抬升与保持，最大抬升 `74.31 mm`。三模块并排证据视频为 `/mnt/sdb/lzr/demo2gripper_runs/isaac_pudding_perception/sota_pose_pipeline_proof.mp4`，逐项结果位于同目录的 `sam3/` 与 `wapr/`。

## 先跑通核心链路

核心运行环境依赖 NumPy、SciPy 和 PyYAML，不需要 GPU。以下合成样例会真正经过接触段检测、物体坐标变换、候选硬过滤和排序：

```bash
cd /home/lzr/code/demo2gripper
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

demo2gripper make-synthetic --output-dir runs/demo
demo2gripper run --config configs/pipeline.example.json
pytest -q
```

结果为 `runs/demo/grasp_intent.json` 与 `runs/demo/selected_grasp.json`。后者是仿真入口，包含 `T_object_gripper_tcp`、`pregrasp_width_m`、`close_target_width_m` 和 `approach_distance_m`。

## 逐模块视频检查

统一渲染器把同一帧组织为四宫格：输入 RGB、手/物体/部件 mask、21 点手骨架与物体 CAD/6D 位姿、稳定接触段与最终夹爪位姿。先用几何一致的合成数据检查接口和投影；成片底部会明确写出它不是模型输出：

```bash
python -m pip install -e ".[visualization]"
python scripts/make_visual_example.py --output-dir runs/visual_example

demo2gripper learn-intent \
  --observations runs/visual_example/observations.npz \
  --object-points runs/visual_example/object_points_m.npy \
  --output runs/visual_example/grasp_intent.json

demo2gripper select-grasp \
  --intent runs/visual_example/grasp_intent.json \
  --candidates runs/visual_example/candidates.npz \
  --output runs/visual_example/selected_grasp.json

python scripts/render_pipeline_video.py \
  --rgb-dir runs/visual_example/rgb \
  --intrinsics runs/visual_example/K.npy \
  --segmentation-dir runs/visual_example/segmentation \
  --mask-label 1:hand --mask-label 2:object --mask-label 3:part \
  --observations runs/visual_example/observations.npz \
  --object-points runs/visual_example/object_points_m.npy \
  --object-confidence runs/visual_example/object_pose_confidence.npy \
  --intent runs/visual_example/grasp_intent.json \
  --selected-grasp runs/visual_example/selected_grasp.json \
  --output runs/visual_example/pipeline_annotated.mp4 \
  --preview runs/visual_example/pipeline_preview.jpg
```

该样例的实际核心结果是稳定接触段 `[30,100)`、示范接触宽度 `0.0800 m`，并选中候选 `0`。真实演示沿用同一渲染命令，只需把路径换成 SAM、RTMPose RGB-D 和 WAPR 产生的同一序列结果。普通 RGB 视频只能检查分割和 2D 手关节；米制手—物关系、物体 6D 位姿和夹爪位姿还需要对齐深度、`K.npy` 与目标 CAD。

## 后端检出与环境隔离

当前项目固定只使用 **GPU 7**。所有 CUDA 推理和仿真命令都必须显式带 `CUDA_VISIBLE_DEVICES=7`；不要把该前缀删掉。

```bash
python scripts/clone_backends.py --list
python scripts/clone_backends.py \
  --root third_party/locked \
  --names segmentation hand_video_segmentation hand_pose known_object_pose grasp_candidates simulation
```

接受 gated checkpoint 和自定义许可时，可另行检出 `frontier_segmentation_optional`；它不会替换默认的 `segmentation`。

脚本按锁文件 checkout detached commit；目标非空时直接停止，不覆盖已有目录。权重、MANO 和 Isaac Sim 不会自动下载或接受许可。

当前实跑中，SAM 3、SAM 2.1/RTMPose、WAPR、FoundationPose fallback、GraspGenX 和 Isaac Lab 保持环境隔离，只通过 `.png`、`.npy`、`.npz`、`.json` 交换结果。安装和输出适配见 [`docs/backends.md`](docs/backends.md)，可执行的磁盘接口见 [`docs/perception_io.md`](docs/perception_io.md)，Isaac Lab 运行见 [`docs/isaaclab.md`](docs/isaaclab.md)。

## 真实示范的最短路径

1. 同步保存 RGB、与 RGB 对齐的深度、相机内参 `K` 和秒级时间戳；深度统一为米。另存标定外参 `calibration/T_world_camera.npy`，约定为相机坐标到仿真世界坐标的 `4×4` 变换。
2. 用 `scripts/segment_sam3.py --text-prompt '<目标类别>'` 得到目标物体 mask；该路径不接收真值框或真值 mask。人手和会独立运动的部件仍用 `scripts/segment_sam2.py` 分配独立 ID 并传播全序列。不要把手与物体合成一个 mask；完整命令见 [`docs/perception_io.md`](docs/perception_io.md)。
3. 运行 `scripts/track_rtmpose_rgbd.py`：每帧从 SAM 手 mask 取 bbox，RTMPose-m Hand5 输出 21 个 2D 关节，再用同帧对齐深度和真实 `K` 将每个关节反投影到相机米制坐标。GPU7、ONNX 和 10 FPS 实跑命令见 [`docs/perception_io.md`](docs/perception_io.md)。
4. 用 `scripts/estimate_wapr_pose.py` 将目标 mask、RGB、米制 depth、`K` 和 CAD 接入 WAPR，输出 object-to-camera 的 `T_camera_object.npy`。原版 `scripts/track_foundationpose_original.py` 保留为已实跑 fallback。
5. 从同一 CAD 表面采样米制 `object_points_m.npy`；点必须位于 WAPR 使用的同一物体坐标系：

   ```bash
   python -m pip install -e ".[mesh]"
   python scripts/sample_object_points.py \
     --mesh assets/object.obj \
     --mesh-unit-scale 1.0 \
     --output runs/real/object_points_m.npy
   ```

6. 打包四项时序观测：

   ```bash
   python scripts/build_observations.py \
     --t-camera-object runs/real/wapr/T_camera_object.npy \
     --hand-joints-camera-m runs/real/hand_pose/hand_joints_camera_m.npy \
     --confidence runs/real/hand_pose/hand_joint_confidence.npy \
     --timestamps-s runs/real/hand_pose/timestamps_s.npy \
     --output runs/real/observations.npz
   ```

7. 学习物体坐标系抓取意图：

   ```bash
   demo2gripper learn-intent \
     --observations runs/real/observations.npz \
     --object-points runs/real/object_points_m.npy \
     --output runs/real/grasp_intent.json
   ```

   若目标是绕 CAD `+Z` 轴连续旋转对称的瓶盖，追加
   `--continuous-symmetry-axis z`。核心会在该对称群内规范化逐帧接触几何和候选比较姿态，
   但最终输出仍保留 GraspGenX 候选在原 CAD 坐标系中的可执行姿态。

   电钻等整手握持不应放宽指尖接触阈值。先生成 GraspGenX 候选，再用
   `scripts/infer_power_grasp_intent.py` 将 RTMPose 2D 手势重投影到 CAD，按共同运动的
   抓持区域选择具有 CAD 双侧接触的候选。完整实跑见
   [`docs/imitrob_powerdrill.md`](docs/imitrob_powerdrill.md)。

8. 在 GraspGenX 的独立环境中为 `franka_panda` 生成候选。官方 YAML 给出 gripper-base pose；转换时沿候选局部 `+Z` 加上 Panda 的 0.1034 m base-to-TCP 偏移，得到本项目的 `T_object_gripper_tcp`：

   ```bash
   CUDA_VISIBLE_DEVICES=7 python scripts/convert_graspgenx.py \
     --input runs/real/grasps.yml \
     --output runs/real/candidates.npz \
     --pregrasp-width-m 0.08 \
     --tcp-offset-m 0.1034
   ```

   然后排序：

   ```bash
   demo2gripper select-grasp \
     --intent runs/real/grasp_intent.json \
     --candidates runs/real/candidates.npz \
     --max-jaw-width-m 0.08 \
     --output runs/real/selected_grasp.json
   ```

9. 将 WAPR 的物体位姿从相机系变换到仿真世界系：

   ```bash
   python scripts/compose_world_object_pose.py \
     --t-camera-object runs/real/wapr/T_camera_object.npy \
     --t-world-camera calibration/T_world_camera.npy \
     --frame-index -1 \
     --output runs/real/estimated_object_pose.json
   ```

   先用 `--use-oracle-object-pose` 验证控制与接触；主链路再改用下面的感知位姿文件。

   先在现有 RoboDojo 环境中用内置动态方块贯通控制、接触和抬升：

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

   再换成真实动态刚体 USD 和 WAPR 生成的世界系估计：

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

   `--smoke-cuboid` 与 `--object-usd` 二选一。`--object-pose` 只设置仿真物体的初始真值；
   `estimated_object_pose.json` 保存 `T_world_object` 和外部 `XYZW` 位姿。脚本在 Isaac 5.1 接口
   边界将其转换为内部 `WXYZ`。仅调试控制时，可用 `--use-oracle-object-pose` 替代估计位姿文件。
   仿真使用同一个 TCP 偏移把目标反变换到 Panda hand link；两处数值必须一致。夹爪从
   `pregrasp_width_m` 开始向 `close_target_width_m=0.0` 闭合，只有左右指都检测到接触才进入
   LIFT；闭合到目标或超时仍无双侧接触即失败。物理 GPU 7 在进程内对应 `cuda:0`。

## 数据合同

全项目采用列向量和 `T_parent_child`：

```text
p_parent = T_parent_child @ [p_child.x, p_child.y, p_child.z, 1]
```

所有长度均为米，时间为秒。旋转矩阵右手系。

WAPR 返回 `T_camera_object`，标定文件保存 `T_world_camera`，因此世界系估计直接为
`T_world_object = T_world_camera @ T_camera_object`，不对任一项求逆。仿真导出的 GT
`evaluation/T_world_object.npy` 只用于 `evaluate_object_pose.py` 评分，不进入分割、位姿估计、坐标合成或外部估计控制。

`observations.npz` 只含：

| key | shape | 含义 |
|---|---:|---|
| `T_camera_object` | `(F, 4, 4)` | 物体点变换到 RGB-D 相机坐标 |
| `hand_joints_camera_m` | `(F, 21, 3)` | 同一相机坐标中的 Hand5 顺序 21 关键点；无有效深度的点为零 |
| `hand_joint_confidence` | `(F, 21)` | RTMPose raw SimCC 峰值裁剪到 `[0,1]`；无有效深度的点为 `0` |
| `timestamps_s` | `(F,)` | 严格递增 |

`object_points_m.npy` 为 `(N, 3)`，在物体坐标系中。`candidates.npz` 含 `poses (K,4,4)`、`scores (K,)`、`widths_m (K,)`；`widths_m` 是每个候选的**预抓取总开口**，不是预测接触宽度。其中 pose 均为 `T_object_gripper_tcp`；TCP 的 `+X` 是闭合轴（正负等价），`+Z` 是朝向接触点的接近轴，预抓取点沿 `-Z` 退回。

`selected_grasp.json` 把预抓取开口写为 `pregrasp_width_m`，并把实际闭合目标写为 `close_target_width_m`。默认闭合目标为 `0.0`，接触传感器决定是否提前进入抬升阶段。

捏取算法使用标准 Hand5 顺序的 wrist=0、thumb tip=4、index tip=8。只有两指同时靠近 CAD 表面、关键点置信度与捏距均满足阈值时，帧才进入接触段；取最长连续段的稳健接触中心和轴。整手握持算法不把遮挡下的伪 3D 指尖当接触：它用 RTMPose 2D 关键点在逐帧 CAD 投影上定位共同运动的抓持区域，再要求 GraspGenX 候选的闭合线与 CAD 存在双侧交点。

## 已知限制

- 这不是从视频端到端学习策略；它学习的是可解释的物体相对接触几何，再映射到真实夹爪候选。
- 整手握持当前只学习“抓 CAD 的哪个区域”；单视角遮挡下的背侧人体接触、接触力和机器人接近轴不可直接观测。双手协作和非刚体物体尚未建模。
- 连续旋转对称物体可用 `--continuous-symmetry-axis` 按 yaw 等价类处理；当前假设该对称轴穿过 CAD 原点。离散对称或偏心对称轴尚未建模。
- SAM/RTMPose/WAPR/GraspGenX 的原始输出格式并不天然一致；本仓库严格定义了交换合同，但相机采集、RGB-D 米制对齐和各后端导出钩子仍需接到你的具体设备与数据目录。
- 仿真通过只证明该抓取在当前资产、摩擦和质量参数下可执行；它不替代真实机器人上的低速、限力验证。
