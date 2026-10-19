# Imitrob 电钻整手握持实跑

这条链路使用 Imitrob v2 `powerdrill/metascrewing/S1/left` 的真实双目 RGB-D 素材，选用 C2 视角，从 `F0006–F0234` 每三帧取一帧，共 `77` 帧、`10 FPS`。输出不是合成示例，也没有把 Imitrob 的 tracker、6DOF 或 bbox 真值传给分割、手姿态、物体位姿、候选选择或控制。

## 实际结果

| 模块 | 后端与实际输出 | 结果 |
|---|---|---|
| 目标发现 | SAM 3，文本 `power drill` | 找到真实电钻，但初始 mask 包含附着的黑色 tracker |
| 视频分割 | SAM 2.1 Hiera-S | 分别稳定跟踪主动手和去除 tracker 后的电钻，77/77 帧有输出 |
| 手姿态 | RTMPose-m Hand5 ONNX，物理 GPU 7 | 77/77 帧输出 21 个 2D 关键点；视频直接显示原始 2D 结果 |
| 物体位姿 | 原版 FoundationPose，物理 GPU 7 | CAD 投影 bbox IoU 中位数 `0.939`，中心误差中位数 `3.3 px` |
| Panda 候选 | 官方 GraspGenX generator + discriminator，物理 GPU 7 | 100 个候选，分数 `0.720889–0.928198` |
| 整手意图 | RTMPose 2D → CAD 重投影 → 双侧 CAD 接触检查 | 完整握持段 `[0,77)`，选择 candidate `62` |
| candidate 62 | GraspGenX + CAD 接触线 | 模型分数 `0.764741`，CAD 双侧接触宽 `34.415 mm` |
| Isaac 5.1 | Panda 差分 IK、双指接触、抬升、保持 | candidate 62：IK/双指接触通过，抬升/保持失败 |

电钻是整手 power grasp，不适合把拇指—食指 pinch 的接触阈值放宽后继续使用。`infer_power_grasp_intent.py` 保留可见手势真正支持的结论：它估计人手在 CAD 上选择的抓持区域，再从 GraspGenX 结果中选择闭合线与 CAD 存在双侧交点的 Panda 候选。候选的接近轴来自 GraspGenX；单视角、整手遮挡的视频本身不能可靠恢复机器人接近轴。

核心命令为：

```bash
python scripts/infer_power_grasp_intent.py \
  --t-camera-object "$RUN/foundationpose_c2_registered_approx/T_camera_object.npy" \
  --object-points "$RUN/object_points_m.npy" \
  --keypoints-2d "$RUN/hand_pose_c2_registered_approx_r10/hand_keypoints_2d.npy" \
  --scores-2d "$RUN/hand_pose_c2_registered_approx_r10/hand_keypoint_scores_2d.npy" \
  --intrinsics "$RUN/input_c2_registered_approx/K.npy" \
  --candidates "$RUN/graspgenx/candidates.npz" \
  --intent-output "$RUN/grasp_intent_power.json" \
  --selected-output "$RUN/selected_grasp_power.json"
```

其中 `RUN=/mnt/sdb/lzr/demo2gripper_runs/imitrob_powerdrill_s1_left`。

## Imitrob depth 的已确认问题

官方发布的 `Depth/C2*.png` 来自 raw rectified depth，而不是 aligned-depth。发布的 `parameters.json` 只含 color `K_C2`，不含 depth 内参或 depth→color 外参：

- [官方采集同时录制 raw 与 aligned depth](https://github.com/imitrob/imitrob_dataset_code/blob/7111ab71e6bc0379fa4464724e02f8fefb6eddd7/data_acquisition/cam_calib/launch/rec_data_sixdof.launch#L10-L19)
- [官方 exporter 默认选择 raw depth](https://github.com/imitrob/imitrob_dataset_code/blob/7111ab71e6bc0379fa4464724e02f8fefb6eddd7/data_acquisition/extraction/extract_data_from_bag_BB.py#L549-L560)
- [官方 topics.yaml 同样选择 raw depth](https://github.com/imitrob/imitrob_dataset_code/blob/7111ab71e6bc0379fa4464724e02f8fefb6eddd7/data_acquisition/extraction/topics.yaml#L1-L9)

该序列实测 depth→RGB 约为 `(-38,+22) px`。本次工程贯通保留原始数据，另生成明确标记为 `approx depth align` 的近似平移版本：

```bash
python scripts/prepare_imitrob.py \
  --sequence-dir /mnt/sdb/lzr/datasets/imitrob/powerdrill/metascrewing/S1/left \
  --output-dir "$RUN/input_c2_registered_approx" \
  --camera C2 --frame-step 3 \
  --depth-translation-x-px -38 \
  --depth-translation-y-px 22
```

未配准 depth 的 FoundationPose CAD bbox IoU 中位数为 `0.469`、中心误差为 `36.3 px`；近似配准后分别为 `0.939` 和 `3.3 px`。这足以检查模块接口和候选选择，但不是严格的毫米级 6D pose benchmark。要做正式度量，应向数据集作者索取 `521_bag_2023-03-08-15-15-13.bag`，重新导出其中的 `/aligned_depth_to_color/image_raw`。

## 产物

- `powerdrill_real_pipeline_annotated.mp4`：77 帧真实模块输出四宫格视频
- `powerdrill_real_pipeline_preview.png`：中间帧预览
- `graspgenx/grasps.yml`、`graspgenx/candidates.npz`：官方 GraspGenX 实际输出
- `grasp_intent_power.json`：人手抓持区域与 CAD 接触意图
- `selected_grasp_power.json`：candidate 62 的 Panda TCP
- `isaac_candidate62/result.json`、`isaac_candidate62/candidate62.mp4`：candidate 62 仿真结果
- `isaac_candidate56/result.json`、`isaac_candidate56/candidate56.mp4`：次选 candidate 56 对照

这段素材从人已经握住电钻开始，因此能验证“抓哪里”和握持期间的相对关系，不能学习伸手、首次接触和闭合时刻。

## Isaac 5.1 物理验证

YCB `035_power_drill` 以 `0.732588 kg` 动态刚体和凸分解碰撞体导入现有 RoboDojo Isaac Sim 5.1。验证使用物理 GPU 7，不固定物体；已知仿真摆放位姿只用于隔离测试 object-relative candidate，不进入真实视频感知。

candidate 62 的结果是：

- IK 可达；
- 双指接触稳定 5 个 step，峰值力左/右为 `5.50 / 3.55 N`；
- Panda 的抬升目标可达；
- 电钻最大只抬升 `0.438 mm`，随后失去接触；
- `LIFT=FAIL`，`HOLD=FAIL`。

次选 candidate 56 在同一摆放下预抓取 IK 不可达。当前结果证明感知、相对抓持区域和候选生成接口已经贯通，也明确否定了“candidate 62 在 Panda 上已经可抓”这一结论：双指碰到物体不等于形成稳定力闭合。后续选择器需要在最终输出前加入机器人碰撞/可达性与物理抬升过滤。
