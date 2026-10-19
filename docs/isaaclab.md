# Isaac Lab 抓取验证

`scripts/isaaclab_validate_grasp.py` 把 `selected_grasp.json` 中的物体坐标系抓取位姿放入
Isaac Lab，使用 Franka Panda 执行一次确定性的
`REST -> PREGRASP -> APPROACH -> CLOSE -> LIFT -> HOLD`。目标运行栈固定为：

- RoboDojo 内置 Isaac Lab 源码提交 `afca7b09d60d8beb9c1cb28b43066499940b969b`
  （安装包版本 `0.54.3`，Isaac Lab 2.x API）
- Isaac Sim 5.1.0.0 / Isaac Sim PhysX
- `/mnt/sdb/lzr/univtac/conda_envs/RoboDojo` 中的 Python 3.11.16
- 米制；命令行和结果 JSON 使用 `XYZW`，Isaac Lab 内部使用 `WXYZ`

控制循环与接触判定直接对齐当前 RoboDojo checkout 中的
`scripts/tutorials/05_controllers/run_diff_ik.py` 和
`scripts/demos/sensors/contact_sensor.py`。脚本使用该版本的直接 tensor 接口、
`root_physx_view.get_jacobians()`、`force_matrix_w` 和 `WXYZ` 四元数约定，不依赖
其他版本的接口。

## 输入合同

抓取文件顶层必须包含：

```json
{
  "T_object_gripper_tcp": [
    [1, 0, 0, 0.00],
    [0, 1, 0, 0.00],
    [0, 0, 1, 0.10],
    [0, 0, 0, 1]
  ],
  "pregrasp_width_m": 0.08,
  "close_target_width_m": 0.0,
  "approach_distance_m": 0.10
}
```

`T_object_gripper_tcp` 把 canonical gripper TCP 中的点变换到物体坐标系。该 TCP 的 `+X`
是夹闭轴，`+Z` 是进给方向。预抓取位姿沿 TCP `-Z` 回退
`approach_distance_m`。`pregrasp_width_m` 是 REST、PREGRASP 和 APPROACH 的总开口；
CLOSE 从该开口逐步闭合到 `close_target_width_m`，以双指接触而不是候选模型猜测的物体宽度
作为停止条件。

Panda 的手指沿 `panda_hand` 的 `+/-Y` 运动。脚本固定使用
`R_canonical_tcp,panda_hand = Rz(-90 deg)`，使 `panda_hand +Y` 对齐 canonical `+X`，并保持
`+Z` 进给方向不变。`panda_hand` 原点到指尖中心 TCP 的默认平移为 hand `+Z 0.1034 m`，可用
`--tcp-offset-m` 修改。该值必须与 `convert_graspgenx.py --tcp-offset-m` 使用的值一致。

`--object-pose X Y Z QX QY QZ QW` 只设置物体在仿真世界中的初始真值位姿，不参与估计模式
的控制目标计算。`--estimated-object-pose` 接收感知模块输出的物体世界系位姿；相机系估计应在
调用脚本前通过外参变换到世界系。两个命令行位姿和位姿 JSON 都使用 `XYZW`。脚本在进入
Isaac Lab 边界时转换为 `WXYZ`，之后所有仿真 pose 和四元数运算均遵循 Isaac 5.1 的原生约定；
结果 JSON 再明确输出为 `XYZW`。估计模式按下式计算抓取目标：

```text
T_world_gripper_tcp = T_world_object * T_object_gripper_tcp
```

`--estimated-object-pose`、`--estimated-object-pose-json` 与 `--use-oracle-object-pose` 必须三选一。
前两者是管线验证入口；后者仅用于调试，在物体静置 0.5 秒后读取仿真真值并生成控制目标。
仿真真值在估计模式中只用于计算抬升指标和写入结果 JSON，不会传入抓取目标生成函数。

## 安装与运行

使用已经安装好的 RoboDojo 环境和其内置 Isaac Lab checkout：

```text
Python:    /mnt/sdb/lzr/univtac/conda_envs/RoboDojo/bin/python
Isaac Lab: /home/lzr/code/RoboDojo/third_party/IsaacLab
Robot USD: /home/lzr/code/RoboDojo/Assets/Robots/franka/Franka.usd
```

当前机器只使用物理 GPU 7；设置 `CUDA_VISIBLE_DEVICES=7` 后，它在进程内映射为
`cuda:0`。不导出相机数据时，`--experience` 可选 RoboDojo 的纯物理 headless 配置；内置方块用于验证安装、Panda 控制、双指接触与抬升链路：

```bash
OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=7 \
  /mnt/sdb/lzr/univtac/conda_envs/RoboDojo/bin/python \
  /home/lzr/code/demo2gripper/scripts/isaaclab_validate_grasp.py \
  --grasp-json /home/lzr/code/demo2gripper/examples/smoke_selected_grasp.json \
  --robot-usd /home/lzr/code/RoboDojo/Assets/Robots/franka/Franka.usd \
  --smoke-cuboid \
  --object-pose 0.55 0.0 0.04 0.0 0.0 0.0 1.0 \
  --use-oracle-object-pose \
  --experience /home/lzr/code/RoboDojo/third_party/IsaacLab/apps/isaaclab.python.headless.kit \
  --device cuda:0 \
  --headless \
  --result-json /home/lzr/code/demo2gripper/runs/smoke/isaaclab_grasp_result.json
```

真实动态刚体 USD 与感知位姿使用同一入口：

```bash
OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=7 \
  /mnt/sdb/lzr/univtac/conda_envs/RoboDojo/bin/python \
  /home/lzr/code/demo2gripper/scripts/isaaclab_validate_grasp.py \
  --grasp-json /home/lzr/code/demo2gripper/runs/real/selected_grasp.json \
  --robot-usd /home/lzr/code/RoboDojo/Assets/Robots/franka/Franka.usd \
  --object-usd /absolute/path/to/object.usd \
  --object-pose 0.55 0.0 0.05 0.0 0.0 0.0 1.0 \
  --estimated-object-pose-json /home/lzr/code/demo2gripper/runs/real/estimated_object_pose.json \
  --tcp-offset-m 0.1034 \
  --experience /home/lzr/code/RoboDojo/third_party/IsaacLab/apps/isaaclab.python.headless.kit \
  --device cuda:0 \
  --headless \
  --result-json /home/lzr/code/demo2gripper/runs/real/isaaclab_grasp_result.json
```

`--smoke-cuboid` 与 `--object-usd` 二选一。不要把 `--device` 写成 `cuda:7`；CUDA 可见设备
已经把物理 7 卡重映射为逻辑 0 卡。

### RGB-D 感知采集

指定 `--rgbd-output-dir` 时，验证器会在 REST 后导出一帧 RGB、米制 depth、OpenCV `K` 和
`T_world_camera`。此时必须把 experience 改为：

```text
/home/lzr/code/RoboDojo/third_party/IsaacLab/apps/isaaclab.python.headless.rendering.kit
```

输出结构为：

```text
<rgbd-output-dir>/
├── rgb/000000.png
├── depth/000000.npy
├── K.npy
├── calibration/T_world_camera.npy
└── evaluation/T_world_object.npy
```

`evaluation/T_world_object.npy` 是隔离的仿真真值，只供 `evaluate_object_pose.py` 评分；SAM、WAPR、`compose_world_object_pose.py` 和使用 `--estimated-object-pose-json` 的控制运行都不读取它。已经执行的 pudding-box 采集命令和实际目录见 [`perception_io.md`](perception_io.md)。

常用判定参数：

```text
--tcp-offset-m 0.1034       Panda hand 到 TCP 的 +Z 距离
--contact-threshold-n 1.0   单指正接触阈值
--lift-height-m 0.10        机械臂世界 +Z 抬升命令
--lift-success-m 0.05       物体位移成功阈值
```

## 状态和输出

机械臂使用 `DifferentialIKController` 的绝对 pose / DLS 模式。每个运动阶段最多运行 300 个
10 ms 物理步；末端连续 5 步满足 `15 mm / 10 deg` 误差才进入下一阶段。左右指各使用一个
`ContactSensor`，并通过 `filter_prim_paths_expr` 只统计目标物体的法向接触力。CLOSE 最多运行
120 步；左右指同时达到阈值才进入 LIFT，到最小开口仍未形成双侧接触就以 CLOSE 失败结束。

结果 JSON 的四个主判定为：

- `ik_reachable`：PREGRASP 和 APPROACH 都在超时前到达。
- `bilateral_contact`：CLOSE 中左右指曾同时超过接触阈值。
- `lifted`：物体相对 CLOSE 前的最大 `+Z` 位移达到 `lift_success_m`。
- `held`：HOLD 结束时物体仍高于成功阈值，且左右指当前都保持接触。

输出还包含各阶段步数、最大与最终接触力、最大与最终抬升距离、实际夹爪命令宽度，以及
TCP 转换参数。`target_pose_source` 标记 `external_estimate` 或 `sim_oracle_after_rest`；
`target_object_pose_xyzw` 是控制输入，`spawn_object_pose_xyzw` 和
`ground_truth_object_pose_after_rest_xyzw` 是仿真真值记录，便于定位感知、IK、接触或保持阶段
的失败。

## 运行前约束与已知限制

- `--smoke-cuboid` 使用脚本内定义的动态刚体，只用于贯通控制和接触验证。真实实验使用
  `--object-usd`；其根 prim 必须是一个动态刚体，并已包含碰撞体、质量和物理材质。脚本不会从
  纯视觉 mesh 猜测或生成碰撞；刚体嵌套在根 prim 以下的 USD 也不符合当前过滤路径合同。
- USD 和抓取文件都必须使用米。物体初始位姿应让其碰撞体位于地面上方；脚本只创建地面，
  不创建针对特定资产高度的桌面。
- Differential IK 验证可达性和闭环跟踪，不做全局无碰规划。若 PREGRASP 路径需要绕障，应先
  接入 cuRobo/MoveIt 规划，再复用 CLOSE、LIFT、HOLD 和接触判定。
- 物体质量、摩擦和碰撞近似会直接改变 `bilateral_contact`、`lifted` 和 `held`。跨物体比较前应
  固定这些 USD 属性和判定阈值。
- Isaac Sim 运行时来自现有 RoboDojo 环境。自定义 USD 的刚体位于
  `/World/envs/env_0/Object`；左右指传感器通过当前 API 的 `force_matrix_w` 读取针对该物体的
  过滤接触力。
