# Agentic 示教控制原型

> **当前状态（2026-10-21）：** 本文的 `TaskSkill`/`AgentLoop` 通用合同仍是
> `review_only`，但仓库已经增加针对当前 FR3 + Hand-E 桌面抓取的确定性真机 runner，
> 并在 Run 05 完成 `OBSERVE` 到 `LIFT`。两者的关系、真机闭环和验证边界见
> [`refined_real_robot_framework.md`](refined_real_robot_framework.md)。

这个原型把一次人手示教编译成“任务阶段意图 + 侧夹相对几何”。示教中拇指和
食指形成侧夹的时刻，提供两项可迁移量：物体 anchor 到捏合中心的偏移，以及基座坐标中
无方向的水平夹爪轴。运行时 agent 根据当前物体、工作空间和阶段意图生成
`APPROACH/LIFT/PLACE/RETREAT` 目标；MoveIt 负责 IK、碰撞检查和到这些目标的轨迹。

```text
RGB-D 人手示教
  -> APPROACH / PRESHAPE / GRASP / LIFT / TRANSPORT / PLACE / RELEASE / RETREAT
  -> 阶段意图 + GRASP 的 base_aligned_planar 侧夹关系
  -> 当前场景 grounding + Hand-E 适配
  -> agent 在线生成各运动阶段目标
  -> MoveIt plan
  -> 执行反馈与阶段 guard
  -> 继续、重新观察、重新规划或恢复
```

## 当前合同

`src/demo2gripper/task_skill.py` 定义 `demo2gripper.agentic_task_skill.v3`：

- `SkillStage` 保存阶段区间、目标策略、grounding anchor、规划模式、前置条件、
  完成条件和失败策略。`GRASP` 使用 `object_relative_grasp`；自由空间运动使用
  `online_agent_plan`；`RELEASE` 使用 `task_intent`。
- `WaypointArtifactRef` 明确区分 `(N,3)` 的 `position_offsets_base_m` 与完整
  `(N,4,4)` 的 `poses_se3`。v3 中运动阶段不引用 waypoint artifact；该结构只用于读取旧合同和
  诊断。
- `ObjectGraspRelation.base_aligned_planar` 保存 base 轴对齐的 anchor 偏移和水平闭合轴。
  闭合轴是一条无方向的直线，`a` 与 `-a` 表示同一侧夹姿态。
- `TaskSkill` 自身包含 `status`、`execution_authorized`、`robot_execution_ready` 和
  `blocking_conditions`。`review_only` skill 不能授权执行。

当前无 CAD 的骰子 anchor 只有基座坐标中的位置，没有物体方向。因此这个关系可以
在物体平移后落地，也可用于方向对任务无影响的对称骰子；它不能随一个任意旋转的物体
一起旋转。完整旋转迁移需要点云模板配准或其他 6D 跟踪，以得到 `T_base_object`，
再把侧夹关系表示为真正的 `T_object_tcp`。

各阶段使用以下运行时目标策略：

| 阶段 | 目标来源 | 含义 |
|---|---|---|
| APPROACH / PRESHAPE | `online_agent_plan` | agent 根据抓取目标、可见性、净空和机器人约束选预抓取位姿 |
| GRASP | `object_relative_grasp` | 使用 anchor 偏移与无方向水平闭合轴决定侧夹终端约束 |
| LIFT / TRANSPORT | `online_agent_plan` | agent 根据“物体已夹住”和净空意图选择运行时目标 |
| PLACE | `online_agent_plan` | agent 根据任务放置区、支撑面与碰撞约束选放置目标 |
| RELEASE | `task_intent` | 放置完成 predicate 通过后打开夹爪 |
| RETREAT | `online_agent_plan` | agent 从实测释放位姿生成安全撤离目标 |

当前真机 runner 把“预抓取点在抓取点上方 `80 mm`”和“夹住后沿基座 `+Z`
抬升 `40 mm`”作为第一版确定性 agent 策略。这两个数值是 robot/task policy，不是从
人手轨迹回放出来的目标。

`src/demo2gripper/grounding.py` 对 `(N,3)` 和 `(N,4,4)` 使用不同返回类型。位置路径的
orientation 始终是 unresolved，不能通过 `require_full_pose()` 取得伪造姿态。

## 编译一次示教

```bash
.venv/bin/python scripts/compile_agentic_skill.py \
  --demo-dir runs/live/<demo-name>
```

默认输出到 `runs/live/<demo-name>/agentic_skill_prototype/`：

- `agentic_skill.json`：规范机器合同；
- `trajectory.npz`：示教分段、轨迹连续性和手物同步的诊断证据；
- `compiler_report.json`：阶段证据、路径连续性、支撑面净空和阻断条件；
- `preview.png`：阶段和基座坐标轨迹总览；
- `preview.mp4`：逐帧阶段标签和人手路径。

编译器保留手和物体的原始位置轨迹，用它们验证分段、连续性、支撑面净空和手物同步。
`trajectory.npz` 不是运行时目标，也不会被解释为需要复现的 TCP 曲线。运行时只消费编译后的
阶段意图和侧夹关系。

## Plan-only agent 循环

`src/demo2gripper/agent_loop.py` 实现：

```text
OBSERVE -> GROUND -> PLAN -> EXECUTE -> VERIFY
```

循环逐项检查阶段声明的 predicate。依赖物体/抓取目标的 anchor 失效时，会先取消正在
执行的运动并等待停止确认，再重新观察和规划。需要恢复撤离时，会进入独立的 RECOVERY
状态并等待恢复完成。对 `review_only`、未授权、机器人未就绪或仍有阻断条件的 skill，
循环不会产生 `EXECUTE` 或 `RECOVERY_RETREAT` 这类运动请求。

触觉监工遵守同一边界：它根据任务和阶段选择已验证的 RGB、形变、力、力矩或滑移原语，
再输出现有语义 predicate 或异常事件；它不生成运动目标。原始通道、时序融合、坐标变换、
失效语义和 profile 选择详见 [`docs/tactile_supervision.md`](tactile_supervision.md)。连续
guard 若在运动中失败，执行器必须先请求取消并确认停止，之后才能进入恢复动作。

## 通用 TaskSkill 执行器仍缺少的部分

`scripts/run_autonomous_pick.py` 已经能对当前骰子实验发送 MoveIt 和 Hand-E 命令，但它是
固定的 pick-to-lift 状态机，并不解释任意 `TaskSkill`。将通用 agent 合同升级为真机执行器
仍需要：

1. 当前物体的完整 SE(3)，或明确限定为不改变物体朝向的平移实验；
2. 将当前 Hand-E command frame 到 TCP 的实测变换升级为完整、可迁移的末端标定；
3. Hand-E、适配板和手指的 URDF/SRDF 与碰撞体；
4. 50 mm 开口约束、夹爪动作反馈和抓取成功判据；
5. 把现有 MoveIt/Hand-E primitives 接入 `AgentLoop`，并为每个阶段实现规划场景、碰撞
   检查、完成 predicate 和恢复策略。

这些条件仍应进入 `TaskSkill.blocking_conditions`。当前真机 runner 使用独立的双重执行
开关和逐阶段 fail-closed 检查，不能据此把任意 `review_only` skill 改成可执行。

## FR3 最小真机 smoke test

`scripts/moveit_lift_smoke.py` 是独立的硬件联通测试，不会执行整份 TaskSkill。它保持
当前 `fr3_link8` 朝向，只允许沿基座 Z 轴上移最多 20 mm，速度和加速度比例最多 5%。
默认只规划；arm-only 启动器不会加载 stock Franka gripper。

先在假硬件验证：

```bash
scripts/start_fr3_moveit_arm_only.sh --fake
```

另一个终端运行：

```bash
source /opt/ros/jazzy/setup.bash
source third_party/franka_ros2_ws/install_ros/setup.bash
/usr/bin/python3 scripts/moveit_lift_smoke.py
```

真机启动器会拒绝 generic 内核。启动到 realtime 内核后，使用 `--real`，先重复默认的
plan-only 命令；审核 `outputs/fr3_lift_smoke_plan.json` 后才可显式执行：

```bash
/usr/bin/python3 scripts/moveit_lift_smoke.py \
  --execute \
  --accept-arm-only-collision-model
```

这个动作只验证 MoveIt IK、FR3 控制器和短距离运动链路。它不验证 Hand-E TCP、夹爪
碰撞体、骰子定位或完整抓取。
