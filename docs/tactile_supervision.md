# Task-conditioned tactile supervision for one-shot manipulation

本文定义 `tactile_agent` 的触觉监工合同。它面向 one-shot 分阶段执行器，不让触觉模型
生成机械臂目标或关节命令。示教与 agent 仍决定任务意图和阶段目标，MoveIt 仍负责运动
规划；触觉监工只回答三类问题：当前接触是否符合本阶段预期、该阶段是否已经完成、是否
出现需要停止或恢复的异常。

本文是设计目标，不表示真机 runner 已经接入触觉。当前
`scripts/run_autonomous_pick.py` 仍使用 Hand-E 的 `gOBJ`、开口和 fault 作为抓取与持物
证据；通用 `TaskSkill` / `AgentLoop` 仍是 review-only。

## 1. 分层合同

```text
ViTai SDK channels
  RGB / diff / depth / markers / XYZ / slip / force6d / timestamp
        │
        ▼
sensor adapter
  serial→left/right, calibration, timestamp, freshness, frame transform
        │
        ▼
tactile primitives
  contact patch, per-finger normal load, shear, slip, task-axis torque, ...
        │
        ▼
task profile + temporal predicates
  grasp_acquisition_verified, object_remains_coupled, jam_detected, ...
        │
        ▼
stage supervisor
  continue / complete / hold / cancel_and_hold / recover / reobserve
```

原始通道、触觉原语和任务 predicate 必须是三层不同的数据。比如
`FORCE6D_VECTOR[2] > threshold` 不是一个可迁移的任务规则；`left_normal_load` 是经过
安装坐标与零点修正的原语，而 `grasp_acquisition_verified` 才是 GRASP 阶段可消费的
语义 predicate。

触觉监工不代替 Franka 的碰撞、力矩限制、急停或其他安全功能。ViTai 的图像和
软件估计力存在采集、推理与进程通信延迟，只能作为任务级监督证据。

## 2. SDK 通道与适用范围

本机 `pyvitaisdk 1.0.11` 的 `GFDataType` 暴露以下通道。SDK 的枚举存在不等于每个
通道已经在两只 GF220 上完成精度、单位和延迟验证。

| SDK 通道 | 主要信息 | 适合的用途 | 不应直接承担的用途 |
|---|---|---|---|
| `TIME_STAMP` | SDK 时间 | 同步、延迟和掉帧检查 | 接触判断 |
| `CALIBRATE_IMG` | 无载参考图 | 零点与标定 provenance | 在线阶段判断 |
| `RAW_IMG` | 原始相机画面 | 光路、曝光、硬件诊断 | 跨设备直接比较 |
| `WARPED_IMG` | 几何校正后的触觉 RGB | 接触区域、材质/纹理、学习特征、审计回放 | 未训练模型下的绝对力 |
| `DIFF_IMG` | 相对标定图的图像差 | 接触出现、面积和形变变化 | 已标定的物理深度 |
| `DEPTH_MAP` | 凝胶表面形变/深度估计 | 压入量、接触面积、接触面形状 | 未验证单位下的绝对位移 |
| `MARKER_IMG` | marker 可视化 | marker 跟踪诊断 | 控制状态本身 |
| `MARKER_ORIGIN_VECTOR` | marker 无载位置 | 每只传感器的参考 | 动态剪切 |
| `MARKER_CURRENT_VECTOR` | marker 当前坐标 | 接触形变场 | 不经参考差分的跨次比较 |
| `MARKER_OFFSET_VECTOR` | marker 位移场 | 剪切、局部滑移、接触中心变化 | 直接当牛顿值 |
| `XYZ_VECTOR` | 三维形变向量场 | 接触几何、法向与切向形变 | 未核对布局/单位时的硬门限 |
| `SLIP_STATE` | SDK 滑移估计 | 滑移候选事件 | 未做实物验证时的唯一停止依据 |
| `FORCE6D_VECTOR` | 估计六维 wrench | 力带、剪切、外载、任务轴力矩 | 安全认证急停或未经坐标变换的双指求和 |

现有 `/home/vitai/fhh` LeRobot 驱动只发布 `WARPED_IMG` 和可选
`FORCE6D_VECTOR`，并把后者记为 `[fx, fy, fz, tx, ty, tz]`；厂商单位、轴方向、符号、
噪声和延迟尚未在这套安装上确认。其他 SDK 通道需要显式扩展采集桥，不能在配置中写了
通道名就声称已经可用。

`SLIP_STATE` 的 SDK 枚举为 `CONTACT=1`、`INCIPIENT_SLIP=2`、`PARTIAL_SLIP=3`、
`COMPLETE_SLIP=4`、`NO_OBJ=5`、`STEADY_HOLD=6`，内部默认使用 5 帧历史。单帧状态不能
直接触发硬停，至少要经过窗口消抖并与 marker、force 或 contact patch 交叉确认。

通道选择遵循“只订阅形成当前原语所需的数据”。名义视频流为 30 Hz，但 2026-10-23
在当前两只 GF220 上测得的新视觉帧约为 24.4 Hz，force6d 约为 19 Hz，全部重通道同时
请求约为 18.8 Hz。每个通道必须携带自己的 SDK 时间戳/sequence；低频力值在图像帧之间
复用时，不能把相同值当成多次新证据。

### 2.1 当前两只 GF220 尚未达到执行门条件

2026-10-23 发现的 SDK serial 是：

```text
GF220GF22000184603F72A
GF220GF22000196FB398F5
```

逻辑 left/right 尚未固化。当前设备实际输出 640×480，而 SDK 的 GF225 默认流 profile 是
640×360。使用默认构造时，自动 warp 会误检重复角点，使 240×240 `WARPED_IMG` 接近全
黑；若启动后立即 `calibrate()`，还可能把无效启动帧设为基线，继而产生虚假 marker、slip
和 force。

现有 fhh 驱动正使用默认流并立即校准，而且没有请求 SDK `TIME_STAMP`。SDK 对同一缓存帧
的重复读取会被驱动用新的 host time 和 write count 发布，因此目前的“新鲜”标志不可信。
在它用于监工前必须：

1. 明确使用与 GF220 匹配的 640×480 MJPEG 流；
2. 为每只传感器固化独立 warp ROI，并验证 warped 图的动态范围；
3. 等待 SDK timestamp 前进且连续有效图像到达后，再用有效帧显式校准；
4. 发布 SDK timestamp，并按它去重、判 stale；
5. 独立记录 image/marker/slip/force 的更新时间；
6. 用已知载荷和力臂核对 force6d 的轴、符号和单位。SDK 对力矩单位存在 N·m 与 N·mm 的
   文档冲突，物理力矩阈值在实物校准前必须保持禁用。

## 3. 传感器数据合同

每只传感器的样本至少包含：

```text
sensor_id              stable SDK serial, never /dev/videoN
logical_side           left | right
capture_monotonic_ns   acquisition-process monotonic time
sequence               strictly increasing published frame number
calibration_id         immutable no-load calibration artifact id/hash
mount_frame            calibrated sensor frame name
status                 VALID | STALE | OFFLINE | UNCALIBRATED | DESYNCED
channels               only the requested SDK outputs
```

`STALE/OFFLINE/UNCALIBRATED/DESYNCED` 是显式状态。驱动为保持固定张量形状而填充的黑图或
零力不能解释为“未接触”。required 传感器无效时，相应 predicate 必须为 `INVALID`，执行器
fail closed。

两指融合前还需要：

- 固定且经过核对的 `T_tcp_sensor_left` 与 `T_tcp_sensor_right`；
- 每只传感器的向内指面法向 `n_inward_sensor`；
- SDK 六维量的次序、单位和正负号验证；
- 无接触零点、温漂和重新标定时间；
- 两路最大允许帧龄 `max_age` 与时间差 `max_skew`。

若传感器坐标中的 wrench 为 `(F_s, tau_s)`，变换到共同 TCP 坐标时应使用：

```text
F_tcp   = R_tcp_sensor F_s
tau_tcp = R_tcp_sensor tau_s + r_tcp_sensor × F_tcp
```

然后才能对左右两指求外部合 wrench。平行夹持中的两侧法向力方向相反，净合力接近零并不
表示没有夹紧。因此系统必须同时保留：

1. 每指向内法向夹持力；
2. 左右法向力的最小值和不平衡度；
3. TCP 上的外部合力；
4. 指定任务轴上的合力矩。

## 4. 标准触觉原语

每个原语输出 `value + unit + confidence + timestamp/window + validity + sources`，而不是只
输出一个无来源布尔量。

| 原语 | 首选输入 | 表达的物理含义 |
|---|---|---|
| `sensor_healthy` | timestamp、sequence、calibration id | 所需传感器新鲜、同步且已标定 |
| `contact_present_left/right` | diff/depth/RGB、marker、normal load | 每个指面是否形成接触 |
| `contact_patch_left/right` | RGB/diff/depth/XYZ | 面积、中心、形状及其变化 |
| `bilateral_contact` | 两侧 contact | 平行夹爪两指是否都接触物体 |
| `normal_load_left/right` | force6d + 指面法向 | 每指向内法向夹持负载 |
| `grip_load_balance` | 两侧 normal load | 两指受力是否严重偏置 |
| `shear_load_left/right` | force6d、marker offset | 指面切向载荷与方向 |
| `net_external_wrench` | 两侧已变换 force6d | 物体/环境施加在 TCP 上的合力与力矩 |
| `task_axis_torque` | net wrench + task axis | 旋拧、卡滞或倾覆相关的轴向力矩 |
| `indentation` | depth/XYZ/diff | 凝胶压入程度，适合柔软/易碎物体 |
| `slip_onset` | slip state、marker/RGB 时序、shear 变化 | 初滑候选事件和方向 |
| `stable_hold` | contact、load、slip、gripper aperture | 一段时间内持续持物 |
| `unexpected_contact` | approach 阶段任一 contact/load | 自由空间运动中出现了不应有的接触 |
| `overload` | normal/shear/wrench/indentation | 超过该任务已验证上限 |
| `support_contact` | load transfer、normal force、torque、运动状态 | 放置时物体开始由支撑面承载 |
| `release_confirmed` | contact patch/load 衰减 + gripper open | 两指已经与物体脱离 |

`object_stays_placed` 不能由释放后的触觉单独证明；触觉最多证明卸载和脱离，物体是否仍在
目标区域必须与外部 RGB-D 或其他场景传感器融合。

## 5. 按任务选择 profile

agent 不直接挑某个 SDK 数组和临时阈值。skill compiler 根据示教阶段和任务语义选择一个
已经标定、版本化的 supervision profile；agent 运行时只能在该 profile 声明的原语和
恢复动作内做选择。

| 任务 profile | 核心原语 | 主要通道 |
|---|---|---|
| `rigid_parallel_jaw_pick` | bilateral contact、每指法向力、平衡、stable hold、slip | force6d + diff/depth；RGB 留作空间证据 |
| `fragile_or_deformable_pick` | contact patch、indentation、法向力上下限、overload | RGB/diff/depth/XYZ + force6d |
| `lift_and_transport` | stable hold、shear、slip、外部合 wrench | force6d + marker 时序 + slip state |
| `placement_and_release` | support contact、载荷转移、力矩稳定、release confirmed | force6d + contact patch，外加外部 RGB-D |
| `insertion_or_connector` | 横向力、接触位置、倾斜力矩、jam | force6d + task-axis torque + RGB/depth/marker |
| `twist_or_lid` | task-axis torque、切向剪切、接触保持、slip | torque + marker/RGB 时序 |
| `surface_follow_or_wipe` | 法向力带、接触中心连续性、切向载荷 | force6d + depth/contact patch |
| `tactile_exploration` | 局部几何、纹理、材质/状态分类 | RGB/depth/XYZ；力只约束安全接触带 |

这里不是“一种任务只用一种输出”。低维力/力矩适合可解释的确定性门，RGB、depth 和 marker
适合描述接触发生在指面的哪里、形变分布和局部滑移。通常由多个原语共同形成阶段
predicate，原始 RGB 也同时保存为审计证据。

嵌套杯顶层抓取属于薄壁柔性抓取、内/外杯沿角色匹配和嵌套分离的组合 profile。该 profile
不套用通用刚体抓取的左右图像或原始法向力对称假设；一指独立匹配 INNER 模板，另一指独立
匹配 OUTER 模板，具体合同见
[`docs/nested_cup_top_pick.md`](nested_cup_top_pick.md)。

### 5.1 下压旋转安全瓶盖

药瓶示范中的 `TWIST_1 -> REGRASP_1 -> ... -> TWIST_3 -> REGRASP_3` 是人手动作证据，
不要求机器人固定重抓三次。运行时将它合并为一个 `TACTILE_TWIST_REGRASP_LOOP`。每次下压
前先采集“已夹盖但未下压”的双指同步 FORCE6D 基线。执行有界下压增量后，只有双指沿
瓶轴的反作用增量连续两个新窗口都稳定、平衡并达到 profile 工作带，才从
`PRESS_ENGAGE` 进入正式 `TWIST`；进入旋转前不要求小角度试转。

设瓶轴单位向量为 `a`，下压反作用符号为 `p`，第 `i` 指的预下压基线为 `b_i`：

```text
f_axis_i   = p * (dot(F_i, a) - b_i)
f_axis_sum = f_axis_left + f_axis_right
```

`f_axis_sum`、每指增量、左右平衡度和窗口内波动共同组成下压接合门；这些量都相对本次
重抓后的新基线计算。

两指在瓶盖两侧的笛卡尔切向方向相反，因此不能把两个三维力向量直接相加。设瓶轴单位向量
为 `a`，第 `i` 指从瓶轴指向接触点的径向向量为 `r_i`，期望旋转符号为 `s`：

```text
t_i       = s * normalize(a × r_i)
f_tan_i   = dot(F_i, t_i)
f_tan_sum = f_tan_left + f_tan_right
```

其中 `F_i` 必须先去无载偏置，并从各自传感器坐标变换到同一瓶盖任务坐标。转换规则是：

- 下压后的 `f_axis_sum` 连续稳定并满足双指平衡门：确认轴向压紧，允许开始正式旋转；
  没有建立轴向反作用时追加一个有界下压增量，在预算内重试；
- 正式旋转的角度和分段来自任务几何与腕部行程，不由切向力数值决定；旋转期间
  `f_axis_sum` 监督压紧是否保持，`f_tan_sum` 只监督双指接触是否持续、方向是否一致和
  受力是否稳定；
- 腕部旋转余量耗尽且触觉仍稳定：微开卸载，确认轴向与切向力回到卸载带后回腕并重新
  夹盖；随后重新采集预下压基线并通过轴向下压门，才开始下一段；
- 已经建立过旋拧负载，随后 `f_tan_sum` 在双侧接触仍存在时持续降到释放带：停止回腕循环，
  进入小幅轴向提盖验证；
- 单指切向力塌陷、两指严重失衡或滑移持续：进入滑移恢复，不把它误判为螺纹脱离；
- 数据过期、不同步、丢失接触或超过已标定负载：保持并停止推进。

示范中的三次重抓只给出本次任务的最大重试预算，不要求机器人一定重抓三次。绝对阈值必须
来自当前两块指垫的版本化实测 profile；未完成 serial 到左右指、安装坐标、轴符号、无载
偏置和单位确认时，这个闭环只能记录 shadow 决策。

## 6. one-shot 阶段映射

| 阶段 | 触觉监工关注点 | predicate / event | 推荐动作 |
|---|---|---|---|
| `APPROACH` | 传感器健康；不应接触 | `sensor_healthy`, `unexpected_contact` | 无接触则继续；意外接触先 cancel/hold，再撤离或重观测 |
| `PRESHAPE` | 两指仍有净空；偶发擦碰 | `sensor_healthy`, `unexpected_contact` | hold 或退回重新 ground |
| `GRASP` | 双侧接触、法向负载进入任务带、接触不严重偏置 | `grasp_acquisition_verified` | 完成；单侧/过载则停止闭合并按 profile 退开或重试 |
| `LIFT` | 持物持续、无初滑/失联/过载 | `object_remains_coupled` | 继续抬升；异常时 cancel_and_hold，停止确认后再恢复 |
| `TRANSPORT` | 动态剪切、初滑、惯性外载、扭矩 | `object_remains_coupled` | 减速/hold；允许时重规划，首版不自动加力 |
| `PLACE` | 支撑接触、载荷从夹爪转移、姿态趋稳 | `placement_support_verified` | 通过后才允许 RELEASE |
| `RELEASE` | 夹爪打开、接触和负载衰减 | `gripper_open_verified`, `release_confirmed` | 触觉脱离后用外部视觉验证 `object_stays_placed` |
| `RETREAT` | 已脱离且无二次碰撞 | `object_stays_placed`, `unexpected_contact` | 安全撤离；触碰则停止并重观测 |

初版刚体骰子 profile 可以将已有语义保持不变：

```text
grasp_acquisition_verified =
    sensor_healthy
    AND bilateral_contact
    AND normal_load_in_valid_band(left, right)
    AND no_overload

object_remains_coupled =
    sensor_healthy
    AND bilateral_contact
    AND stable_hold
    AND NOT slip_onset
    AND no_overload
```

这些触觉证据与 Hand-E `gOBJ`、开口保持是融合关系。第一版宜使用一致性检查：二者一致才
推进；若冲突则进入 `HOLD/INVALID` 并保存证据，不能让其中一个静默覆盖另一个。

### 6.1 主动触觉试探合同

薄壁物体允许使用离散的 step-and-observe 试探闭环，但不把单帧图像直接接成高速伺服：

```text
GF220 sidecar
  → 双指同步新鲜时间窗
  → 纯 evaluator
  → 有界确定性 probe policy
  → task runner / executor
```

evaluator 只返回状态、左右 patch 特征、有效性、置信度、连续通过数和动作请求；它不持有
USB、串口或 ROS 句柄，也不发送夹爪或机械臂命令。policy 只可从 profile 声明的动作集中
选择微闭、等待、微开卸力、请求 TCP 修正、接受或终止。runner 校验绝对开口、低速度/低力
上限、单步幅度、TCP 修正、累计位移和重试预算后才执行。

一次试探循环为：

1. 以低速度和低力上限向一个绝对开口微闭合，并记录命令完成的 host monotonic time；
2. 等命令停止和凝胶稳定，再采 4--6 个时间晚于命令完成、SDK timestamp 各不相同的新帧；
3. 评估 contact patch、左右独立法向载荷、压入/屈曲、帧龄和双指时差；
4. 无接触时在剩余闭合预算内再次微闭；
5. 单边、越出 ROI 或方向不符时先只返回拟议修正；微开卸载后，必须用后续新鲜窗口确认
   脱离，才能授权有界 TCP 修正；
6. 双侧形状正确但载荷偏低时再微闭一步；
7. 过载、屈曲、模糊多带、传感器 INVALID 或预算耗尽时卸力、保持或终止；只有 profile
   配置了经过实物标定的 task-frame 浅层修正时，分离接触带才可作为“夹深了”的假设，在
   卸载并确认脱离后尝试该修正，不能把两条光学边直接判为夹住两只杯；
8. 只有连续 K 个新鲜窗口均满足 profile 才接受。

承载接触时不得移动 TCP；重复 SDK timestamp 不得累计稳定证据；同一位姿只开合而不修正
TCP，不能解决单侧或低位接触。Hand-E 只能统一改变开口，左右“独立力”表示分别测量和
判门，不表示可以分别命令两根手指。当前约 19--24 Hz 的有效通道频率适合 150--250 ms
的任务级试探窗口，不适合软件力伺服。现有阻塞式 runner 若要在一次长闭合命令中即时截停，
还需可取消/分段夹爪命令或长驻 Hand-E 会话。

“无斑纹/contact patch”只有在两指法向力也接近无载时才是 `NO_CONTACT`，policy 才可建议
下一次微闭。若 force 提示接触但图像没有稳定 patch，则是模态冲突，结果为
`CONTACT_UNRESOLVED` 并建议卸力，不能按“没夹到”继续闭合。这里的 150--250 ms 是命令
停稳、凝胶稳定和采集一个窗口的循环周期；evaluator 默认 `max_age_s=0.15`，参与判断的每个
样本在评估时仍须满足 150 ms 帧龄上限。

每个试探回合创建非空 `probe_context_id`，绑定当前 probe episode、固定 TCP 位姿和 profile。
只要 TCP 保持不动，微闭、微开、修正 proposal 和 release confirmation 使用同一 id；TCP
移动、重新 grounding、新回合、profile 或任一 calibration id 改变时必须创建新 id 并重置
stability gate。缺失 id 只允许 shadow evaluation，不能累计 `CAPTURED` 或授权 TCP 修正。

TCP 修正使用
`authorize_tcp_adjustment(proposal, release_evaluation, release_receipt, config, now_s=...)`。
runner 先完成 proposal 要求的正向开口并生成 `ReleaseCommandReceipt`；随后以命令完成时间作为
`not_before_s` 评估 release 窗口。只有 release 状态为 `NO_CONTACT`、所有必需通道的 SDK 与
host timestamp 都严格前进且 host 样本均在 release 完成之后，并且 context、profile 与左右
calibration id 全部一致时，才能签发授权。默认 TTL 为从 proposal 评估时刻起 1 s，配置硬
上限为 5 s。授权 token 只能由 `TcpAdjustmentAuthorizationLedger` 消费一次；真机 runner
还必须在移动 TCP 前把已消费 id 原子持久化到 stage journal，避免进程重启后重放。

当前 GF220 设备和 sidecar 尚未达到执行门条件，因此上述 evaluator 与 probe policy 全部处于
shadow mode。`CLOSE_STEP`、`RELEASE`、`RELEASE_THEN_ADJUST`、`APPLY_TCP_ADJUSTMENT` 和
`ACCEPT` 都只能记录为建议，不能发出夹爪/机械臂命令，也不能停止、完成或推进真机阶段。

## 7. 时间窗、滞回和 predicate 状态

单帧不能完成抓取或滑移判断。每个原语配置：

- `max_age_ms` 与双指 `max_skew_ms`；
- `window_ms` 或最近 `N` 帧；
- 进入阈值与退出阈值，形成滞回；
- 最少连续通过/失败帧数；
- 阈值单位、坐标系和 calibration id；
- 缺失通道时是否允许降级。

predicate 使用三态：

```text
SATISFIED     有效证据在完整时间窗内通过
UNSATISFIED   有效证据明确不通过
INVALID       陈旧、离线、未标定、不同步或所需通道缺失
```

`INVALID` 不等于 `UNSATISFIED`，更不能被转成全零值。阶段完成条件只有
`SATISFIED` 才能推进。连续 guard 在运动中变为 `UNSATISFIED/INVALID` 时，应请求取消并
保持；必须等底层确认运动已停止后，才能执行张开、撤离或重规划。

阈值不能凭 SDK 输出范围猜测。每只传感器先采无载基线和温漂，再执行已知方向按压、剪切、
扭转、稳定夹持与人为初滑，记录中位数/MAD、响应符号、饱和、延迟和重复性。物理单位确认
前可以使用带 calibration id 的归一化分数做实验，但不能把它标成 N 或 Nm。

## 8. 配置与证据格式

触觉硬件和阈值不写进 `TaskSkill v3`。同一 one-shot skill 应能在不同夹爪/传感器安装上
复用；近期使用独立、显式加载的配置：

```json
{
  "schema": "demo2gripper.tactile_supervision_config.v1",
  "profile_id": "rigid_parallel_jaw_pick.v1",
  "status": "review_only",
  "sensors": {
    "left": {
      "sdk_serial": "GF220...",
      "mount_frame": "left_tactile",
      "calibration_id": null,
      "required": true
    },
    "right": {
      "sdk_serial": "GF220...",
      "mount_frame": "right_tactile",
      "calibration_id": null,
      "required": true
    }
  },
  "primitives": {
    "bilateral_contact": {
      "channels": ["diff_img", "depth_map"],
      "window_ms": 150,
      "max_age_ms": 150,
      "thresholds": null
    },
    "normal_load_band": {
      "channels": ["force6d"],
      "window_ms": 200,
      "thresholds": null
    },
    "no_slip": {
      "channels": ["slip_state", "marker_offset_vector"],
      "window_ms": 120,
      "thresholds": null
    }
  },
  "predicates": {
    "grasp_acquisition_verified": {
      "all": ["bilateral_contact", "normal_load_band", "no_overload"]
    },
    "object_remains_coupled": {
      "all": ["bilateral_contact", "normal_load_band", "no_slip", "no_overload"]
    }
  }
}
```

`null` threshold 或 calibration id 会保持 `review_only` 并阻止执行门启用。以后若将
`supervision_profile_id` 或 `continuous_invariants` 纳入 skill，应显式升级 schema，不能让
v3 reader 静默丢弃新字段。

每次阶段判断保存 `demo2gripper.tactile_evidence.v1`：profile/config hash、
`probe_context_id`、两只传感器序列号、左右 calibration id、SDK 与 host 采样时间范围、
`not_before_s`、原语数值和单位、predicate 三态、置信度、建议动作和原始图像/数组 artifact
引用。涉及 TCP 修正时还要记录 proposal evidence key、release receipt/command id、release
evidence key、authorization id、签发/过期时间和一次性消费结果；真机 stage journal 在物理
动作前原子持久化已消费 id。原始高带宽数据留在独立 NPZ/Zarr/视频中，stage report 只引用
它们。

## 9. 进程与执行器集成

厂商 SDK 留在能够稳定运行 `pyvitaisdk` 的 fhh 环境中。独立 tactile sidecar 拥有 USB
句柄、同步两指、计算原语并通过版本化序列化协议向 runner 发布最新状态；ROS/MoveIt
进程不导入厂商 SDK。在线状态可通过本机 Unix socket 或共享内存传递，阶段证据再原子写入
JSON/NPZ。不得靠反复读取 `/dev/videoN` 推断左右手指，映射必须使用 SDK serial。

现有 runner 的 MoveIt 调用是阻塞式阶段调用，因此只能可靠地做阶段前/后检查。要实现
LIFT/TRANSPORT 中的连续滑移监工，执行器还需具备：

1. 运动期间并发读取 supervisor event；
2. 可取消的 MoveIt action 或等价的分段执行；
3. `cancel requested → execution stopped` 的确认握手；
4. 停止后才根据阶段策略执行 hold、开爪、撤离或重新规划；
5. 全程记录触发事件和实测机器人状态。

`AgentLoop` 目前可以消费融合后的 predicate 名称和 `recoverable_failure`，但没有数值证据、
严重级别或一般化的连续 guard 取消通路。近期由 adapter 把触觉结果映射为已有
`grasp_acquisition_verified` / `object_remains_coupled`；在通用执行器落地时再升级反馈
schema。

## 10. 实现顺序

1. **设备资格确认。** 固定左右 serial 映射；为两只 GF220 固化 640×480 stream 与独立
   warp ROI；确认两指空载、有效帧后校准且能稳定输出非零动态范围的 warped 图。
2. **观测模式。** 修正现有 sidecar 的默认 profile、启动帧校准和缓存帧去重，再记录
   warped RGB、marker/depth、force6d、slip、SDK 时间戳和有效性，但不影响机器人动作。
3. **轴与单位标定。** 做已知方向按压/剪切/扭转和初滑数据集，确定 frame、符号、噪声、
   迟滞、延迟与阈值；版本化 calibration。
4. **骰子 MVP。** 先接 `sensor_healthy`、`bilateral_contact`、`normal_load_band`、
   `no_slip`、`no_overload`，与 Hand-E 证据并行记录。
5. **阶段边界 gate。** 在 GRASP 完成和 LIFT 完成处启用 predicate；至少重复 20 次并统计
   误报、漏报和动作成功率。
6. **连续 guard。** 补可取消执行与停止确认后，才允许 slip/contact-loss/overload 在运动中
   触发 `cancel_and_hold`。
7. **增加 profile。** 依次验证易碎抓取、放置释放、插入、旋拧；每种 profile 使用自己的
   task frame、原语组合和标定阈值。

在第 3 步完成前，RGB/force/torque/slip 都只是可记录通道；在第 5 步完成统计验证前，全部
触觉判断和 probe policy 都必须保持 shadow，不能发送夹爪或 TCP 动作，不能停止、完成或
推进任何真机阶段，也不能替代当前 Hand-E 判据。
