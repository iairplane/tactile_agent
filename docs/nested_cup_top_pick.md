# Nested disposable cup: top-cup-only pick

本文定义从一叠相互嵌套的一次性杯中只取出最上面一只的任务合同。假设杯口朝上，使用
FR3、Robotiq Hand-E、双指 ViTai 触觉和固定外部 RGB-D。该任务同时包含薄壁柔性抓取、
嵌套分离和数量验证，不能直接复用骰子的 `rigid_parallel_jaw_pick`。

这是设计与数据采集方案，不表示当前 runner 已能执行该任务。

## 1. 机械可行性先于感知

Hand-E 的配置最大开口为 50 mm，安装触觉垫后的真实可用开口还需重新测量。若杯口外径
超过可用开口，就不能跨杯口直径侧夹。即使杯底小于 50 mm，嵌套状态下从杯身侧面闭合也
容易同时压住多只杯。

首选抓法是局部杯沿夹持：夹爪从上方接近，闭合轴沿杯口的局部径向，一指进入杯内，一指
留在杯外，只夹最上层杯的卷边或外露领口。这里的 50 mm 只负责跨过局部杯壁，并不需要跨
整个杯口。

执行前必须满足：

```text
最上杯沿到第二杯沿的可见/可接触高度差
    > 触觉垫在杯轴方向上的有效接触高度 + 定位裕量
```

若不满足，大平面触觉垫会机械性地同时碰到两层杯沿，感知算法无法补救。此时需要窄指尖、
rim-hook 适配件或固定式分杯夹具。杯堆也应有底座限制其水平移动，同时允许最上杯拔出。

## 2. 必须先测的实体参数

| 类别 | 必测信息 | 用途 |
|---|---|---|
| 杯子 | 材质、透明度、单杯质量、高度、杯口内外径 | 判断视觉、开口和载荷可行性 |
| 杯沿 | 卷边截面、厚度、外露高度、相邻杯沿层距 | 判断能否只夹一层及接触 ROI |
| 嵌套 | 插入深度、堆叠数量、是否静电/真空粘连 | 选择分离动作和异常模型 |
| 分离力 | 多次慢速拔出的峰值、释放突降、方差 | 建立 extraction load envelope |
| 指端 | 安装传感器后的开口、垫面宽高、厚度、圆角 | 碰撞与单层接触检查 |
| 触觉 | 左右 serial、轴和符号、零点、温漂、法向/剪切响应 | 把 SDK 数据转为物理原语 |
| 视觉 | 顶部杯沿中心/轴检测率、相机标定、遮挡 | 决定实时杯沿 grounding 是否可用 |

透明 PP/PET 杯可能让 D435 深度因透射和反射失真。此时以 RGB 杯沿/轮廓为主，使用高对比
背景、背光或侧视相机；深度只作为带有效性 mask 的辅助信息。纸杯或不透明杯通常更适合
直接使用 RGB-D。

## 3. 信息优先级

### 3.1 外部 RGB(/D)：全局杯沿 grounding

需要估计：

- 最上层杯沿中心、平面法向和杯堆轴；
- 可进入杯内、同时避开杯堆和夹具的局部杯沿弧段；
- 当前杯沿 task frame 和可达接近方向。

前视相机不负责数杯，也不从第二杯沿或试抬后的轮廓数量决定是否继续。抬升前许可来自下述
内/外杯沿角色触觉门；多光学带仍只是 `MULTI_RIM_OR_TOO_DEEP` 假设，不能声称测得杯子数量。

### 3.2 触觉 RGB、diff、depth/XYZ：局部杯沿几何

- `WARPED_IMG/DIFF_IMG`：杯沿进入接触区、接触中心、接触线、贴边和多层边缘候选；
- `DEPTH_MAP/XYZ_VECTOR`：压入量、接触面积、薄壁屈曲和接触线几何；
- 两侧 contact patch：在外部视觉已经 ground 杯沿、TCP 位姿和左右安装映射已知的前提下，
  支持检查一指在杯内、一指在杯外，并且都落在预定 rim ROI。

`WARPED_IMG` 是校正到固定触觉垫坐标的凝胶 RGB 图，不是场景相机画面；`DIFF_IMG` 是它
相对无载基线的变化。二者能说明接触在垫面的哪里、面积和形状如何，不能单独识别“这是最
上层杯”。最上层身份由外部视觉 grounding 和 TCP/杯沿 task frame 提供，触觉只判断实际
局部接触是否符合该目标杯沿的预期模板。

确定性初版从每指 `DIFF_IMG` 的多帧中值提取 contact mask、面积、质心、PCA 切向、细长度、
ROI 占比、分离接触带数和质心抖动；可选 depth/XYZ 增加压入证据。左右传感器必须使用各自
的 ROI、内/外侧模板和图像到 task frame 的标定，不能直接镜像比较。一个卷边本身也可能
形成两条光学边，因此多连通域只能标成 `CONTACT_UNRESOLVED`，不能直接宣布夹到了两只杯。
只有 profile 提供经过实物标定的 task-frame“向较浅杯沿高度”修正时，分离的接触带才触发
“夹深了”的先验假设；也必须先微开卸载并确认脱离，再尝试该修正。没有明显斑纹/contact
patch 也只有在两指法向力都接近无载时才能解释为 `NO_CONTACT` 并继续微闭；若 force 已提示
接触，则属于图像/力不一致，必须卸力，不能继续夹。

当前 depth/XYZ 的尺度尚未验证，初版只作为 soft evidence。形变预算先使用已标定的
diff/contact 面积和饱和趋势；物理压入量在尺度验证前不能单独作为硬门。若要从 RGB 判断
单层/双层杯沿，必须先采带标签的数据，不能直接用未训练模型作执行门。

### 3.3 每指 force6d：轻夹持与分离载荷

原始两指 wrench 先变换到共同 TCP/task frame，并分别计算：

```text
normal_load_inner/outer    内侧指和外侧指各自的向内法向夹持力
role_load_ratio(optional)  仅在实测标定后启用的内/外侧载荷比容差
extraction_shear           每指沿拔取方向的切向负载
net_extraction_force       TCP 上沿杯轴的外部合力
tilt_torque                与斜拉、卡滞相关的横向力矩
task_axis_torque           仅在采用扭转分离策略时作为主量
```

内外侧接触几何不同，因此不能默认要求两指法向力相等。每个角色先分别满足自己的低力安全
带；只有实测表明确有稳定判别力时，才额外启用版本化的内/外载荷比容差。左右法向力方向
相反，不能直接相加作为夹紧力。`net_extraction_force` 可用于观察嵌套阻力
峰值及释放后的下降，但重量差只能作为辅助证据。力矩适合检测偏心、勾住第二只或倾覆；在
轴、符号和单位完成已知力臂标定前，只能使用归一化趋势，不能设物理硬阈值。

### 3.4 marker、slip 和机器人状态：动态耦合

- `MARKER_OFFSET_VECTOR`：切向形变、接触中心移动和旋转趋势；
- `SLIP_STATE`：初滑候选，必须经过时间窗，并和 marker/force/contact patch 交叉确认；
- Hand-E 开口、`gOBJ`、fault：独立夹爪证据；
- TCP 位姿、速度和命令位移：形成 force-versus-displacement 分离曲线。

## 4. 阶段合同

该任务需要把嵌套分离和自由空间抬升分开：

```text
OBSERVE
  → GROUND_TOP_RIM
  → APPROACH
  → RIM_TOUCH_ALIGN
  → GENTLE_PINCH
  → SEPARATE
  → VERIFY_ONE
  → CLEAR_LIFT
  → TRANSPORT / PLACE / RELEASE / RETREAT
```

| 阶段 | 主信息 | 完成条件 | 失败动作 |
|---|---|---|---|
| `OBSERVE/GROUND_TOP_RIM` | 外部 RGB-D、杯和指端几何 | 顶部杯沿中心/轴稳定、局部 rim frame 可达 | 不执行，换视角或更换指端 |
| `APPROACH` | TCP、视觉、触觉 health | 到达杯沿上方且无接触 | 提前接触则 cancel、撤离、重定位 |
| `RIM_TOUCH_ALIGN` | RGB/diff/depth/XYZ contact patch | 一指命中 INNER 模板，另一指命中 OUTER 模板；不比较左右图像是否对称 | 停止闭合、微开卸载、确认脱离，再有界重对中 |
| `GENTLE_PINCH` | 每角色法向力、patch、压入量、开口 | 连续 K 个新鲜窗口同时满足 `inner_rim_match` 与 `outer_rim_match`，且未超开口/力/步数预算 | 过载/屈曲时停止闭合并卸力 |
| `SEPARATE` | 剪切、marker/slip、轴向合力、倾斜力矩 | 完成有界微抬且内外角色接触保持 | 卡滞或触觉失配时停止、回落、松开重试 |
| `VERIFY_ONE` | 双指触觉、载荷和 Hand-E 状态 | 抬升前的内/外角色 gate 仍成立且持物稳定；该历史阶段名不再表示视觉计数 | 触觉失配则停止、保持或慢速回落 |
| `CLEAR_LIFT` | stable hold、marker/slip、开口 | 达到安全净空且持续持物 | 滑移时 hold 或慢降，不直接张开 |

`SEPARATE` 推荐先做几毫米慢速微抬。若直拔阻力不稳定，可在验证后加入很小的径向 peel 或
小角度倾转；不能用不断增加夹持力来克服嵌套摩擦，否则容易压瘪杯壁或同时带起多只。

当前 `TaskSkill v3` 没有 `SEPARATE/VERIFY_ONE`。快速 MVP 可以把它们做成任务 runner 中
`LIFT` 前的内部微阶段；正式通用版本应升级 skill schema，显式保留这两个阶段。

### 4.1 杯沿主动试探循环

主动试探不是逐帧力伺服，而是停稳后的离散闭环：

```text
低速、低力微闭
  → 等命令停止和凝胶稳定
  → 采 4--6 个 SDK timestamp 各不相同的新帧
  → evaluate_rim_probe(...)
      ├─ NO_CONTACT                 无 patch 且 force 接近无载，再微闭一步
      ├─ ROLE_MISMATCH              缺角色或角色模板错配；卸载后微调 TCP/偏航
      ├─ MULTI_RIM_OR_TOO_DEEP      超出该角色模板的多带；卸载后移向较浅高度假设
      ├─ ROLE_MATCH_LOW_LOAD           再微闭一步
      ├─ CAPTURE_CANDIDATE          保持并采下一个新窗口
      ├─ INNER_OUTER_RIM_MATCH  连续 K 个候选后 ACCEPT，进入 SEPARATE
      └─ OVERLOAD / INVALID / 模糊  卸力、保持或终止本轮
```

纯判断与稳定门实现在
[`src/demo2gripper/tactile_rim_probe.py`](../src/demo2gripper/tactile_rim_probe.py)。
`evaluate_rim_probe()` 输入已经同步、去重的左右 `DIFF_IMG`、每指 `FORCE6D` 和可选 depth
窗口，输出接触特征、predicate、置信度以及 `CLOSE_STEP`、`WAIT`、`RELEASE`、
`RELEASE_THEN_ADJUST` 或 `ACCEPT` 请求。它不持有硬件句柄，也不直接移动机械臂。
runner 把上一条夹爪命令完成时的 host monotonic time 作为 `not_before_s`，从而排除命令前
的旧帧；窗口内任一 force 峰值过载或载荷未稳定，都会优先请求卸力。

每个试探回合必须创建非空 `probe_context_id`，它绑定当前 probe episode、固定 TCP 位姿和
profile 以及当前 left/right 到 INNER/OUTER 的角色映射。微闭、微开、proposal 与 release
confirmation 在 TCP 保持不动时沿用同一 id；一旦
TCP 移动、重新 grounding、开始新回合、profile 或任一 calibration id 改变，就必须换新 id
并重置 stability gate。缺失该 id 只允许 shadow evaluation，不能累计 `CAPTURED`，也不能
授权 TCP 修正。

调整只允许在微开卸力并确认脱离之后执行。质心误差通过左右各自标定的
`image_error_to_task_m_per_px` 转为限幅 task-frame 平移；切向误差通过单独标定的符号和
增益转为限幅偏航。返回量只是 proposed correction；runner 必须在后续新鲜窗口中确认接触
和载荷都已释放，才能执行 TCP 移动。若映射尚未标定，函数只请求 `RELEASE`，不会猜测
TCP 修正方向。
`authorize_tcp_adjustment(proposal, release_evaluation, release_receipt, config, now_s=...)`
实现这个第二道门。`release_receipt` 必须证明 proposal 要求的正向开口已完成；release 窗口
必须以该完成时间作为 `not_before_s`，状态为 `NO_CONTACT`，并且所有必需通道的 SDK 与 host
timestamp 都严格前进、host 采样时间都晚于 release 完成时刻。proposal、release、receipt
还必须具有相同的 `probe_context_id`，前两者必须匹配当前 profile、左右 calibration id 和
冻结的 INNER/OUTER 角色映射。
默认授权 TTL 为 1 s，配置硬上限为 5 s；从 proposal 评估时刻起过期后返回 `None`。

成功时函数返回短时效 `TcpAdjustmentAuthorization`，而不是可重复使用的动作。runner 必须用
`TcpAdjustmentAuthorizationLedger.consume(...)` 在当前 context/profile/calibration/role mapping
下消费一次，
并在真正移动 TCP 前将已消费 `authorization_id` 原子写入持久 stage journal，防止进程重启
后重放同一物理命令。
Hand-E 是单自由度对称夹爪，只能统一微开/微闭；“每指独立”指左右法向力分别测量和判门，
不是分别命令两根手指。

当前实测图像约 24 Hz、force6d 约 19 Hz；150--250 ms 指每次命令停稳后的凝胶稳定与采集
周期，不是允许证据保持这么久。evaluator 的默认 `max_age_s` 是 0.15 s，评估时使用的每个
样本仍必须满足该 150 ms 帧龄上限。同一个 SDK timestamp 重复出现不能增加连续通过数；
`STALE/DESYNCED/INVALID` 也不能解释成 `NO_CONTACT`。同一 TCP 位姿只反复开合不会修复
单边或低位接触，必须卸载后按接触误差调整。

## 5. 触觉 predicate

```text
rim_capture_candidate =
    sensor_healthy
    AND inner_rim_match(inner_role_finger)
    AND outer_rim_match(outer_role_finger)
    AND normal_load_in_fragile_band(inner_role_finger)
    AND normal_load_in_fragile_band(outer_role_finger)
    AND cup_deformation_budget_ok
    AND NOT slip_onset
    AND NOT overload

rim_capture_verified =
    K_consecutive_fresh_windows(rim_capture_candidate)

bounded_separation_complete =
    rim_capture_verified
    AND inner_rim_match
    AND outer_rim_match
    AND tactile_lift_permission
    AND bounded_micro_lift_completed

prelift_role_match_gate_preserved =
    inner_rim_match
    AND outer_rim_match
    AND stable_hold

cup_remains_coupled =
    inner_rim_contact
    AND outer_rim_contact
    AND stable_hold
    AND NOT slip_onset
    AND NOT overload
```

前视 RGB-D 只负责定位杯沿中心/轴，不负责杯子数量判定。当前 gate 只有在一个物理手指命中
其 INNER 模板、另一个命中其 OUTER 模板，并连续满足各自载荷与形变安全门时才允许抬升。
两块触觉图像的面积、形状和位置可以不同；代码不要求左右图像对称。多带只能标记为
`MULTI_RIM_OR_TOO_DEEP` 假设并触发上移重试，不能写成视觉测得的杯子数量。

## 6. 最小采集与验证顺序

1. 修复 GF220 的 640×480 profile、逐只 warp、有效帧后校准和 SDK timestamp 去重。
2. 测量杯沿层距与触觉垫有效高度，先通过机械单层可达性门。
3. 使用固定杯堆，在 shadow mode 同步记录外部 RGB(/D)、两路触觉全部候选通道、Hand-E
   状态和 TCP；不让触觉改变动作。
4. 至少覆盖：正确单杯、带起两杯、接触过低、单边接触、杯体屈曲、卡滞和滑移。
5. 标定 rim ROI、低法向力带、形变预算、提取力-位移包络和 marker/slip 窗口。
6. 施加已知的高/低、径向、切向和旋转偏移，分别标定左右图像误差到 rim-task frame 的
   局部 Jacobian；没有该标定时只运行判定，不自动修正 TCP。
7. shadow replay 判定器和策略，覆盖重复 timestamp、不同步、单边、越 ROI、过载、同一
   卷边双光学边和真正双层接触，再启用阶段边界判断。
8. 补充可取消运动和停止确认后，再实现连续 `SEPARATE` guard。
9. 多次重复后统计单杯成功率、多杯误取率、杯体损坏率和 UNKNOWN 率，再将 profile 从
   `review_only` 升级。

当前两只 GF220 尚未完成上述设备资格确认，因此全部触觉 evaluator、probe policy 和先验
状态转移都只能做数据采集与 shadow evaluation。任何 `CLOSE_STEP`、`RELEASE`、
`RELEASE_THEN_ADJUST`、`APPLY_TCP_ADJUSTMENT` 或 `ACCEPT` 都只是记录建议，不能发送夹爪/
机械臂命令，也不能停止、完成或推进真机阶段。
