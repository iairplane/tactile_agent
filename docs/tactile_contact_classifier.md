# 双指杯沿触觉五分类器

[`tactile_contact_classifier.py`](../src/demo2gripper/tactile_contact_classifier.py)
只判断两块触觉垫上的局部接触几何，不使用前视相机，也不通过试抬判断结果。输入是左右两指
同步的新鲜 `WARPED_IMG`、`DIFF_IMG` 时间窗，以及可选的、已经标定轴和单位的
`FORCE6D`。输出恰好是以下五类之一：

| 类别 | 判定含义 |
|---|---|
| `NO_CONTACT` | 两指都没有持续 contact patch；配置了力时，两指也都处于无载区间 |
| `INNER_OUTER_RIM_MATCH` | 内侧指和外侧指分别匹配自己的 role-specific 模板；不要求左右接触看起来相同 |
| `MULTI_RIM` | 至少一指的分离带数量达到该角色实测的 `multi_rim_min_band_count` |
| `ROLE_MISMATCH` | 少一个角色的接触，或任一指的面积、位置、方向、形状、稳定性、带数不符合其角色模板 |
| `UNKNOWN` | 过期/重复/不同步、瞬态、`WARPED_IMG` 与 `DIFF_IMG` 冲突、图像与力冲突或过载 |

判断顺序为：输入有效性与过载 → 图像/力一致性 → `NO_CONTACT` → `MULTI_RIM` →
`ROLE_MISMATCH` → `INNER_OUTER_RIM_MATCH`。因此不会把“力已经接触但图像没斑纹”解释成
没夹到，也不会因为正确的内外侧接触面积、位置或法向力不同而误报不对称。

`MULTI_RIM` 是**超过角色模板的多条触觉带**，不是“两只杯”的真值。外侧指接触卷边时，
单个杯沿本身就可能稳定产生两条光学边，因此每个角色各自配置 `expected_band_count` 和
`multi_rim_min_band_count`。例如外侧模板可以接受 2 条，只在 3 条及以上时输出 `MULTI_RIM`。
这些数量和 `min_distinct_band_separation_px` 必须由标注样本确定。在完成标定以前，
`MULTI_RIM` 只能作为 shadow 建议，不能证明夹了两只杯。

物理 left/right 不等于永久的 INNER/OUTER。当前 grasp/probe hypothesis 必须显式选择 left 和
right 各自使用哪个角色模板，并保证恰好一个 `INNER`、一个 `OUTER`；该映射在一次 probe
窗口内冻结。夹爪绕杯轴反向、closing axis 翻转或重新规划后，可以交换这两个角色。传感器
baseline/warp 仍按 serial 标定，任务模板按“serial + 当前角色”绑定，`profile_id` 应记录这次
绑定，防止把旧窗口按另一种角色解释。分类结果的 `evidence_key` 同时带上 `profile_id`、左右
calibration id 和角色映射，再附通道时间戳，因此交换角色后的同一批帧不会与旧证据同 key。

## 实时输入合同

- 每个窗口至少包含 4 个不同 SDK timestamp 的图像样本；可选力窗口至少 3 个不同样本。
- `WARPED_IMG` 与 `DIFF_IMG` 必须在同一个标定版本和固定 pad 坐标中；左右两指各自使用
  独立 baseline、ROI 和方向，不能直接镜像复用。
- pyvitaisdk 1.0.11 实测 `WARPED_IMG`/`DIFF_IMG` 都是 240×240×3 `uint8`，
  `FORCE6D_VECTOR` 是长度 6 的 `float64`，`TIME_STAMP` 是标量 `int64`。一次
  `collect_sensor_data` 可以用同一个 timestamp 同帧取得这些通道。
- 该 SDK 的无变化 `DIFF_IMG` 是全 127，不是全 0。分类器会先转浮点并减去 profile 中的
  `diff_neutral_value`，再求幅值；不能直接对 `uint8` 做减法或绝对值。
- 默认只接受 150 ms 内的样本，所有必需通道的末端 host monotonic time 相差不超过 80 ms。
- baseline 必须在传感器无载、曝光稳定后采集；更换传感器、指套、曝光、warp 或重新安装后
  必须更新 `calibration_id`。
- 如果 profile 配置 `FORCE6D`，左右两指都必须提供新鲜力样本；否则 classifier 明确按
  image-only 模式运行。

分类器先用多帧 `DIFF_IMG` 中值提取 contact mask，再用 `WARPED_IMG` 相对无载 baseline 的
变化检查该 mask 是否有实际凝胶图像支持。每指随后计算面积、ROI 占比、质心、质心抖动、
PCA 切向、细长度和分离带数量，再只与该指当前绑定的 INNER 或 OUTER 模板比较。可选
`FORCE6D` 只作每指接触一致性、载荷稳定和过载否决，不替代图像几何，也不默认要求左右
法向力相等。

## 必须用实测样本确定的阈值

代码没有可直接用于真机的通用数值。以下每个 profile 字段都必须按左右传感器分别标定：

- `unloaded_warped`、`diff_neutral_value`、`diff_threshold`、`warped_delta_threshold`：无载噪声、
  DIFF 中心值、曝光漂移和温漂；pyvitaisdk 1.0.11 的实测中心值为 127，仍应绑定 SDK 和
  calibration 版本保存；
- INNER 和 OUTER 各自的 `rim_roi_mask`、`target_center_uv_px`、`target_tangent_uv`：正确角色接触
  在 pad 坐标中的模板，不能用左右镜像或共同模板代替；
- 每个角色各自的 `min/max_contact_area_px`、`min_band_area_px`、`min_band_elongation`：正确杯沿、
  杯壁整片接触和局部噪声的分布；
- `min_contact_frame_fraction`、`max_center_jitter_px`：停稳接触和擦过/滑动接触的时间分布；
- `min_warped_support_fraction`：真实凝胶变化与 SDK diff 的一致性；
- `min_roi_fraction`、`max_center_error_px`、`max_tangent_error_deg`：可接受的高低、径向和偏航误差；
- `expected_band_count`、`multi_rim_min_band_count`、`min_distinct_band_separation_px`：每个角色下
  单个卷边的正常光学边数，以及超过正常模板后进入多带候选的边界；
- 可选力参数 `inward_normal_sensor`、`no_contact_max_n`、`contact_min_n`、`overload_n`、
  `max_load_range_n`：按每个传感器/角色用已知方向和已知载荷标定，不能由 SDK 默认轴猜测。

代码已移除默认的左右面积平衡和法向力平衡硬门。若实测显示内外侧面积比或力范围对任务有
稳定判别力，应把它作为显式、带版本的 role template 参数加入；不能重新引入“默认比值为 1”
的假设。

建议按独立物理试次采集并标注：无载、正确单杯沿、同一卷边双光学边、真正双层杯沿、左/右
单边、过高/过低/偏航、整片杯壁、屈曲和过载。阈值在训练试次上选择，再按杯子和试次整体
划分 held-out 集，报告五类混淆矩阵，重点报告 `MULTI_RIM` 假阳性、
`INNER_OUTER_RIM_MATCH` 假阳性、`ROLE_MISMATCH` 和 `UNKNOWN` 率。

## 当前数据边界

现有 `cup_top_demo_20261023_01`、`cup_top_retry_aggressive_20261023_02/03` 保存的是
640×480 raw UVC GF220 图像。manifest 明确说明它们不是 SDK `WARPED_IMG/DIFF_IMG`；A/B 尚未
映射为物理左右指，也没有 `FORCE6D`。第一轮验证还显示两块 pad 全程无接触，变化主要来自
曝光/照明。因此这些数据只能检查采集和无载稳定性，不能估计上述五类阈值。

GF220 原始画面实测是 640×480，而 SDK 输出的手动 warp 是 240×240。GF225 的默认 auto-warp
不适用于这批 GF220；每个 serial 都必须保存自己的四角和 calibration id。

当前单元测试使用互不相同的 INNER/OUTER 位置、面积和正常带数，覆盖角色交换、合法的不等
面积/不等力、超角色带数、单边接触、错误角色、通道冲突、力冲突和过期窗口。
在采到带标签的 SDK 数据并完成 profile 标定以前，输出只能作为 shadow 日志，不能驱动
夹爪、机械臂或阶段转换。
