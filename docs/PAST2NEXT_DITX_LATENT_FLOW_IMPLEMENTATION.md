# Past2Next：冻结 OAT + DINOv3-S/16 + DiT-X Latent Flow 实现计划

文档日期：2026-09-24

目标仓库：`/workspace/ysk/past2next_bug_fixed`

状态：**待实现的技术规范。此次任务只新增本文，不修改代码、下载权重或启动训练。**

关联文档：[DINOv3-S/16 自回归版本实现计划](PAST2NEXT_DINOV3_SMALL_IMPLEMENTATION.md)。本文是独立的 latent-flow 路线，原 AR 文档与已有实现继续保留。

## 1. 已确定的目标与边界

用户明确要求本分支也保留指定的**冻结 OAT 编码器、FSQ 和解码器**。因此采用 ManiFlow-inspired latent flow：在 OAT 的量化标量码空间训练连续向量场，在采样终点投影回合法 FSQ 网格，再用冻结 OAT 解码为机器人动作。

必须交付两个独立策略：

| 用户所需变体 | 新策略标识 | 条件与模块 |
|---|---|---|
| p2n | `p2n_latent_flow` | 视觉、当前状态、过去动作和命令差分；无额外状态历史编码器 |
| p2n_state_gate | `p2n_state_gate_latent_flow` | 公共条件，再增加状态历史摘要与 summary gate |

两个策略共用架构和冻结权重来源，分别训练可训练参数、optimizer 与 EMA，保存独立 artifact。基础版本不能由关闭 gate 的方式代替。

本方案保留 ManiFlow 的时间/步长条件、DiT-X 调制结构、flow matching 与 consistency training 思路；DINOv3、OAT 潜变量、SwiGLU、双侧 QK-Norm、Past2Next history 和 summary gate 是本仓库的扩展，不是原论文已验证的同一套模型。原始 ManiFlow 直接生成连续动作，论文的少步推理性能不能直接作为本方案的结果。[S1][S2]

本次不加入架构消融矩阵、语言模型、点云分支、MoE、分类器引导或 tokenizer 微调。第一版不加入通过冻结解码器反传的辅助动作损失；目标是把已确定的 latent flow 完整实现并验证。

## 2. 固定配方与张量总览

| 项目 | 固定设计 |
|---|---|
| 视觉主干 | 冻结 `facebook/dinov3-vits16-pretrain-lvd1689m` |
| 输入图像 | 两相机 × 两帧；原始 128×128 RGB 重采样到 224×224 |
| Patch features | 每张图 196×384，排除 CLS/register |
| 视觉适配器 | 384→768 投影；2 层 Resampler，64 queries/图，12 heads |
| 公共条件 | 256 visual + 2 proprio + 7 past-action + 2 command-diff = 267 tokens |
| Gate 条件 | 增加 4 个 state-history summaries，共 271 tokens |
| Flow 目标 | OAT 量化后的 `[B, 8, 5]` 标量码，40 个连续坐标 |
| Flow 网络 | 16 层，768 维，12 heads，head dimension 64 |
| FFN | SwiGLU，中间维度 2048 |
| 动作潜变量 attention | 双向 self-attention，learned OAT slot embedding |
| 条件 attention | Cross-attention，使用 padding 与 summary additive bias |
| 调制 | AdaLN-Zero，self/cross/FFN 各有 shift、scale、residual gate |
| QK-Norm | Self/cross attention 均按 head feature 维执行 RMSNorm |
| 时间输入 | 当前时间 t、相对步长 dt，分别编码 |
| 动作输出 | OAT 解码 `[B,16,7]`，默认执行前 8 步 |
| 初始验收采样 | 8 步 Euler；两步作为后续部署验收目标 |
| 训练精度 | 网络 BF16 autocast；flow 算术、损失、FSQ 投影保留 FP32 |
| 初始双卡配方 | 每卡 microbatch 4，2 ranks，累积 8 次，有效 batch 64 |

16 层和 768 维为容量选择，不代表在 77 条示范上已证明最优。所有参数量、显存和速度估计必须与最终实例化/实测结果区分。

## 3. 当前仓库可复用内容

文档编写时，仓库已存在新版 DINO/AR 基础设施。以下是只读检查到的模块，不表示本次任务创建或验证了这些实现：

| 现有文件 | 本方案复用方式 |
|---|---|
| [DINO patch encoder](../oat/perception/dinov3_patch_encoder.py) | 预处理、冻结、patch 提取与离线结构恢复 |
| [Visual Resampler](../oat/perception/visual_resampler.py) | 可训练空间 token 适配器 |
| [Token observation encoder](../oat/perception/token_obs_encoder.py) | 相机/帧/状态编码；增加可接收预计算 frozen patches 的接口 |
| [ContextBatch](../oat/model/common/context_batch.py) | memory、segment、有效性和 summary bias |
| [新版策略公共实现](../oat/policy/p2n_new_common.py) | normalizer、history、self-past 课程、执行确认；提取与 AR 无关的部分 |
| [新版 gate 策略](../oat/policy/p2n_state_gate_new.py) | 状态历史对齐、rot6d 几何、summary gate |
| [新版训练 workspace](../oat/workspace/train_p2n_new.py) | BF16、DDP、累积、恢复、更新计数、显存日志的实现基础 |
| [能力路由](../oat/common/p2n_new_capabilities.py) | 显式 history/validity 参数与执行协议路由 |
| [OAT tokenizer](../oat/tokenizer/oat/tokenizer.py) | 冻结 encode/decode 及原始动作 normalizer |
| [FSQ](../oat/tokenizer/oat/quantizer/fsq.py) | 合法 scalar-code 网格与 ID 的一一映射 |

不能直接调用现有 AR `forward()` 或 `_generate_actions()` 完成 flow：这些接口包含 BOS、分类 logits、top-k 和 token generation。新策略不应先创建整套 AR 模块再遗留不用的参数。

旧 `transformer_for_diffusion.py` 也不是已经实现的 ManiFlow。只复用适用的通用算子与基础设施，不把 DDPM/DDIM 的 epsilon loss 或 scheduler 当作本方案的 flow 目标。

当前 `ContextBatch.validate_variant()` 识别的是 AR variant 字符串；未来需提取独立的条件布局类型 `plain` / `state_gate`，让 AR 与 flow 的明确 variant 映射到布局，保留旧调用兼容。不要把 flow checkpoint 假标为 AR 变体以绕过校验。

## 4. 已选冻结 tokenizer 与 latent adapter

### 4.1 固定来源

```text
/workspace/ysk/past2next_bug_fixed/output/training/nut_washer_v3_N77_gated_so3aug_20260924_081008_317043427/tokenizer/checkpoints/ep-1540_mse-0.000.ckpt
```

此前已在 CPU 严格加载该 checkpoint 的 EMA tokenizer；本次文档任务不重复占用 GPU 验证。

| 核验项目 | 值 |
|---|---|
| Tokenizer 类型 | `OATTokSO3Aug` |
| 采用权重 | checkpoint 内 EMA 权重，保持冻结 |
| 动作维度 / horizon | 7 / 16 |
| Latent horizon / 每 slot 维度 | 8 / 5 |
| FSQ levels | `[8,5,5,5,5]` |
| 离散 codebook | 5000 个 codes，无 BOS 生成问题 |
| Codec 内部宽度 | 256；不是 latent 维度 |
| Parameter 元素数 | 5,804,854，包含 normalizer 参数 |
| FP32 tensor payload | 约 22.24 MiB，包含 buffers；不等于 checkpoint 文件大小 |

训练目标直接调用 `OAT.encode(raw_actions)` 获取量化码；该函数已经执行 action normalization、encoder 和 FSQ。不能在其之前再次归一化动作。`OAT.decode(z_grid)` 已包含动作反归一化。

OAT 的 SO(3) augmentation 属于 tokenizer 训练 forward，冻结 encode/decode 不执行该增强。真机动作仍为 7 维，旋转通道是 3 维 rotation-vector delta；观测 rot6d 为独立的状态表示，不能替换动作中的旋转维度。

### 4.2 Latent adapter 的职责

拟新增 `FrozenOATLatentAdapter`，封装已有 tokenizer，不引入可训练参数：

```python
# 拟实现 API，不是当前已有接口。
encode_actions(raw_actions) -> LatentTargets
# LatentTargets.codes: float32 [B, 8, 5]
# LatentTargets.indices: int64 [B, 8]，用于诊断/round trip

snap_codes(continuous_codes) -> float32 [B, 8, 5]
decode_grid_codes(grid_codes) -> actions [B, 16, 7]
```

- encode 和普通解码在 `no_grad()` 下运行，tokenizer 始终 eval、`requires_grad=False`。
- `codes` 使用 FSQ 原始归一化坐标，不另拟合 latent mean/std，不学习 token-ID embedding 来替代码空间。
- 若内部使用 `inference_mode()`，离开后 clone 成普通张量再送入需要反传保存输入的层；训练目标准备优先直接使用 `no_grad()`。
- 通过 checkpoint 配置核验 latent horizon、levels、action horizon/dim，而不是仅信任文件名。
- `mse-0.000` 是文件名中的舍入显示，不是零重建误差证明。

### 4.3 FSQ 终点投影

令 `L=[8,5,5,5,5]`，`h=L//2=[4,2,2,2,2]`。合法网格为：

- 第 1 维：`{-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75}`。
- 第 2–5 维：`{-1, -0.5, 0, 0.5, 1}`。

仅对最终采样结果执行以下 FP32 投影；先检查所有值 finite：

```python
# 伪代码：levels/half_width 跟随输入 device。
q = torch.round(z_hat.float() * half_width + half_width)
q = torch.minimum(torch.maximum(q, torch.zeros_like(q)), levels - 1)
z_grid = (q - half_width) / half_width
```

投影后的 `z_grid` 直接送 `OAT.decode()`，也可通过 `codes_to_indices(z_grid)` 再 `detokenize()`，两者应数值一致。当前逆映射 API 名为 `indices_to_embedding()`。

**禁止对 z_grid 再调用完整 quantizer。** 其 `bound/tanh` 属于从未约束 encoder 输出到网格的过程，再执行会改变合法码。也不能把连续 z_hat 直接送 `codes_to_indices()`，因为该函数不负责最近网格投影和合法性检查。

ODE 中间点、FM 目标和 EMA consistency 终点估计均不 snap、不逐步 clamp。只在最终采样输出量化；NaN/Inf 直接报错，不能用 clamp 掩盖。

## 5. 两变体的条件布局与门控

| Segment | p2n_latent_flow | p2n_state_gate_latent_flow |
|---|---:|---:|
| VISUAL | 256 | 256 |
| PROPRIO | 2 | 2 |
| RAW_ACTION | 7 | 7 |
| ACTION_DIFF | 2 | 2 |
| HISTORY_SUMMARY | 0 | 4 |
| 合计 | **267** | **271** |

保留相机身份、观测帧位置、Resampler query 身份、条件类型与历史相对时间。任务标识进入本体状态特征，不另外增加未声明 token。

基础变体不创建 history encoder、summary gate 或其专用 observation pooling；ContextBatch 的 summary/gate 字段均为 None。Gate 变体使用 8 个实测状态与 7 条过去命令、小型 128 维历史编码器、4 个 summaries；真机使用 rot6d rows 几何适配。

`valid_mask=True` 表示可读取。所有 cross-attention 都使用 ContextBatch 的 additive padding bias；summary gate 的 `logsigmoid(logit)` 仅加在 HISTORY_SUMMARY 位置。gate hidden dimension 128、初始概率 0.9，正式训练使用 learned；open/closed 仅用于接口正确性检查。

**两类 gate 不混用：**

1. DiT-X AdaLN-Zero residual gate 控制整层子路径的残差贡献，初始化为 0。
2. Past2Next summary gate 控制额外状态历史摘要的可见性，初始为 0.9。

AdaLN 全局输入只包含时间、步长与当前本体状态/任务信息，不读取 history-summary、history pooled vector 或 state-history valid fraction。视觉与本体状态编码也不读取额外摘要。这样 closed summary gate 才能保证输出不受额外摘要扰动影响；raw-action/diff 条件仍可见。

Gate memory 在每个 chunk 内固定，不随 flow 时间变化。不能照搬官方 DiT-X 中 `mask=None` 的 cross-attention 调用；本仓库的 validity 和 summary bias 必须进入实际 SDPA。

## 6. DiT-X Latent Flow 网络

### 6.1 输入、位置与全局条件

网络签名建议为：

```python
velocity = flow_model(z_t, time=t, step_size=dt,
                      context=context, current_state=state_features)
# z_t / velocity: [B, 8, 5]
# t / dt: float32 [B]，子批 B=1 时不得 squeeze 掉 batch 轴
```

输入线性投影 5→768，与 8 个 learned slot embeddings 相加。8 个 slots 沿用 OAT 的固定身份，不宣称每个 slot 对应两个物理动作时刻。

`t` 与 `dt` 各使用 256 维 sinusoidal embedding，再经 `256→1024→768` MLP 编码。两帧当前状态/任务向量按固定 schema 拼接，由独立 MLP 映射至 768。拼接三个 768 维向量后投影至 768，得到全局条件 `c_global`。

当前状态编码使用训练集 state normalizer；不把未来动作目标、FSQ 目标码或未来有效性 metadata 放入条件。

### 6.2 Block 结构

每个 block 包含双向 self-attention、条件 cross-attention、SwiGLU。三个分支各自使用无 affine 的 LayerNorm，经 AdaLN 的 shift/scale 调制，输出经 residual gate 后加入残差：

```text
shift_sa, scale_sa, gate_sa,
shift_ca, scale_ca, gate_ca,
shift_ff, scale_ff, gate_ff = Linear(SiLU(c_global))  # 9 × 768

u = LN_sa(x) * (1 + scale_sa) + shift_sa
x = x + gate_sa * SelfAttention(u, causal=False)

u = LN_ca(x) * (1 + scale_ca) + shift_ca
x = x + gate_ca * CrossAttention(u, context.memory, context.bias)

u = LN_ff(x) * (1 + scale_ff) + shift_ff
x = x + gate_ff * SwiGLU(u)
```

- 每层独立的 768→6912 调制投影；不共享以隐藏容量变化。
- Self/cross QK RMSNorm 在 reshape 成 heads 后沿最后 64 维计算；FP32 累加，保留 `1/sqrt(64)` scaling。
- SwiGLU 使用三个 bias-free 投影，`down(silu(gate(u))*up(u))`，中间维度 2048。
- Self-attention 不使用 causal mask。8 个 noisy slots 均为待联合生成的变量，不是未来真实动作泄漏。
- 无 BOS、5001 类输出、teacher-forcing shift、top-k 或 AR prefix。
- 不新增动作侧 RoPE；视觉主干原有位置处理不变。
- 条件 memory 不施加依赖 t/dt 的 AdaptiveLN；调制只作用在 action-query 路径及 residual 输出，允许 chunk 内静态 cross-KV。

### 6.3 输出与初始化

最终 `RMSNorm → Linear(768,768) → GELU → Linear(768,5)` 输出速度。最后线性层与所有 AdaLN modulation 最后一层零初始化；其余新投影采用记录在配置中的 Xavier/normal 初始化，预训练 DINO/OAT 不参与重新初始化。

零初始化会使部分模块在最初一次 backward 的梯度为零。验收检查梯度有限性，并检查经过若干 optimizer updates 后有效梯度和损失变化；不要求第一步每个参数梯度都非零。

Flow 专用初始配方将 DiT-X、Resampler 和 history encoder 等可训练模块 dropout 明确设为 **0.0**，用于减少 student/teacher 一致性目标的独立随机性。轻量光度增强与 weight decay 仍启用。此值是本方案的工程选择，不是静默继承 AR 配方的 0.1，也不是原论文最优值声明。

## 7. Flow matching 与 consistency 目标

### 7.1 FM 目标

冻结 OAT 对同一条动作样本给出 `z1: [B,8,5]`。采样标准高斯 `epsilon`，构造：

```text
z_t = (1 - t) * epsilon + t * z1
v_target_FM = z1 - epsilon
L_FM = mean((v_theta(z_t, t, 0, context) - v_target_FM)^2)
```

MSE 对 8 slots × 5 维及 FM 子批求均值，所有坐标默认同权。Gaussian noise 与码空间均为当前 FSQ 归一化坐标，不对整数 token ID 做回归。

### 7.2 EMA consistency 目标

对 CT 子批，以同一 `epsilon` 和 `z1` 构造两个插值点：

```text
t_next = min(t + dt, 1)
z_t    = (1 - t)      * epsilon + t      * z1
z_next = (1 - t_next) * epsilon + t_next * z1

# no_grad，EMA teacher 使用自己的可训练 condition encoder 参数
v_next = v_ema(z_next, t_next, dt, teacher_context)
z1_est = z_next + (1 - t_next) * v_next
v_target_CT = stop_gradient((z1_est - z_t) / (1 - t))

L_CT = mean((v_theta(z_t, t, dt, student_context) - v_target_CT)^2)
L_total = L_FM + L_CT
```

`z_next` 来自真实码与同一 Gaussian 的插值，不用 student 先走一步替代。Teacher 终点估计与 CT 目标始终连续，不 snap；student/teacher 使用同一观测增强结果、相同混合后的历史命令和 validity。

若 `t_next=1`，目标终点直接为 `z1`；可对这些行跳过 teacher 推理，避免无效的 `0 * nonfinite`。其他行按上式。不能把 CT 起始 t 设为 1。

### 7.3 固定采样与批划分

以下时间配方沿用原作者公开实现，[S3][S4]；本方案保留其相对 dt 约定：

| 项目 | 值 |
|---|---|
| FM t | `0.999 * Beta(1.0, 1.5)`，缩放而非 rejection sampling |
| FM dt | 0 |
| CT t | `randint(0,10)/10`，范围 `{0,0.1,...,0.9}` |
| CT dt | `Uniform[0,1)` |
| Teacher t | `min(t+dt,1)` |
| Teacher dt | 复用同一原始 dt，即使 t_next 被截断也不另改步长条件 |
| 批划分 | 每个训练 microbatch 随机分配 75% FM、25% CT |
| 损失权重 | 两个子批分别 mean 后相加，各权重 1 |

3:1 是样本分配，不是将 loss 再乘 0.75 和 0.25。训练要求每卡 microbatch 是 4 的倍数且至少 4，使用 drop_last；B=4 时 FM=3、CT=1。启动前拒绝不符合此默认合同的 batch，而不是 int 截断后漏掉样本。

验证允许任意尾批：默认对全部验证样本计算固定随机种子的 FM loss 和解码动作指标，不对尾批强行做 3:1 切分。若另外报告 CT loss，用全部样本独立构造 CT 诊断目标，并与训练混合 loss 分开命名。

时间、noise、插值、teacher 终点、除法、Euler 累积、loss reduction 均使用 FP32；网络前向可以 BF16。默认 CT 起点使 `1-t >= 0.1`，仍需验证输入范围和 finite。半精度张量不能先做不稳定运算后再 cast 为 FP32。

## 8. EMA、DDP 与训练调用设计

### 8.1 EMA 是训练依赖

Consistency teacher 使用 EMA 的完整可训练策略：视觉投影、Resampler、状态投影、history/gate 与 DiT-X。冻结 DINO/OAT 与 student 相同。Teacher 必须 eval、无梯度；本训练配方要求 `use_ema=true`，不能把 EMA 当作可关闭的评估选项。

复用现有 EMA schedule 的参数：`update_after_step=0, inv_gamma=1, power=0.75, min_value=0, max_value=0.9999`。只在成功 optimizer update 后执行一次 EMA；梯度累积 microbatch、验证、被跳过的 optimizer update 都不推进 EMA 或 self-past 计数。原 ManiFlow workspace 的 microbatch 更新位置不能原样复制到本仓库的累积训练。[S5]

### 8.2 Fresh 与 resume 的初始化顺序

1. 设置种子，构造 student，准备 dataloader 与 optimizer。
2. 让 DDP/Accelerate 同步 student；fresh training 的 EMA 必须从**同步后的 unwrapped student** 初始化，避免不同 rank 的初始化差异留在 teacher 中。
3. EMA 使用普通 device-local module，`requires_grad_(False).eval()`；不为 teacher 增加 DDP wrapper。
4. Resume 时分别恢复 student 和保存的 EMA；不能再次用 student 覆盖已恢复的 EMA。
5. 恢复 optimizer、scheduler、EMA schedule、self-past 与所有 RNG 状态后开始下一 update。

第一批 CT 前验证各 rank 的 teacher 一致。仅同步 student 而把同步前的 deepcopy 当作 teacher 不满足此要求。

### 8.3 单次 DDP student forward

建议工作区准备一个无梯度的 `FlowTrainingBatch`，再执行一次 DDP-wrapped student forward。拟采用以下边界：

```python
# 伪代码：所有接口均为待实现。
prepared = prepare_flow_training_batch(
    batch, student=unwrapped_policy, teacher=ema_policy,
    generator=train_generator,
)  # no_grad：编码目标、self-past、时间/noise、teacher targets、frozen patches

with accelerator.accumulate(student_ddp):
    loss = student_ddp(prepared)  # 所有 student 可训练条件模块都在此 forward 内执行
    accelerator.backward(loss)
    # 同步/裁剪/step 的实际调用遵循 Accelerate 的累积与 skipped-step 语义
    # 仅 successful_optimizer_update 时推进 scheduler、EMA、课程及计数
```

`FlowTrainingBatch` 至少包含 noisy latents、t、dt、detached velocity targets、FM/CT 行索引、混合后的过去命令及其有效性、当前状态、gate 所需实测历史和共享的 frozen DINO patches。

- Teacher 不注册为 student 的子模块，不通过成员字段造成循环 state_dict、重复 optimizer 参数或双倍序列化。
- 不在 `unwrapped_student` 外部运行需要训练的 Resampler/context builder 后再绕过 DDP 返回 loss。
- FM/CT 行合并成一个网络调用；同一张训练图内计算两个子集 loss，避免两次独立 DDP forward 的状态问题。
- 不把 teacher 网络放入传给 DDP 的 batch。准备函数只返回 detached 数据和目标。
- Teacher forward 使用普通无梯度调用；其前向不能污染在线 history/pending buffer。

### 8.4 Student 与 teacher 条件一致性

图像几何变换与随机光度增强每个样本只采样一次。冻结 DINO 对准备好的图像提取一次 patches，student 与 teacher 可读取同一 detached patches；两者分别用自己的可训练 Resampler、状态投影和 gate 构建 context。

这需要将现有 observation encoder 的“图像准备/冻结主干”与“可训练适配器”接口拆清，不重复对 teacher 随机增强，也不把已经需要 student 梯度的 context 当作 teacher 自己的 EMA context。

全 EMA conditioner 是本方案的明确选择；原作者公开实现存在 teacher 使用已有 student 视觉特征的路径，不能声称两者完全相同。[S4] Teacher 与 student 的 feature 数值可因 EMA 权重不同而不同，但观测、历史、mask、augmentation realization 和条件语义必须一致。

若需要共享 DINO/OAT 实例以节省内存，必须先保证 ownership、state_dict 与冻结行为正确；第一版优先保留独立冻结副本，只共享当前 batch 的无梯度 features。

## 9. 采样、缓存与部署步数

一次 chunk 的采样：

```text
context = build_context(current_obs, acknowledged_past, validity)
z = Gaussian(shape=[B,8,5], dtype=float32)
for k = 0 ... N-1:
    t = k / N
    dt = 1 / N
    v = flow_model(z, t, dt, context).float()
    z = z + dt * v
z_grid = snap_codes(z)
actions = frozen_OAT.decode(z_grid)
```

默认 `N=8`，solver 为均匀时间网格 Euler；初版不增加 Heun、自适应 solver 或 classifier-free guidance。两步采样作为训练后部署验收目标，不能仅修改 N 就承诺质量不变。正式部署记录采用的步数、checkpoint、EMA/raw 选择与随机策略。

训练 self-past 初始也采用 8 步；后续如果部署使用 2 步，需要明确重新验收 self-past/推理步数配置，不静默切换。

缓存规则：

- 一次 chunk 内复用 DINO patches、Resampler 输出、ContextBatch 和静态 cross-KV。
- 基线设计不对 memory 施加 t/dt 调制，因此每层 cross-KV 可只计算一次；K cache 存已做 K-Norm 的结果。
- Self-attention 的 8 个输入每一步都变化，禁止把上一 flow step 的 self-KV 当作当前值。
- 若未来启用官方 `pre_norm_modality` 的时间调制 memory，必须每步重算 cross-KV；不能继续使用静态缓存。[S2]
- 缓存仅用于 eval/no_grad generation，不进入 checkpoint，不跨新观测或 episode。
- 训练 activation checkpointing 仅用于无 KV cache 前向，闭包显式传入可变 context/bias，避免重算时读取已被替换的缓存。

为可复现验证，按固定 seed、数据集 identity 与 §11 的稳定 sample_id 派生每个样本的 noise 和验证时间采样；使用固定哈希算法而非进程随机化的 Python hash。不要靠 dataloader batch 顺序或 rank 决定 noise。在线会话的 generator 按 episode 初始化并逐次推进，不能每次调用都无意重置同一个 seed。

若准备函数使用独立 `torch.Generator`，必须逐 rank 保存/恢复 `get_state()` / `set_state()`，包括专用训练noise/时间/划分及self-past generator。全局 CPU/CUDA RNG capture 不会自动涵盖这些独立对象；恢复顺序必须早于下一次随机采样。

## 10. Self-past 与在线执行历史

两个变体都保留 self-past 课程：最大概率 0.5，warmup 1000 次成功 optimizer updates，ramp 4000 次；课程不按 microbatch 或验证调用推进。

- previous-window 生成先于当前训练图，避免两套激活重叠。
- 对 `prev_window_valid=False` 的样本先筛除；其 previous 状态历史可能全无效，不能先送进 history encoder 再屏蔽。
- 使用当前 student 的 eval/no-grad 采样路径生成，默认 8 步 Euler、终点 FSQ 投影、冻结 OAT 解码；每卡 self-past chunk size 默认 2。
- 历史替换发生在 7 维命令空间。生成的 latent 或未量化的速度不能直接进入 raw-action/diff 条件。
- 替换后重新计算命令差分；动作有效性来自真实窗口元数据，不从生成值是否为零推断。
- 同一训练 batch 的 self-past 混合只采样一次，student 与 EMA 共享结果。Teacher 不单独再次生成历史或选择替换样本。
- 保留 `_clean_autocast_cache()` 和 inference-mode 外 clone 的语义，恢复所有 train/eval 状态，DINO/OAT 仍保持 eval。
- 不把生成命令积分为下一实测状态；gate 分支始终读取数据集或控制端提供的实际状态历史。

在线 API：

```python
predict_action(obs_dict, past_actions=None, past_action_valid=None,
               num_flow_steps=None, generator=None)
# {'action': [B,8,7], 'action_pred': [B,16,7]}
record_executed_actions(actions, executed_lengths=None)
reset()
```

显式 past_actions 与 bool validity 必须成对给出，调用为无状态预测，不修改在线 buffer。未给显式历史时使用 `[B,7,7]` 已执行命令与 `[B,7]` validity；predict 只建立 pending，执行端确认后才推进历史。

支持零执行、部分执行、不同环境长度；batch size 变化要求 reset。Reset 清除历史、validity、pending 和生成缓存。两个新变体均采用此执行确认协议。

不把 AR 的 temperature/top-k/use_k_tokens 静默映射成 flow 步数。Flow API 使用明确的 `num_flow_steps` 和 noise generator；初版固定 8 个 latent slots，不引入可变长度 OAT 前缀生成。

## 11. 数据、padding 与动作有效性

### 11.1 本次真机任务

- 数据：`/workspace/ysk/zarr/nut_washer_v3_N77.zarr`。
- 示范：77 条；`val_ratio=0.05`，seed 42；按现有规则预计 73/4，启动时以实际 episode masks 验证。
- 两路 RGB、两帧观测，未来动作 horizon 16，执行步数 8，过去动作 7。
- Gate 使用 8 个实测状态，姿态 rot6d rows，几何运算 FP32。
- 保留指定 tokenizer 的 action normalizer 和训练数据来源，不因切换生成目标重新拟合或替换其统计。
- 真机无模拟器 runner，离线指标不命名为机器人闭环成功率。

两变体均保留 `history_padding=zero`、`return_history_validity=true`；未来动作仍按 tokenizer 训练时的末端重复规则 padding。过去动作 padding 与未来动作 padding 是两套不同约定。

### 11.2 必需 batch 字段

| 字段 | 形状/用途 |
|---|---|
| obs | 当前两帧图像、状态、任务信息 |
| action | `[B,16,7]`，沿用原终端重复 padding |
| past_action / past_action_valid | `[B,7,7]` / bool `[B,7]` |
| prev_obs | previous execution window 的观测 |
| prev_past_action / prev_past_action_valid | `[B,7,7]` / bool `[B,7]` |
| prev_window_valid | bool `[B]` |
| future_action_valid | **拟新增** bool `[B,16]`，只用于物理动作指标 |
| sample_id | **拟新增** int64 `[B]`，绝对动作起点；结合dataset identity形成稳定样本标识 |
| obs/prev_obs 的 state_history__* | 仅 gate，8 个对齐实测状态 |
| state_history_valid | 仅 gate，bool `[B,8]` |

Gate 检查 `past_action_valid == state_valid[:, :-1] & state_valid[:, 1:]`，previous window 同理。不能把 7 位动作 mask 和 8 位状态 mask 直接判等。

### 11.3 Future validity adapter

当前 sequence sampler 的真实样本范围保存在四元组索引中。拟新增 flow 专用 dataset mixin/subclasses，仅增加 metadata，不改现有 action 数组、窗口数、episode split 或 normalizer：

```python
# 当前 PrevWindow/StateHistory 的 action_start == self.pad_before。
result = super().__getitem__(idx)
buffer_start, _, sample_start, sample_end = self.seq_sampler.indices[idx]
positions = self.pad_before + np.arange(self.n_action_steps)
result['future_action_valid'] = torch.as_tensor(
    (positions >= sample_start) & (positions < sample_end),
    dtype=torch.bool,
)
result['sample_id'] = torch.as_tensor(
    buffer_start + self.pad_before - sample_start, dtype=torch.int64,
)
```

`sample_id` 使用绝对动作anchor，不能使用验证子集内idx或rank/batch内序号；结合稳定的数据集fingerprint区分不同数据源。同一物理窗口在train/validation视图、不同batch和rank划分下的标识保持一致。该字段仅用于随机复现、指标和诊断，不进入模型条件。

公式依赖当前采样布局，必须用边界测试锁定；未来修改 dataset 切片时不能无检查地保留旧偏移。

**Latent 监督对全部 8 slots 生效。** 它们压缩了整段动作，不与物理时刻逐个对应，不下采样 future mask，不把 latent 末尾若干项随意置为无效。编码目标沿用原始 padded 16-step chunk，保留 episode 尾部样本覆盖。

`future_action_valid` 不进入 ContextBatch、AdaLN 或在线 API，因为它包含训练样本结束边界的信息。只用于解码动作指标；第一版没有 decoded auxiliary loss。

### 11.4 指标的正确归约

至少报告：

- 固定 seed 下的 validation FM loss。
- 终点量化后的 decoded action MSE：expert history 与 generated history 分开。
- 按平移、旋转增量、夹爪维度分组的动作误差，标明原始单位或归一化尺度。
- 连续 latent 到合法网格的距离、投影前超界比例、投影后合法码比例。
- 相同目标的 OAT autoencoding error，用于区分 codec 本身的重建误差与策略生成误差。

分布式 masked MSE 先汇总平方误差总和与有效元素总数，再相除：

```text
global_mse = all_reduce(sum_squared_error_on_valid_steps)
             / all_reduce(number_of_valid_steps * action_dim)
```

不能平均不同有效长度 batch/rank 的局部 mean。没有有效元素的 batch 不构造 NaN；累计 denominator 为零时明确报告无指标。

验证采用**不补齐样本的分片**：各rank处理互不重叠的验证索引，不让Accelerate的even_batches复制尾部样本。评估使用普通EMA模块（或明确选定的unwrapped raw模块）执行，循环内没有DDP前向collectives；允许各rank验证batch数不同，最后统一归约分子、分母与样本数。不要仅在补齐后的loader上all-reduce就宣称与world size无关。验收要求聚合后的sample_id恰好覆盖验证集一次，包括某rank没有样本的情形。

少步性能以解码结果和闭环结果验收，不能只看 latent MSE 更低。量化前很小的偏差也可能跨越 FSQ 边界，导致动作变化。

## 12. 数值、模式与参数边界

| 运算 | 精度与梯度 |
|---|---|
| DINO trunk | 冻结、eval、no_grad；可 BF16 前向 |
| OAT encode / decode | 冻结、eval、no_grad；首版保留 FP32 以保证 codec 一致性 |
| Latent/grid 数值 | FP32；不保存 teacher autograd graph |
| Resampler / DiT-X / state/history/gate | Student 参与反传，BF16 autocast |
| Norm 统计 | 必要累计 FP32 |
| 旋转几何 | FP32 |
| 时间、插值、teacher target、Euler、snap、loss | FP32 |
| Teacher 所有路径 | eval、no_grad；目标返回普通 detached tensor |

无效过去命令和无效状态在归一化/几何变换前清理，memory 再按 mask 清理，避免 NaN 即使经过 attention mask 仍传播。Closed summary 同样在 K/V matmul 前清理。

每个样本必须至少有一个有效观测 token。当前 gate 状态历史要求最后一个实测状态有效；不能靠构造全无效 dummy history 来代替基础变体。

`history_log_gate=-inf` 时使用 where/masked assignment，禁止 `-inf*0`。训练与采样共用同一 bias 构造代码。

冻结参数不进入 optimizer，teacher 参数不进入 student optimizer。保存/恢复、EMA 更新和参数计数按 Parameter 对象身份去重。

## 13. 优化器、双 4090 与性能验收

### 13.1 正式训练初值

| 项目 | 默认值 |
|---|---|
| Optimizer | AdamW，betas `(0.9,0.95)` |
| Flow / state / history / gate LR | `5e-5` |
| 视觉投影 / Resampler LR | `1e-4` |
| Weight decay | `0.01`；bias、norm 参数不 decay |
| 梯度裁剪 | 1.0 |
| Scheduler | cosine，计划成功 optimizer updates 的前 5% 为 LR warmup |
| Dropout | 本 flow 配方显式 0.0 |
| 图像增强 | 轻量 brightness/contrast，student/teacher 共享一次采样结果 |
| EMA | 必需；配置与计数见 §8 |
| Self-past | p 最大 0.5；1000-update warmup，4000-update ramp |
| 训练每卡 microbatch | 4 |
| World size / accumulation | 2 / 8 |
| 有效 batch | 64 |
| 验证每卡 batch | 4；允许不足 4 的尾批 |
| Self-past chunk size | 2；测量后可提升至 4 |
| Activation checkpointing | 训练 DiT-X/Resampler 开启，no-cache forward |
| 真机训练 epoch 上限 | 2001，沿用当前新真机配置；用户明确覆盖优先 |
| Seed | 42 |

这些是工程初值，不是已经测得的最优超参数。LR warmup、self-past warmup 和 EMA decay 是独立机制；从分布式实际 dataloader 与 accumulation 解析 update 数，不能直接用单进程 batch 数计算。

Epoch 上限不等于必须选最后一个 checkpoint 部署。按留出解码指标保留最佳点、周期恢复点与 latest；明确评价采用的 solver、EMA/raw 和 seed。

### 13.2 容量与显存估算

沿用上一版约 174.3M 公共 AR/Resampler 参数估算：去掉约 3.84M 的 token embedding，再加上 16 个 `768→9×768` 调制投影约 85M，以及时间/状态编码与输出层，预计约 **260M 可训练参数**，额外条件组件使精确值略有变化。

每 rank 的 FP32 权重、梯度、Adam 两个 moments 和 EMA 合计约 `20 bytes × 260M ≈ 4.84 GiB`。这是基础状态，不包括冻结 DINO/OAT 副本、BF16 cast、DDP buckets、激活、SDPA workspace、teacher/self-past 临时内存。

两张 RTX 4090 24GB 预计可支持该配方，但 DDP 不把两卡合并成单个 48GB 显存池，每个 rank 都必须满足单卡峰值限制。两个 variant 默认分别使用双卡先后训练。[S6]

初始 `4 × 2 × 8 = 64`；通过完整峰值检查后可采用 `8 × 2 × 4 = 64`。当前 3:1 FM/CT 划分不支持直接降到每卡 batch 1/2；若最低 4 仍 OOM，应先降低 self-past 分块、核查重复参数/teacher图/缓存并保留 activation checkpointing。再改变批划分算法属于明确的新配方，需要同步更新目标权重和验收。

正式启动前重新检查 GPU 空闲情况，不根据以前 GPU 2/3 空闲的快照自动占用正在工作的设备。本次文档任务没有运行 GPU probe。

### 13.3 必须测量的阶段

- Adam moments 已建立后的训练 forward/backward/update。
- self-past 达到最大混合概率、存在有效 previous windows 的 batch。
- CT teacher 与 student 图的实际生命周期；确认 teacher 无梯度、没有额外副本堆积。
- Expert/generated-history validation 与 OAT 解码。
- 真实 DINO、16×768 模型、8 步以及拟部署的 2 步完整 predict_action。
- 两 rank DDP 通信、累积及 epoch 尾部 flush。

记录 peak allocated/reserved、step 时间、batch=1 完整推理 p50/p95、GPU型号、精度、步数、相机/帧数、activation checkpointing 与权重选择。GPU计时需同步，不能把异步 kernel 提交时间当作延迟。

建议正式长跑前保留约 2 GiB reserved 余量。不能从 8 次 AR token decode 与 2 次 flow 网络调用的数量直接推导固定倍数加速。

## 14. Artifact、恢复与离线部署

采用独立 family：`policy_family=oat_latent_flow`，variant 明确为 `p2n_latent_flow` 或 `p2n_state_gate_latent_flow`。AR、两个 flow 变体和其他任务的 checkpoint 不能相互作为 resume。

完整 artifact 必须包括：

- Policy family、variant、任务类型与 artifact schema version。
- DINO model/config/processor/revision/权重；OAT 结构、EMA来源摘要、权重与 normalizer。
- FSQ levels、码维度、latent horizon、投影公式版本和 rounding 约定。
- DiT-X、AdaLN、QK-Norm、FFN、输出层和初始化配置。
- Context schema、相机/帧/状态顺序、history 参数与 rot6d rows 布局。
- 时间采样分布、CT relative-dt 约定、3:1 划分和 loss 权重。
- Solver、时间网格、默认采样步数、self-past 步数与 noise seed 策略。
- Student、完整 EMA、optimizer、scheduler、EMA schedule计数、成功更新计数、self-past计数。
- 各 rank 全局RNG及每个专用torch.Generator状态、数据 episode masks/identity、稳定sample_id定义、软件与相关源码版本。

Fresh training 读取获准的本地 DINO 与指定 OAT checkpoint。Resume/deployment 从 artifact 内结构先构造，再 strict load；**DINO 和 OAT 均不能在构造时依赖原路径或联网下载**。

部署优先使用保存的 EMA 推理权重，部署 artifact 可以不含 optimizer 或另一套 student，但仍必须自包含 DINO/OAT。恢复训练需要包含 student、EMA 与训练状态的完整 artifact。

Teacher 引用、PreparedFlowBatch、KV cache、在线过去命令/validity/pending 不作为长期 checkpoint 状态。加载后 eval/reset，重新建立控制会话。

Resume 时严格核对 family、variant、FSQ与动作schema、数据划分、loss与solver约定。不使用 `strict=False` 隐藏 missing/unexpected keys。AR 到 flow 的可训练参数 warm-start 不是本次默认路径。

## 15. 拟新增文件与有限适配范围

以下均为未来实施项，本次未创建这些代码文件。

| 文件 | 职责 |
|---|---|
| `oat/model/flow/ditx_latent.py` | DiT-X latent velocity network、AdaLN、时间/状态编码 |
| `oat/model/flow/consistency_flow.py` | FM/CT采样、FP32目标、loss归约 |
| `oat/model/flow/euler_sampler.py` | 连续latent Euler采样、静态context缓存 |
| `oat/tokenizer/oat/latent_adapter.py` | FrozenOATLatentAdapter、合法网格投影、round trip |
| `oat/policy/p2n_latent_flow_common.py` | Flow policy 公共构造、loss forward、采样、self-past |
| `oat/policy/p2n_latent_flow.py` | `P2NLatentFlowPolicy`，基础变体 |
| `oat/policy/p2n_state_gate_latent_flow.py` | `P2NStateGateLatentFlowPolicy`，状态历史门控变体 |
| `oat/dataset/latent_flow_dataset.py` | Flow数据集mixin/子类，增加future validity与稳定sample_id |
| `oat/workspace/train_p2n_latent_flow.py` | 独立flow工作区：EMA生命周期、PreparedFlowBatch、指标 |
| `oat/common/latent_flow_batch.py` | 拟用PreparedFlowBatch数据合同和校验 |
| `scripts/train_p2n_latent_flow.py` | 显式variant/task、配置解析、预检和双卡启动 |
| `train_p2n_latent_flow.sh` | 薄launcher，不覆写用户超参数 |
| `tests/test_p2n_latent_flow_*.py` | 码空间、目标、门控、恢复、数据和运行合同 |
| `tests/p2n_latent_flow_ddp_smoke.py` | 双rank EMA/梯度/恢复 smoke |

有限修改现有模块：

- observation encoder：拆出 frozen-patch 提取与可训练适配入口，保留旧 forward 行为。
- ContextBatch：将布局校验从 AR variant 命名中分离，保留旧API兼容。
- P2N 公共helper：提取context/history/self-past/执行确认，避免复制后出现行为分叉。
- capability resolver：新增明确的flow/loss-preparation能力，旧策略保持保守回退。
- runner：复用执行确认与状态历史采集逻辑，使其按能力调用，不硬编码 AR policy 名称或采样参数。

不重写旧 AR loss、不改变旧 checkpoint 加载、不修改历史命令更新时间。必须检查没有把未用的 AR、teacher 或 gate 模块遗留在 student module tree。

## 16. 四个配置与启动接口

### 16.1 配置矩阵

| 任务 | Variant | 拟新增配置（相对 oat/config） |
|---|---|---|
| LIBERO | p2n_latent_flow | `train_p2n_latent_flow.yaml` |
| LIBERO | p2n_state_gate_latent_flow | `train_p2n_state_gate_latent_flow.yaml` |
| 真机 | p2n_latent_flow | `experimental/train_p2n_latent_flow_real_robot.yaml` |
| 真机 | p2n_state_gate_latent_flow | `experimental/train_p2n_state_gate_latent_flow_real_robot.yaml` |

数据集在 `oat.dataset.latent_flow_dataset` 中分别提供 `LatentFlowZarrDatasetWithPrevWindow`、`LatentFlowZarrDatasetWithStateHistory`、`LatentFlowRealRobotZarrDatasetWithPrevWindow`、`LatentFlowRealRobotZarrDatasetWithStateHistory`，继承相应现有数据集并复用 future-validity mixin。

真机 gate 类通过显式state schema 使用现有 rot6d encoder；LIBERO 使用匹配的四元数状态布局。真机 runner 为 null；LIBERO runner 保留相应执行确认/状态历史能力，需通过新的variant验收。

**当前指定 tokenizer 只固定用于 nut_washer 真机两变体。** LIBERO 配置必须提供任务匹配的 tokenizer，不能因为 action_dim 同为7就自动复用真机normalizer和动作语义。

### 16.2 核心配置示意

下面是拟定 schema，只用于指导实现，字段尚不保证当前可执行：

```yaml
policy_family: oat_latent_flow
variant: p2n_latent_flow
task_type: real_robot

policy:
  _target_: oat.policy.p2n_latent_flow.P2NLatentFlowPolicy
  variant: ${variant}
  construction_mode: fresh
  embed_dim: 768
  n_layers: 16
  n_heads: 12
  ffn_dim: 2048
  dropout: 0.0
  num_visual_queries: 64
  resampler_depth: 2
  activation_checkpointing: true
  flow:
    latent_space: fsq_normalized_codes
    num_slots: 8
    code_dim: 5
    levels: [8, 5, 5, 5, 5]
    fm_fraction: 0.75
    ct_weight: 1.0
    fm_beta: [1.0, 1.5]
    fm_time_scale: 0.999
    ct_time_bins: 10
    teacher_dt_mode: same_relative_dt
    solver: euler
    inference_steps: 8
    self_past_steps: 8
    endpoint_projection: fsq_nearest_grid
  self_past_chunk_size: 2
  self_past_p: 0.5
  self_past_warmup_steps: 1000
  self_past_ramp_steps: 4000

training:
  use_ema: true
  allow_bf16: true
  gradient_accumulate_every: 8
  num_epochs: 2001
  resume: false

dataloader:
  batch_size: 4
  drop_last: true

val_dataloader:
  batch_size: 4
  drop_last: false
```

代码必须读取并校验上述字段，不能只把无效开关写进 YAML。FSQ shape/levels 从实际 tokenizer 再核验；配置中的声明不能覆盖 checkpoint 结构。

Gate 配置单独增加state/history字段，history encoder dropout也为0；基础变体拒绝用于构造额外gate模块的字段。Artifact保存完整 resolved config，不能依赖未来defaults变化。

### 16.3 Launcher 合同与未来命令

Launcher 必须显式选择variant、task和GPU；预检数据schema、tokenizer provenance、DINO本地权重、EMA要求、microbatch整除性、输出目录和resume兼容性。Dry-run只解析/检查，不构造GPU模型或训练。

以下是**未来接口示意，当前不可直接当作已实现命令**。默认真机配置内写入 §4 指定的 tokenizer；允许用户显式覆盖为通过相同schema/provenance校验的来源：

```bash
/venv/real_robot/bin/python scripts/train_p2n_latent_flow.py \
  --variant p2n_latent_flow --task real_robot \
  --gpus 2,3 \
  --dino /path/to/dinov3_s_snapshot \
  --output output/training/nut_washer_p2n_latent_flow_seed42 \
  -- dataloader.batch_size=4 training.gradient_accumulate_every=8

/venv/real_robot/bin/python scripts/train_p2n_latent_flow.py \
  --variant p2n_state_gate_latent_flow --task real_robot \
  --gpus 2,3 \
  --dino /path/to/dinov3_s_snapshot \
  --output output/training/nut_washer_p2n_state_gate_latent_flow_seed42 \
  -- dataloader.batch_size=4 training.gradient_accumulate_every=8
```

两条命令表示两个独立训练任务，默认先后运行。GPU编号只是示例，执行前重新确认空闲设备；launcher设置可见GPU并启动两个进程。

所有batch、epoch、checkpoint、device覆盖原样透传；不能由文件名含gate与否触发旧launcher的batch64/val32覆盖。Resume单独提供完整训练artifact，不重新要求原DINO/OAT外部路径。

交付时为四个配置各提供经过解析与smoke验证的实际命令，并将本节示意更新为真实schema。

## 17. 实施顺序与阶段出口

| 阶段 | 工作 | 完成条件 |
|---|---|---|
| A | Frozen OAT latent adapter、schema/provenance | 8×5码正确，全部合法码round trip，原始动作解码一致 |
| B | 提取公共context与frozen-patch接口 | 两变体267/271布局正确，AR既有行为仍通过相关回归 |
| C | DiT-X与8步Euler | shape/时间/FP32/非因果/缓存合同正确，真实latent生成链路可运行 |
| D | FM目标与独立正确性检查 | 合成已知速度场、真实batch有限loss/backward通过 |
| E | EMA CT与单次DDP forward | 初始同步、独立teachercontext、梯度累积和loss参考结果通过 |
| F | Self-past、gate、执行确认 | history/noise/validity一致，closed gate不泄漏，在线状态协议通过 |
| G | Dataset、masked指标、四配置与artifact | 数据边界正确，离线恢复、跨family拒绝与resume通过 |
| H | 两variant真实单卡及双ranksmoke | 包含CT/self-past/EMA/Adam的峰值、梯度与运行结果可核验 |
| I | 按本次真机配方分别训练 | 两套独立checkpoint与固定评估协议下的结果 |
| J | 两步部署验收 | 解码误差、延迟和闭环性能达到接受条件后明确保存2步部署配置 |

阶段D的FM检查是目标/梯度验证，不要求额外长跑FM基线。正式模型采用联合FM+CT配方；没有详细架构消融任务。

阶段H不得仅用mock DINO或tiny DiT宣告完成。两变体最终均需真实DINO、选定OAT和16×768模型验收。两步采样若质量不足，保留通过验收的8步部署；不以少步数本身替代任务质量。

## 18. 必要验证矩阵

| 检查 | 必须成立 |
|---|---|
| OAT编码合同 | 原始16×7动作得到8×5码；normalizer只应用一次 |
| FSQ全码round trip | 5000个code的embedding→snap→indices保持一致 |
| 网格边界 | 第1维上界0.75，其他1；超界投影正确；NaN/Inf被拒绝 |
| 解码等价 | decode(grid)与detokenize(indices)一致，codec权重未变 |
| ODE中间状态 | 不做snap/clamp；终点恰好投影一次，无重复tanh |
| 时间/速度符号 | t=0为noise、t=1为target；恒定已知速度可积分到已知终点 |
| FM采样 | Beta参数与0.999缩放正确，dt=0，张量batch维保留 |
| CT目标 | 同一noise/target构造z_next，teacher停止梯度，dt约定和边界正确 |
| CT端点 | t_next=1时target终点为z1，起始t不为1，无除零 |
| 批划分 | B=4时3FM/1CT；全部样本恰好分配一次，两项mean分别归约 |
| Packed loss | packed FM/CT与独立参考计算loss及梯度一致 |
| 条件一致性 | teacher/student共享patch realization与混合history，各用自己的adapter |
| Fresh EMA | 两rank不同初始RNG下，同步后student与初始EMA一致 |
| EMA生命周期 | 全程eval/no_grad，无teacher DDP forward；仅成功optimizer update后更新 |
| 参数/图归属 | student state_dict无teacher子树；optimizer无冻结/teacher/重复参数 |
| AdaLN零初始化 | 第一轮允许零梯度；多次更新后相关模块有有限有效梯度 |
| 基础变体 | 不提供state-history也能训练/推理，无额外history/gate参数 |
| Gate变体 | 缺实测历史明确报错，8状态/7命令对齐，rot6d几何FP32 |
| Closed gate | 固定其他输入扰动summary，输出不变；AdaLN无摘要旁路 |
| Attention mask | 无效/closed memory含任意填充值不影响有效输出，不产生NaN |
| Cache | 静态cross-KV与每步重算一致；无跨flow-step self-KV复用 |
| BF16与self-past | 无inference-tensor或autocast-cache污染，模式恢复正确 |
| 未来有效性 | 起始/中间/最后一步/短episode/相邻episode正确，最后一步仅一项有效 |
| 数据兼容 | 新adapter不改变action bytes、padding、窗口数、episode划分 |
| 指标归约 | 不同rank/batch划分的全局masked MSE相同，future mask不进入条件 |
| 验证尾部 | 样本数不能整除world_size×batch时仍无重复/遗漏；空rank正确参与最终归约 |
| 样本与随机数 | sample_id不依赖rank/视图；固定seed时验证noise一致；专用generator恢复后采样连续 |
| 显式history调用 | 不改变在线buffer/validity/pending |
| 执行确认 | 零/部分/不同长度正确，reset清空状态，batch变化要求reset |
| Resume | 保存的EMA差异、schedule、optimizer、课程、RNG连续，不被student覆盖 |
| 离线加载 | 移走外部DINO/OAT路径并禁网后仍能strict restore |
| Family/variant拒绝 | AR与flow、plain与gate之间错误resume明确拒绝 |
| 双卡完整运行 | 两变体真实模型均完成CT/self-past训练与8步生成，无DDP unused参数错误 |
| 部署步数 | 8步与2步各自记录指标；不混用生成预算或宣称未测速度提升 |

检查聚焦数学、输入、梯度、状态和部署合同，不添加机械镜像实现的测试。共享helper修改只回归受影响AR路径，不为纯文档改动执行训练测试。

## 19. 完成交付与结果边界

- [ ] 两个独立flow策略、明确variant/family与配置。
- [ ] 指定冻结OAT的EMA来源、哈希、结构和normalizer元数据。
- [ ] 精确8×5 FSQ码空间与终点投影，保留原encode/decode。
- [ ] 冻结DINOv3-S/16与可训练Resampler，学生/教师条件边界明确。
- [ ] 16×768 DiT-X、AdaLN-Zero、SwiGLU、QK-Norm及静态memory缓存。
- [ ] FM+CT、正确时间采样、独立EMA teacher和成功update计数。
- [ ] 两变体self-past、summary门控、动作validity与执行确认。
- [ ] Future有效性metadata与全局masked decoded指标。
- [ ] 四配置入口、独立launcher、完整preflight和用户override支持。
- [ ] 自包含artifact、离线部署、训练恢复与family/variant校验。
- [ ] 两variant真实双4090训练峰值及完整推理测量。
- [ ] 按固定数据划分和闭环协议报告的结果，明确最终部署步数。

保留OAT也保留其量化与重建误差；latent flow不会自动消除这些误差。对量化码分布训练连续flow再投影是本方案的工程适配，少步下的网格边界误差必须实测。77条示范下，更大容量和更低latent loss均不足以单独证明实际任务更强。

论文报告的1–2步性能、当前估算的260M参数与双4090可行性，都应与本任务实测区分。新文档不构成代码已实现、依赖已就绪、模型已训练或真实机器人已验证的声明。

## 20. 资料与设计来源

- [S1：ManiFlow 论文](https://arxiv.org/html/2509.01819v1)：flow/consistency训练与DiT-X动机。
- [S2：原作者 DiT-X block](https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/model/diffusion/ditx_block.py)及[DiT-X model](https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/model/diffusion/ditx.py)：调制、初始化及memory时间调制。
- [S3：原作者时间采样实现](https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/model/common/sample_util.py)。
- [S4：原作者 image policy](https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/policy/maniflow_image_policy.py)及[配置](https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/config/maniflow_image_timm_policy_robotwin.yaml)：FM/CT目标、批划分与relative-dt。
- [S5：原作者 EMA](https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/model/diffusion/ema_model.py)及[训练 workspace](https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/workspace/train_maniflow_robotwin_workspace.py)。
- [S6：PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)及[Activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)。
- [DINOv3 官方仓库](https://github.com/facebookresearch/dinov3)与[所选模型](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)。
- [本仓库 OAT](../oat/tokenizer/oat/tokenizer.py)、[FSQ](../oat/tokenizer/oat/quantizer/fsq.py)、[SO3Aug](../oat/tokenizer/oat/tokenizer_so3_aug.py)、[sequence sampler](../oat/common/seq_sampler.py)。

参考实现的具体commit应在开始实施时固定并记录。本文的OAT latent-space适配、终点投影、EMA全conditioner、零dropout、历史门控和资源配置是针对本仓库的明确设计选择，不宣称全部来自原ManiFlow。

[S1]: https://arxiv.org/html/2509.01819v1
[S2]: https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/model/diffusion/ditx.py
[S3]: https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/model/common/sample_util.py
[S4]: https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/policy/maniflow_image_policy.py
[S5]: https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/workspace/train_maniflow_robotwin_workspace.py
[S6]: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
