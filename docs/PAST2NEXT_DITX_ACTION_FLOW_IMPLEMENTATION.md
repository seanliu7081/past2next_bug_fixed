# Past2Next 历史扩展：DINOv3-S/16 + ManiFlow DiT-X 直接连续动作实现计划

文档日期：2026-09-24

目标仓库：`/workspace/ysk/past2next_bug_fixed`

状态：**待实现的技术规范。本次仅新增本文，不修改实现代码、下载权重或启动训练。**

关联方案：[OAT 自回归版本](PAST2NEXT_DINOV3_SMALL_IMPLEMENTATION.md)、[冻结 OAT latent-flow 版本](PAST2NEXT_DITX_LATENT_FLOW_IMPLEMENTATION.md)。三份文档分别维护，不相互替代。

## 1. 用户确认的范围

本分支直接生成连续动作，生成路径更接近原始 ManiFlow。用户在澄清后明确要求保留两种 Past2Next 历史条件扩展：

| 名称 | 实际含义 | 新 variant |
|---|---|---|
| 动作历史版 | ManiFlow 直接动作生成器，加过去动作和命令差分条件 | `p2n_action_flow` |
| 状态历史门控版 | 在动作历史版上再增加实测状态历史、摘要和门控 | `p2n_state_gate_action_flow` |

两者是本仓库的正式交付目标。这里保留 p2n 名称表示历史条件设计，**不表示继续使用 OAT 自回归生成器**。也不把基础版本实现为 gate=closed，因为后者仍包含额外模块与输入要求。

“更接近 ManiFlow”具体指：在连续动作空间训练 flow matching + consistency，使用 DiT-X 的 LayerNorm、GELU 和 AdaLN-Zero 核心。DINOv3、空间 Resampler、16层容量、Past2Next history/self-past/gate 和部分工程训练设置属于我们的扩展。因此本文不是未经改动的官方模型复现，也不直接承诺论文中的任务成功率或少步效果。[S1][S2]

**本分支不加载 OAT encoder、FSQ 或 OAT decoder。** 用户此前选定的冻结 tokenizer 继续服务于另外两条 OAT 路线，不成为本分支的训练、恢复或部署依赖。无需请求、复制或占用其网络权重。

本次不增加标准无历史的第三个变体、不展开架构消融矩阵，也不引入 latent-flow 的量化或辅助解码目标。

## 2. 三条路线的准确区别

| 项目 | OAT 自回归 | OAT latent flow | 本文直接 action flow |
|---|---|---|---|
| 预测对象 | 8 个离散 FSQ IDs | 8×5 FSQ 标量码 | **16×7 归一化动作** |
| 生成模型 | Causal AR Transformer | DiT-X 风格向量场 | DiT-X 连续动作向量场 |
| 训练目标 | Token CE | Latent FM + CT | **Action FM + CT** |
| 推理后处理 | OAT detokenize | 终点网格投影 + OAT decode | **动作反归一化一次** |
| 依赖 OAT checkpoint | 是 | 是 | **否** |
| 量化边界误差 | 有 | 有 | 无 FSQ 量化环节 |
| 两种历史条件 | 保留 | 保留 | 保留 |
| 已证明优于其他路线 | 否 | 否 | 否 |

直接动作版本移除了 codec 的表示约束，同时需要自行学习动作分布与轨迹协调。不能把“没有量化误差”推导为必然更高的闭环成功率。

## 3. 整体数据流与默认规格

```text
两相机 × 两帧 RGB
→ 冻结 DINOv3-S/16
→ 可训练投影 + 2层 Resampler
→ 256 个视觉 tokens

视觉 + 当前状态 + 过去7步命令 + 命令差分
→ 267-token context
→ gate 变体额外增加4个实测历史 summaries：271-token context

Gaussian [B,16,7]
→ 条件 DiT-X 向量场，Euler 更新全部16个动作位置
→ 归一化连续动作 [B,16,7]
→ action_normalizer.unnormalize，恰好一次
→ 原始动作 [B,16,7]
→ 执行前8步 → 执行确认后更新历史
```

| 模块 | 本方案配方 |
|---|---|
| DINO | 冻结 `facebook/dinov3-vits16-pretrain-lvd1689m` |
| 图像 | 原始128×128 RGB，完整视野重采样到224×224 |
| DINO features | 每图196×384，排除CLS/register |
| 视觉适配器 | 384→768；2层Resampler，64 queries/图，12 heads |
| Resampler FFN | 沿用公共SwiGLU，intermediate=2048；属于我们的视觉扩展 |
| DiT-X | **16层、768维、12 heads**，每head64维 |
| 动作位置 | 16个可学习时间位置，对应当前锚点t至t+15 |
| 动作输入/输出 | 7→768；每位置输出7维速度 |
| Self-attention | 双向，无causal mask，无self QK-Norm |
| Cross-attention | QKV含bias；Q/K按head做LayerNorm，保留1/sqrt(64) scaling |
| Block归一化 | 无affine LayerNorm，eps=1e-6 |
| DiT-X FFN | **GELU(tanh)，768→3072→768，含bias** |
| 调制 | 每层Linear(768,9×768)，self/cross/FFN各有shift、scale、residual gate |
| AdaLN全局条件 | **仅t与相对dt**；本体状态和历史保留在cross memory |
| 最终层 | RMSNorm→Linear(768,768)→GELU→Linear(768,7) |
| 采样 | 初始8步Euler；两步作为训练后部署验收目标 |
| 执行 | 默认16步预测、前8步执行 |
| 双卡起点 | 每卡batch4，2 ranks，accumulation8，有效batch64 |

DiT-X核心采用原作者模块类型，而非照抄latent-flow方案的SwiGLU与双侧QK RMSNorm。Resampler与动作主干的FFN配置分开命名，避免把2048误传为DiT-X的3072。

所查官方2D配置为12层、768维、8 heads，并显式开启cross QK-Norm和qkv_bias；我们沿用组件语义，将容量保持为此前选定的16×768/12-head。官方默认2D视觉路径也不等于本方案的256个空间tokens；DINO+Resampler是明确改造。[S2][S3]

## 4. 动作语义与 normalizer

### 4.1 当前真机任务

- 数据：`/workspace/ysk/zarr/nut_washer_v3_N77.zarr`，77条示范。
- 划分：val_ratio=0.05，seed=42；预计73 train/4 validation，以实际episode masks验证。
- 当前观测：两路RGB、两帧状态；姿态观测为rot6d rows。
- 动作：7维；位置增量、3维rotation-vector delta、夹爪绝对命令。
- 现有数据说明中夹爪命令0=open、1=closed，状态夹爪宽度使用毫米；状态与动作量纲不能混用。
- 真机没有训练期间的模拟器runner；离线指标不报告成闭环成功率。

动作的单位、控制坐标系、delta含义和控制频率从数据与控制端schema固定并写入artifact，不由模型名字推断。旋转分量采用现有欧氏rotvec增量表示；本模型不是SO(3)流形上的内禀flow。

### 4.2 本分支自己的归一化

Fresh training直接复用数据集的训练集normalizer逻辑：在**训练episode的真实replay frames**上拟合action/state的逐维affine统计，默认mode=limits。不能在重叠窗口或末端padding后的数组上重新拟合，也不能使用validation frames。

现有实现依据：[real_robot_dataset.py](../oat/dataset/real_robot_dataset.py)、[zarr_dataset.py](../oat/dataset/zarr_dataset.py)、[normalizer.py](../oat/model/common/normalizer.py)。

- 未来目标、过去动作和生成历史使用同一action normalizer。
- 归一化后的动作作为flow目标；最终结果恰好反归一化一次。
- 状态使用自己的字段normalizer，不能用action统计归一化rot6d或夹爪测量值。
- DINO使用其processor规定的RGB范围与mean/std，跳过旧ResNet RGB归一化，避免重复处理。
- normalizer的常量维处理沿用现有实现，并以有限性/round-trip测试确认。
- normalizer拟合后冻结，存入student/EMA/artifact；resume/deployment只恢复，不重新拟合。

本分支无需从OAT checkpoint抽取normalizer。即使同一数据划分得到相同统计，也只是结果一致，不构成对OAT权重路径的依赖。

### 4.3 连续输出边界

Limits normalization将训练范围映射到固定尺度，**不表示采样值天然限制在[-1,1]**。Gaussian、中间flow状态与网络输出不加入tanh或逐步clamp；终点保持连续并反归一化，不执行FSQ projection、整数化或OAT decode。

执行端如已有平移/转动限幅或夹爪离散转换，保持其明确的控制接口并记录该转换。策略不擅自引入新的0.5阈值，也不把控制端限制伪装成模型输出。执行确认必须记录控制端实际发送的命令，包括已有后处理造成的变化。

## 5. 两种历史条件与时间对齐

| Segment | p2n_action_flow | p2n_state_gate_action_flow |
|---|---:|---:|
| VISUAL | 256 | 256 |
| PROPRIO | 2 | 2 |
| RAW_ACTION | 7 | 7 |
| ACTION_DIFF | 2 | 2 |
| HISTORY_SUMMARY | 0 | 4 |
| 合计 | **267** | **271** |

相机、观测帧、Resampler query、条件类型与历史相对时间均显式编码。task_uid已进入当前本体/任务特征，不额外增设未声明token。

以决策时刻t表示：观测窗口为[t−1,t]，历史命令覆盖[t−7,t)，gate状态历史为[t−7,t]，目标动作为[t,t+15]。

**当前仓库dataset已经将action切到当前执行锚点。** 推理返回`action_pred[:, :8]`，不能复制其他仓库的`start=n_obs_steps−1`再偏移一次，导致执行从t+1开始。

基础变体不创建history encoder、summary gate或其专用池化模块，不要求state_history字段。Gate变体保留8状态/7命令、内部宽度128的history encoder、4 summaries、hidden128且初始0.9的learned gate。真机继续使用rot6d rows几何，LIBERO使用相应四元数schema。

`past_action_valid`为bool[B,7]，`state_history_valid`为bool[B,8]。Gate检查动作validity与相邻状态的transition mask一致；真实零动作仍可有效，不以数值为零判断padding。

历史gate仅在summary列的attention logits加logsigmoid bias；closed为−inf，raw-action/diff仍可见。AdaLN的时间条件不读取summary、history pooled向量或历史valid fraction，因此不会绕过summary gate。

所有cross-attention接入ContextBatch的padding/summary additive bias，不能沿用原作者某路径中忽略mask的调用。Invalid/closed memory在K/V matmul前清理，使用where构造bias，避免NaN和−inf×0。

## 6. DiT-X 网络实现合同

网络输入：`x_t[B,16,7]`、FP32时间`t[B]`、相对步长`dt[B]`与ContextBatch。输出为同形状速度。

`t`、`dt`分别经过128维sinusoidal embedding与`128→512→768`的Mish MLP，拼接后线性映射至768。全局条件仅来自t/dt；当前状态通过PROPRIO tokens进入cross-attention，与官方时间调制思路一致。[S2]

每层：

```text
shift_sa, scale_sa, gate_sa,
shift_ca, scale_ca, gate_ca,
shift_ff, scale_ff, gate_ff = Linear(SiLU(c_time))

u = LayerNorm_sa(x) * (1 + scale_sa) + shift_sa
x = x + gate_sa * SelfAttention(u, causal=False)

u = LayerNorm_ca(x) * (1 + scale_ca) + shift_ca
x = x + gate_ca * CrossAttention(u, memory, padding_and_summary_bias)

u = LayerNorm_ff(x) * (1 + scale_ff) + shift_ff
x = x + gate_ff * GELU_MLP(u)
```

Cross Q/K LayerNorm位于head reshape后沿64维执行，具有其自身的affine参数；它与block的无affine LayerNorm是不同层。Self-attention维持普通QKV，不新增QK RMSNorm。

所有16个noisy动作位置可互相读取，目标动作不作为额外输入；双向attention不意味着把未来真实动作泄漏给条件。

初始化沿用DiT-X零初始化机制：每层AdaLN投影与最终输出线性层置零，其他新增投影使用记录的Xavier/normal初始化。冻结DINO不参与递归初始化。第一轮部分参数零梯度是预期现象，检查若干次更新后的有效梯度，不要求第一步所有参数非零。

本工程初值将DiT-X、Resampler和history encoder等可训练模块dropout显式设为0.0，配合共享观测增强降低teacher/student条件随机差异。官方self-attention常用0.1；零dropout是我们的工程选择，不是声称官方默认如此。保留轻量brightness/contrast增强与weight decay。

`pre_norm_modality=false`：不对memory使用依赖t/dt的AdaptiveLN。每个chunk的context固定，使静态cross-KV可复用。

## 7. 直接动作 FM 与 consistency training

### 7.1 Flow matching

令`a_raw[B,16,7]`为数据集目标，包括原有终端重复padding。使用本分支action normalizer：

```text
x1 = normalize(a_raw)
epsilon ~ Normal(0, I), shape=[B,16,7]
x_t = (1-t)*epsilon + t*x1
v_target_FM = x1 - epsilon
L_FM = mean((v_student(x_t,t,0,context) - v_target_FM)^2)
```

所有16×7归一化坐标默认同权。这里预测的是从噪声到动作的速度，不是DDPM epsilon，也不是OAT潜变量。

### 7.2 Consistency target

CT子批使用相同的x1和epsilon构造两个点：

```text
t_next = min(t + dt, 1)
x_t    = (1-t)      * epsilon + t      * x1
x_next = (1-t_next) * epsilon + t_next * x1

# EMA、eval、no_grad
v_next = v_ema(x_next, t_next, dt, ema_context)
x1_est = x_next + (1-t_next)*v_next
v_target_CT = stop_gradient((x1_est - x_t) / (1-t))

L_CT = mean((v_student(x_t,t,dt,student_context) - v_target_CT)^2)
L_total = L_FM + L_CT
```

`x_next`来自真实目标和noise插值，不能替换为student先走一步。Teacher和student使用同一观测增强结果、同一组self-past混合命令及validity，各自使用自己的可训练条件模块。

`t_next=1`的行直接令`x1_est=x1`，可跳过teacher计算。其余teacher估计不clamp、不反归一化、不做controller后处理。起点t不能为1。

### 7.3 确定的时间配方

| 项目 | 默认 |
|---|---|
| FM t | `0.999 * Beta(1.0,1.5)` |
| FM dt | 0 |
| CT t | `randint(0,10)/10`，即0至0.9的十点网格 |
| CT dt | `Uniform[0,1)` |
| Teacher t | `min(t+dt,1)` |
| Teacher dt | 同一个原始relative dt；t_next截断后不另改dt条件 |
| 每microbatch分配 | 75% FM，25% CT，随机选行 |
| Loss归约 | 两子批分别mean，`L_FM + L_CT`，各权重1 |

上述时间/相对步长与配比依据官方公开实现。[S4] 75/25是样本比例，不是再把两个loss乘0.75/0.25。

训练每卡batch必须是4的倍数且至少4，drop_last=true；B=4即3个FM、1个CT。不要用两个独立int截断导致漏样本或空子批；不要用裸squeeze丢失CT子批为1时的batch维。

验证尾批允许任意大小，默认对所有样本计算固定seed的FM loss与生成动作指标，不强制3:1切分。额外CT诊断若启用，另行定义全样本CT指标，不与训练混合loss混名。

时间、noise、插值、teacher终点、除法、loss reduction与Euler更新使用FP32；网络主体BF16。默认CT t使分母至少0.1，仍需finite与范围校验。

## 8. EMA、条件准备与 DDP

EMA是CT的必要依赖：覆盖DiT-X、投影、Resampler、状态/历史/gate等可训练模块。DINO冻结，teacher必须eval、无梯度、作为普通device-local模块运行，不包装teacher DDP，也不注册为student的子模块。

Fresh training流程：

1. 设置种子，构造student，拟合并设置训练集normalizer。
2. DDP/Accelerate同步student参数与buffers。
3. 从同步后的student初始化EMA，确保各rank初始teacher一致。
4. 首个CT batch前检查normalizer及teacher参数一致性。

Resume分别恢复student与EMA，不把保存的EMA覆盖成student。恢复EMA schedule、optimizer、scheduler、self-past计数和RNG后再采样下一batch。

EMA配方：`update_after_step=0, inv_gamma=1, power=0.75, min_value=0, max_value=0.9999`。只在成功optimizer update后更新一次；累积microbatch、验证、skipped step不推进。不能把原作者workspace的microbatch EMA更新位置直接照搬到本项目的梯度累积中。[S5]

工作区准备detached `PreparedActionFlowBatch`：

```text
原始batch
→ 一次self-past混合
→ 一次光度增强与冻结DINO patch提取
→ action normalize、Gaussian/t/dt、FM/CT row selection
→ EMA自己的adapter/history/gate构造teacher context
→ no_grad consistency targets
→ 一次DDP student forward
→ student自己的可训练context + packed FM/CT网络 + loss
```

- Student的全部可训练context计算必须在DDP forward内部，不在unwrap后绕过DDP建立训练图。
- 可共享冻结DINO的detached patch tensors；student/teacher Resampler等使用各自参数。
- Teacher不能直接把student_context.detach当作自己的EMA context；全EMA conditioner为本工程明确选择，与原作者已有student视觉特征复用路径区分。
- 不在传给DDP的batch中携带teacher module/reference；PreparedBatch只含数据与detached targets。
- 使用no_grad准备targets，防止inference tensors进入需要保存输入的学生反传；保留self-past的autocast cache清理与退出inference模式后的clone。
- 同一随机增强结果和历史替换结果供两侧使用，不能teacher再采样一次augmentation/history。
- 不沿用现有工作区每epoch对EMA调用train的行为；本flow teacher始终eval。

## 9. 采样、缓存与在线动作

```text
context = build_context(obs, acknowledged_past, validity)
x = Gaussian([B,16,7], dtype=float32)
for k in range(N):
    t  = k / N
    dt = 1 / N
    x  = x + dt * flow_model(x, t, dt, context).float()
action_pred = action_normalizer.unnormalize(x)
action = action_pred[:, :8]
```

默认N=8、均匀Euler；正式训练后将N=2作为部署验收目标。原论文的1–2步能力不能直接替代本任务测量。[S1] 初版不加入Heun、自适应solver、CFG或midpoint clamp。

- DINO features、Resampler输出、context和每层cross-KV在一个chunk内复用。
- 每一步全部16个动作位置都改变，因此self-KV不可跨flow step复用。
- 若未来改变为时间调制memory，必须取消静态cross-KV或每步重算。
- Cache仅用于eval/no_grad生成，不跨新观测、episode，不进入artifact。
- Activation checkpointing只用于无cache训练前向。
- 验证noise/time由固定seed、dataset identity和stable sample_id决定，不能随rank或batch位置改变。
- 专用torch.Generator逐rank保存/恢复get_state；全局CPU/CUDA RNG快照不能替代它。
- 在线generator按episode初始化并推进，不在每次调用偷偷重设相同seed。

策略API使用num_flow_steps/generator，不接受AR top-k、BOS、use_k_tokens或OAT latent长度。输出仍保持现有`action`、`action_pred`字典合同，便于控制端接入。

## 10. Self-past 与执行确认

两变体保留self-past：p最大0.5、warmup1000成功updates、ramp4000。先完成previous-window生成，再构建当前训练图；默认8步Euler、每卡分块2，测量后可提高到4。

生成动作必须先反归一化到原始7维命令，再替换历史。构建下一个context时只按正常路径归一化一次；不能把已归一化生成值当成raw action再归一化。重新计算命令差分，validity仍来自窗口metadata。

`prev_window_valid=False`先筛除；previous状态mask全无效不能进入gate encoder。Teacher与student共享同一次self-past结果。实测state history保持实际观测，不用生成动作伪造未来状态。

在线`predict_action()`只返回预测并建立pending，不自动将预测前缀写入已执行历史。`record_executed_actions(actions, executed_lengths)`记录真实发送的命令；处理部分执行、零执行、不同环境长度。

显式past_actions/validity调用为stateless，不改变在线buffer/pending。没有显式历史时使用[B,7,7]命令buffer与bool[B,7]有效性。真实零命令也有效。Reset清空buffer、validity、pending与缓存；batch size变化要求reset。

## 11. Dataset metadata、padding 与验证

### 11.1 数据集选择

用户确认保留两个历史扩展，因此沿用WithPrevWindow数据结构，gate使用WithStateHistory；不切换为完全无历史的标准ManiFlow数据集。

未来新增`action_flow_dataset.py` mixin/subclasses，分别包装现有LIBERO/真机的PrevWindow与StateHistory数据集。保持原动作bytes、窗口数、padding、split和train-only normalizer规则，仅补充future validity和稳定sample identity。

```python
result = super().__getitem__(idx)
buffer_start, _, sample_start, sample_end = self.seq_sampler.indices[idx]
positions = self.pad_before + np.arange(self.n_action_steps)
result['future_action_valid'] = torch.as_tensor(
    (positions >= sample_start) & (positions < sample_end), dtype=torch.bool,
)
result['sample_id'] = torch.as_tensor(
    buffer_start + self.pad_before - sample_start, dtype=torch.int64,
)
```

当前PrevWindow/StateHistory的action_start等于pad_before；该关系必须有测试。Sample ID结合dataset fingerprint识别稳定物理窗口，不能用子集内idx、rank或batch序号替代。

### 11.2 未来padding的固定选择

沿用原dataset的末端动作重复padding，**完整16×7 padded chunk参与FM与CT监督**，与公开ManiFlow全horizon均值目标接近。所有16个位置训练和推理时都有定义。

`future_action_valid[B,16]`仅用于真实物理动作指标，不输入context/AdaLN/self-attention。初版不采用“只mask loss而保留无监督尾部参与双向attention”的额外设计，也不要求在线获得未来episode结束边界。

过去动作使用zero padding和独立validity，不能与未来目标的edge padding混淆。保持episode尾部训练样本，不为方便flow而删除最后15个锚点。

### 11.3 必需字段与指标

公共batch包括obs、action[B,16,7]、past_action[B,7,7]、past_action_valid[B,7]、prev_obs、prev_past_action、prev_past_action_valid、prev_window_valid、future_action_valid与sample_id。仅gate增加当前/previous的8-state历史与validity。

报告：

- 固定noise/time的validation FM loss。
- Expert/generated history下的normalized-action MSE。
- 反归一化后的平移、旋转增量、夹爪误差，注明各分量单位。
- 生成动作超出训练归一化范围/控制端既有限制的比例；若有后处理，分别记录前后结果。
- 完整predict_action延迟、显存和实际闭环成功率；后者使用独立真实机器人协议。

这里不再报告OAT token CE、FSQ格点距离或codec reconstruction error为本模型的训练指标。

Masked MSE跨rank归约平方误差总和与有效元素数量，再相除；不能平均各batch的局部mean。动作维数为7，denominator为有效step数×7。分组指标使用自己的维数。

验证按不补齐样本的分片执行，避免Accelerate even_batches重复尾部。用普通EMA（或明确选定的unwrapped raw model）推理，循环内无DDP forward collectives，允许各rankbatch数不同；末尾统一归约。检查sample_id无重复遗漏，也支持某rank零样本。

验证集不能整除world_size×batch_size时仍应与单卡指标一致。Future validity与sample_id均为评估/随机复现metadata，不成为模型可利用的未来信息。

## 12. 数值与冻结边界

| 部分 | 要求 |
|---|---|
| DINO主干 | 冻结、eval、no_grad；父模块train后仍保持eval |
| Resampler/DiT-X/状态/历史/gate | Student反传，BF16 autocast |
| EMA全部路径 | eval、no_grad，不进入student optimizer |
| Action normalizer | Fresh只拟合训练真实帧，随后冻结；输入/输出变换FP32 |
| 时间、noise、插值、CT target、Euler、loss | FP32 |
| Norm累计与旋转几何 | 必要计算FP32 |
| 执行输出 | 反归一化后的原始单位连续动作，先验证finite |

不把整个observation encoder放入no_grad，否则Resampler无法训练。无效历史在normalization/geometry前清理，rot6d padding使用现有identity规则；memory在线性层/attention前再次清理。

不使用张量最大值猜测RGB范围；两相机输入uint8，DINO预处理一次。验证、推理和self-past采用确定性图像处理，训练的随机光度增强由student/teacher共享。

所有Parameter按对象身份去重进入optimizer。基础变体没有额外history/gate参数，teacher不属于student.module tree，normalizer固定参数和DINO不进入Adam状态。

## 13. 双 RTX 4090 训练配方

| 项目 | 工程初值 |
|---|---|
| Optimizer | AdamW，betas=(0.9,0.95) |
| DiT-X/条件/历史/gate LR | 5e-5 |
| 投影/Resampler LR | 1e-4 |
| Weight decay | 0.01，norm/bias为0 |
| LR调度 | cosine；计划成功updates前5% warmup |
| Clip norm | 1.0 |
| Precision | BF16网络 + FP32 flow算术 |
| Dropout | 本工程显式0.0 |
| 每卡microbatch | 4 |
| Ranks / accumulation | 2 / 8 |
| 有效batch | 64 |
| 验证每卡batch | 4，允许尾批 |
| Self-past chunk size | 2 |
| Activation checkpointing | 开启DiT-X与Resampler的无cache训练前向 |
| EMA | 必需，见§8 |
| 真机epoch上限 | 2001，沿用当前任务配方；用户覆盖优先 |
| Seed | 42 |

保持16×768/12heads，预计约**260M可训练参数**，包含每层9d AdaLN投影、时间编码、视觉适配器等；最终实例化分别报告两个variant的精确总量与可训练/冻结量。

FP32权重、梯度、Adam两矩与EMA基础状态约`260M×20 bytes≈4.84 GiB/卡`。另外还有DINO及其EMA副本、BF16 casts、DDP buckets、激活、CT teacher/self-past临时张量和SDPA workspace。这里没有OAT权重，也不能把4.84 GiB当作峰值。

相较8-slot latent flow，这版处理16个动作位置，query相关计算与激活有所增加；序列仍较短，但不能直接沿用前一版的实测结果。两张24GB RTX4090预计可行，当前没有本模型峰值实测。

DDP每张卡保留完整模型、optimizer和EMA，不形成一个48GB显存池。初始`4×2×8=64`；通过真实CT/self-past峰值测试后可提高到`8×2×4=64`。默认FM/CT切分要求每卡batch至少4，不在未调整loss配方时直接降到1/2。

测量必须覆盖Adam状态建立后的更新、self-past最大概率、CT teacher、expert/generated validation、8步完整推理、两步部署候选和双rank通信。记录allocated/reserved峰值、同步GPU计时、p50/p95、精度、solver步数与输入规格。

两个variant分别使用双卡先后训练；设备启动前重新检查，不假定此前空闲的GPU仍可用。建议正式长跑保留约2 GiB reserved余量。本次不启动GPU probe或训练。

## 14. 模块组织与复用范围

拟用family：`continuous_action_flow`。新variant为`p2n_action_flow`与`p2n_state_gate_action_flow`，与AR和OAT latent flow分开。

| 拟新增文件 | 职责 |
|---|---|
| `oat/model/flow/ditx_action.py` | 原式LN/GELU核心、16-position连续动作速度网络 |
| `oat/model/flow/consistency_flow.py` | 可与latent方案共享的时间采样、FM/CT数学和FP32归约 |
| `oat/model/flow/euler_sampler.py` | 可共享的连续Euler核心，不硬编码OAT或FSQ后处理 |
| `oat/common/action_flow_batch.py` | PreparedActionFlowBatch与数据合同 |
| `oat/policy/p2n_action_flow_common.py` | Normalizer、context、loss、采样和self-past公共策略 |
| `oat/policy/p2n_action_flow.py` | `P2NActionFlowPolicy`，动作历史版 |
| `oat/policy/p2n_state_gate_action_flow.py` | `P2NStateGateActionFlowPolicy`，状态历史门控版 |
| `oat/dataset/action_flow_dataset.py` | 四类数据集适配器，增加future validity和sample_id |
| `oat/workspace/train_p2n_action_flow.py` | Direct-action训练、EMA、指标与恢复 |
| `scripts/train_p2n_action_flow.py` | 配置映射、数据/normalizer预检、双卡启动 |
| `train_p2n_action_flow.sh` | 不覆盖超参数的薄launcher |
| `tests/test_p2n_action_flow_*.py` | 数学、历史、normalizer、时间对齐和artifact合同 |
| `tests/p2n_action_flow_ddp_smoke.py` | 双rank训练与恢复验收 |

本次不创建这些代码文件。若latent-flow后续先实现，共享loss/sampler/helpers应保持family无关；不要为了复用而让direct policy继承一个强制构造OAT/FSQ的父类。

可复用当前已有的[DINO encoder](../oat/perception/dinov3_patch_encoder.py)、[Resampler](../oat/perception/visual_resampler.py)、[token observation encoder](../oat/perception/token_obs_encoder.py)、[ContextBatch](../oat/model/common/context_batch.py)、[历史与执行helper](../oat/policy/p2n_new_common.py)、[state gate](../oat/policy/p2n_state_gate_new.py)、[训练workspace基础](../oat/workspace/train_p2n_new.py)。

提取公共context/执行helper时保留现有AR行为，不先构造AR decoder再遗留unused参数。当前ContextBatch variant校验需将明确family/variant映射为plain/state_gate布局；禁止假报AR variant绕过检查。

Teacher生命周期和一次DDP forward协议与latent方案共享，但direct target preparation只做action normalize；不调用encode_actions/snap_codes/decode_grid_codes。

## 15. 四配置与未来启动命令

| 任务 | Variant | 拟新增配置，相对oat/config |
|---|---|---|
| LIBERO | p2n_action_flow | `train_p2n_action_flow.yaml` |
| LIBERO | p2n_state_gate_action_flow | `train_p2n_state_gate_action_flow.yaml` |
| 真机 | p2n_action_flow | `experimental/train_p2n_action_flow_real_robot.yaml` |
| 真机 | p2n_state_gate_action_flow | `experimental/train_p2n_state_gate_action_flow_real_robot.yaml` |

Dataset targets在`oat.dataset.action_flow_dataset`中分别为`ActionFlowZarrDatasetWithPrevWindow`、`ActionFlowZarrDatasetWithStateHistory`、`ActionFlowRealRobotZarrDatasetWithPrevWindow`、`ActionFlowRealRobotZarrDatasetWithStateHistory`，组合已有对应dataset与metadata mixin。

LIBERO使用匹配的执行确认/状态历史runner；真机runner=null。Task的action_dim、horizon、单位与状态schema来自数据和明确配置，不能再从tokenizer读取。

核心schema示意，**以下字段尚待实现，不保证当前可执行**：

```yaml
policy_family: continuous_action_flow
variant: p2n_action_flow
task_type: real_robot

policy:
  _target_: oat.policy.p2n_action_flow.P2NActionFlowPolicy
  variant: ${variant}
  construction_mode: fresh
  action_dim: 7
  horizon: 16
  n_action_steps: 8
  n_obs_steps: 2
  past_n: 7
  embed_dim: 768
  n_layers: 16
  n_heads: 12
  num_visual_queries: 64
  resampler_depth: 2
  resampler_ffn_type: swiglu
  resampler_ffn_dim: 2048
  dropout: 0.0
  activation_checkpointing: true
  flow:
    space: normalized_actions
    ffn_type: gelu
    ffn_hidden_dim: 3072
    self_qk_norm: false
    cross_qk_norm: layernorm
    qkv_bias: true
    time_embed_dim: 128
    global_condition: time_and_step_only
    pre_norm_modality: false
    fm_fraction: 0.75
    ct_weight: 1.0
    fm_beta: [1.0, 1.5]
    fm_time_scale: 0.999
    ct_time_bins: 10
    teacher_dt_mode: same_relative_dt
    loss_padding: supervise_edge_repeated_targets
    solver: euler
    inference_steps: 8
    self_past_steps: 8
    output_transform: action_unnormalize
  self_past_p: 0.5
  self_past_warmup_steps: 1000
  self_past_ramp_steps: 4000
  self_past_chunk_size: 2

normalization:
  source: training_replay_frames
  mode: limits
  refit_on_resume: false

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

Gate配置另加8-state/4-summary/history/gate字段；基础配置拒绝额外gate模块参数。两种配置均不包含tokenizer checkpoint、FSQ levels、codebook/BOS或OAT latent horizon。

拟定命令接口：

```bash
/venv/real_robot/bin/python scripts/train_p2n_action_flow.py \
  --variant p2n_action_flow --task real_robot \
  --gpus 2,3 --dino /path/to/dinov3_s_snapshot \
  --output output/training/nut_washer_p2n_action_flow_seed42 \
  -- dataloader.batch_size=4 training.gradient_accumulate_every=8

/venv/real_robot/bin/python scripts/train_p2n_action_flow.py \
  --variant p2n_state_gate_action_flow --task real_robot \
  --gpus 2,3 --dino /path/to/dinov3_s_snapshot \
  --output output/training/nut_washer_p2n_state_gate_action_flow_seed42 \
  -- dataloader.batch_size=4 training.gradient_accumulate_every=8
```

示例GPU须重新确认空闲。命令不带tokenizer；若误传OAT/latent-flow专用配置，应报出family/schema错误，不静默忽略。

Preflight检查本地DINO、数据schema、normalizer训练来源、split、batch整除性、EMA设置、variant/target/runner和输出目录。Dry-run不得下载模型或启动训练。用户的batch/epoch/checkpoint/device覆盖原样透传。

## 16. Artifact 与离线恢复

训练artifact至少包含：

- family=`continuous_action_flow`、variant、任务与artifact version。
- 完整DINO结构、processor、固定revision及权重。
- 可训练网络与EMA权重、所有normalizer及其训练episode来源。
- 动作schema：单位、控制坐标系、rotvec delta、夹爪命令语义、horizon与执行锚点。
- Context布局、相机/帧/状态顺序、history/gate/rot6d参数。
- DiT-X的LN/GELU/crossQK类型、时间embedding、初始化与dropout。
- FM/CT采样、相对dt、配比/权重、未来padding监督约定。
- Solver、采样步数、self-past步数、noise seed/sample identity规则。
- Optimizer、scheduler、成功update、self-past、EMA schedule和epoch状态。
- 各rank全局RNG与所有独立torch.Generator状态。
- Dataset identity、train/validation episode IDs与相关软件/代码版本。

不包含OAT网络、FSQ grid或tokenizer外部路径。Fresh training拟合训练normalizer；resume恢复已有统计、不可重拟合；部署仅依赖自包含artifact，不需要数据集或OAT文件。

DINO从保存的config构造后strict load，不在恢复构造时先联网或要求原始pretrained目录存在。Teacher不作为student子模块序列化；训练artifact分别保存student与EMA，部署包可只包含选定EMA推理权重与完整必要冻结模块/统计。

AR、latent flow、direct flow以及plain/gate之间拒绝resume；不得用strict=False隐藏缺失的OAT或不同输入/输出头。跨family迁移初始化不属于当前默认流程。

Online history/pending、KV cache与PreparedBatch不跨episode恢复；部署加载后eval/reset。保存的EMA不能在resume后被student重新覆盖。

## 17. 实施顺序

| 阶段 | 工作 | 出口 |
|---|---|---|
| A | Action schema、normalizer、锚点与future metadata | normalize round trip、train-only拟合、时间对齐通过 |
| B | 共享DINO/Resampler与两种context | plain267/gate271正确，不含OAT或unused AR |
| C | DiT-X原式核心与Euler sampler | 16×7速度/采样，LN/GELU/时间调制与cache合同通过 |
| D | FM与EMA CT | 时间采样/符号/边界、目标与packed loss参考一致 |
| E | 双rank EMA与训练工作区 | Fresh同步、resume保留EMA、累积/计数/梯度通过 |
| F | Self-past、gate与执行确认 | 原始命令历史、无摘要旁路、实际执行状态正确 |
| G | 四配置、launcher与artifact | preflight、用户覆盖、离线加载及family拒绝通过 |
| H | 两variant完整双4090 smoke | 真实DINO、16×768、CT/self-past/Adam峰值和生成通过 |
| I | Nut_washer正式训练与评估 | 两套checkpoint、固定split与明确评估协议 |
| J | 两步部署验收 | 任务质量与延迟达标后保存明确2-step配置 |

FM独立正确性检查不等于额外训练一组长期基线；正式目标为FM+CT。无详细消融要求。若2-step质量不足，保留验收通过的8-step模型，不以少步数本身替代性能。

## 18. 必要验证矩阵

| 验证 | 要求 |
|---|---|
| 无OAT依赖 | Fresh/推理/恢复可在OAT checkpoint不可见时运行，无codec/FSQ参数 |
| 动作normalizer | 只使用训练真实帧；validation view不重拟合；normalize/unnormalize一次且可逆 |
| 动作锚点 | obs结束t，target开始t，执行pred[:8]，无额外To−1偏移 |
| 当前状态语义 | rot6d观测与3维rotvec动作区分，夹爪状态/命令单位不混用 |
| DiT-X核心 | GELU3072、block无affineLN、crossQK LayerNorm、self无QKnorm |
| AdaLN来源 | 仅t/dt，无summary旁路；零初始化允许首轮零梯度 |
| Flow方向 | t=0噪声、t=1动作；已知常速场积分到已知端点 |
| FM/CT目标 | 同一noise/target，teacher stop-grad，t_next/dt边界与FP32正确 |
| 3:1配比 | B=4含3FM/1CT，无漏样本，两个mean等权 |
| Full chunk监督 | edge-repeated target的全部16×7参与loss，无未来边界输入 |
| 连续输出 | 无FSQ/snap/逐步clamp/隐式gripper threshold，仅一次反归一化 |
| 条件模式 | Student/teacher共享augmentation与history，各用自己的adapter |
| EMA/DDP | Fresh从同步student初始化；teacher eval/noGrad/无DDP；成功update后更新 |
| Resume EMA/RNG | 恢复EMA差异与schedule、专用generator，不复制student覆盖 |
| 基础变体 | 不要求state history，无额外encoder/gate参数 |
| Gate变体 | 8状态/7命令validity对齐、rot6d FP32、closed gate不泄漏summary |
| Mask数值 | 无效/closed内容不影响输出，不发生−inf×0或masked NaN传播 |
| 缓存 | 静态cross-KV结果与重算一致，self-KV不跨步，新观测清缓存 |
| Self-past | 生成动作反归一化后写历史，后续正常归一化，无重复缩放 |
| 执行确认 | 零/部分/不同长度/reset/stateless调用正确，记录实际发送命令 |
| Future mask与ID | 起/中/末/短episode、相邻episode正确，last anchor仅一项future有效 |
| 原始数据不变 | 包装dataset的action bytes、padding、窗口数、split保持 |
| 分布式指标 | 尾部不重复；不能整除world×batch/空rank也与单卡一致 |
| Family隔离 | AR/latent/direct、plain/gate错误resume被拒绝 |
| 离线部署 | 原DINO目录和数据集不可见、禁网情况下strict restore成功 |
| 完整模型 | 两variant真实DINO、完整16×768、CT/self-past BF16双卡运行 |
| 部署步数 | 8/2步分别测完整延迟与任务质量，不承诺未经测量的倍数加速 |

检查聚焦数学、数据、梯度与状态合同。没有因纯文档新增而执行训练测试；未来实现后再运行对应验收。

## 19. 交付清单与结果边界

- [ ] 两个独立直接动作flow策略，明确是ManiFlow的历史条件扩展。
- [ ] 16×7生成空间、训练集normalizer和正确执行锚点。
- [ ] 不依赖OAT/FSQ的训练、推理和部署路径。
- [ ] DINOv3-S/16空间features、Resampler与267/271 context。
- [ ] LN/GELU/AdaLN-Zero DiT-X、t/dt-only全局条件和正确cross mask。
- [ ] 联合FM+CT、全EMA conditioner、单次student DDP forward。
- [ ] Self-past、state gate、validity与执行确认。
- [ ] 四配置、独立launcher、数据与normalizer preflight。
- [ ] 自包含artifact、完整恢复、stable IDs与RNG延续。
- [ ] 真实双4090峰值和完整预测延迟。
- [ ] 固定数据划分下两variant的动作指标与闭环结果。
- [ ] 通过验收的部署步数、控制端后处理约定和已知限制。

没有量化环节不代表自动更准确；直接flow要自己学习轨迹结构，77条示范上的泛化需要实测。更大模型、较低FM loss或较少采样步数都不能单独证明闭环更强。

本文件是新增计划。现有AR和冻结OAT latent-flow方案均保持原样，用户选择的tokenizer仍适用于那两条路线。

## 20. 来源与适配说明

- [S1：ManiFlow论文](https://arxiv.org/html/2509.01819v1)：连续动作flow、consistency与少步生成思路。
- [S2：官方DiT-X block](https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/model/diffusion/ditx_block.py)与[模型](https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/model/diffusion/ditx.py)：LN、GELU、QK、AdaLN、时间编码与输出结构。
- [S3：官方2D配置](https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/config/maniflow_image_timm_policy_robotwin.yaml)：实际qk_norm/qkv_bias/time embedding配置；构造默认值不等于使用配置。
- [S4：官方时间采样](https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/model/common/sample_util.py)与[image policy](https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/policy/maniflow_image_policy.py)：FM/CT与relative-dt约定。
- [S5：官方EMA](https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/model/diffusion/ema_model.py)与[workspace](https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/workspace/train_maniflow_robotwin_workspace.py)：EMA schedule；本项目按成功optimizer update更新。
- [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)与[activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)。
- [DINOv3官方仓库](https://github.com/facebookresearch/dinov3)与[选定模型](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)。
- 本地依据：[动作与状态normalizer](../oat/dataset/real_robot_dataset.py)、[previous-window切片](../oat/dataset/zarr_dataset_with_prev_window.py)、[sequence sampler](../oat/common/seq_sampler.py)、[ContextBatch](../oat/model/common/context_batch.py)。

实施开始时固定外部参考代码commit并写入元数据。直接动作目标和DiT-X核心沿用ManiFlow；DINO/Resampler、16层容量、两种历史扩展、全EMA conditioner、零dropout、执行确认与具体GPU配方为本仓库的明确选择。

[S1]: https://arxiv.org/html/2509.01819v1
[S2]: https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/model/diffusion/ditx.py
[S3]: https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/config/maniflow_image_timm_policy_robotwin.yaml
[S4]: https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/policy/maniflow_image_policy.py
[S5]: https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/workspace/train_maniflow_robotwin_workspace.py
