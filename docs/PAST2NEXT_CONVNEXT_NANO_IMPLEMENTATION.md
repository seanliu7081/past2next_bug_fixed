# Past2Next：ConvNeXt V2-Nano + 窄 Resampler 实现计划

日期：2026-10-01

目标仓库：`/workspace/ysk/past2next_bug_fixed`

状态：**待实现。本次仅新增本 Markdown，不修改代码或配置、不下载权重、不启动训练。**

关联文档：[DINOv3-S/16 原始实现规范](PAST2NEXT_DINOV3_SMALL_IMPLEMENTATION.md)、[当前 P2N New 使用说明](P2N_NEW_USAGE.md)。本文是视觉编码器替换方案，不替代原 DINO 文档。

## 1. 目标与范围

用户反馈的主要问题是训练速度慢。采用冻结的预训练 ConvNeXt V2-Nano，加双尺度特征融合和 256 维 Resampler，降低观测编码开销；现有 modern AR 动作 backbone 保持架构与超参数，继续正常训练。

本次正式覆盖两个真机策略：

| Variant | 策略类 | 历史条件 |
|---|---|---|
| `p2n_new` | `P2NNewPolicy` | 动作历史、差分条件、self-past |
| `p2n_state_gate_new` | `P2NStateGateNewPolicy` | 上述条件，加实测状态历史与 summary gate |

两个 variant 共用新观测编码器。保留原 DINO 配置和旧 artifact 的加载兼容性。新增的是视觉 backend，不能通过改用旧版策略类实现替换。

本方案不改成 latent flow 或直接 action flow。self-past 的样本选择与生成分块优化列为独立后续项目，不混入本版视觉替换的默认实现和性能归因。

## 2. 必须保持的策略合同

| 项目 | 固定要求 |
|---|---|
| 动作网络 | 现有 `ModernAutoregressiveModel` |
| AR 层数 / 宽度 / heads | 16 / 768 / 12 |
| AR FFN | SwiGLU，intermediate=2048 |
| AR attention / normalization | 现有 causal attention、RMSNorm、QK-Norm、cache 语义 |
| 动作位置编码 | 沿用当前实现，不新增 RoPE |
| 训练目标 | OAT token 的 teacher-forcing cross entropy |
| OAT | 原指定 checkpoint 的冻结 EMA 权重和自身 action normalizer |
| 动作 token 数 | 当前指定 OAT 为 8；构造时验证实际 latent horizon |
| 输出 / 执行 | 16×7 动作，前8步执行，保留执行确认协议 |
| 观测 | 两相机、两帧 |
| 动作历史 | 过去7步，保留 validity 与差分条件 |
| Gate 历史 | 8步实测状态、4个 summaries、现有门控公式与 rot6d rows |
| 每图视觉 tokens | 64 |
| 编码器输出 | `visual[B,256,768]`、`proprio[B,2,768]` |
| 完整 context | 基础版267 tokens，gate版271 tokens |

继续使用用户选定的 tokenizer：

```text
/workspace/ysk/past2next_bug_fixed/output/training/nut_washer_v3_N77_gated_so3aug_20260924_081008_317043427/tokenizer/checkpoints/ep-1540_mse-0.000.ckpt
```

该路径是已选训练来源，本次不重新加载或验证其可用性。Fresh preflight 继续检查文件、EMA、动作 schema 和数据来源；完整 policy artifact 恢复不依赖原 tokenizer 路径。

“保留 backbone”表示保留动作网络架构，不表示冻结 AR，也不表示旧 DINO policy checkpoint 可以直接 resume 为新视觉模型。

## 3. 总体数据流与张量形状

```text
每相机原始 RGB [B,2,H,W,3]
→ 按 batch → frame → camera 合并为 [4B,H,W,3]
→ RGB预处理、完整视野 resize 到224×224
→ 冻结 ConvNeXt V2-Nano，取第3/第4个 stage
   ├─ F16 [4B,320,14,14]
   └─ F32 [4B,640,7,7]
→ 双尺度可训练投影与融合
→ [4B,196,256] 空间 tokens
→ 256维、2层、64-query Resampler
→ [4B,64,256]
→ Linear(256,768)
→ 加入768维 camera/frame embeddings
→ visual [B,256,768]

已归一化当前状态
→ 原状态拼接与768维投影
→ proprio [B,2,768]

visual + proprio + 原动作/差分条件 [+ 原状态历史 summaries]
→ 原 modern AR Transformer
→ 原冻结 OAT decode
```

保持 batch/frame/camera 的布局和展平顺序；不能只让 shape 相同而交换相机或时间含义。

## 4. ConvNeXt 主干与图像预处理

### 4.1 模型与权重

固定模型：`convnextv2_nano.fcmae_ft_in22k_in1k`。该权重来自 FCMAE 预训练及 ImageNet-22K、ImageNet-1K 微调。官方模型卡列出的完整模型约15.6M参数；特征提取实现不保留分类头，实际参数量单独报告。[S1]

- 使用 `timm` 的特征提取接口，输出 stage indices `(2,3)`，对应 stride16/32。
- 固定实际采用的 timm 版本、权重 revision、文件 SHA256 和模型构造参数。
- Fresh 仅从明确的本地权重来源加载，验证所有主干特征参数已加载；缺少权重时失败，不能退化为随机冻结主干。
- 分类头移除和权重 key 映射必须明确校验；不能用无约束的 `strict=False` 掩盖特征层缺失。
- 全部相机和帧共享一个主干，不为相机重复创建模型。
- 初版不解冻主干、不增加 LoRA、不做额外视觉预训练。

### 4.2 预处理

- 接收任务 schema 中的原始 RGB `uint8`，当前真机图像为128×128。
- 转为 FP32 `[0,1]` 后完整视野 resize 至224×224；使用固定 bicubic、antialias 设置，并保存到 artifact。
- 使用选定权重的 mean/std，仅做一次图像归一化。
- 不沿用旧 ResNet 的 RGB normalizer，不自动调用分类评估的 resize/center-crop 组合改变视野。
- 保留现有 brightness=0.1、contrast=0.1 光度增强语义。
- 不新增几何翻转或旋转。
- 验证、部署和 self-past 采用确定性图像处理。
- 图像范围由 schema/config 决定，不能用当前 batch 最大值猜测。

## 5. 双尺度融合与窄 Resampler

### 5.1 固定融合公式

```text
P16 = Conv1x1_320_to_256(F16)                    # [N,256,14,14]
P32 = Conv1x1_640_to_256(F32)                    # [N,256,7,7]
P32 = bilinear_resize(P32, 14, 14, align_corners=False)
F   = RMSNorm_channels((P16 + P32) / sqrt(2))    # [N,256,14,14]
M   = flatten_spatial(F)                        # [N,196,256]
```

`RMSNorm_channels` 在每个空间位置跨256个通道归一化，不能跨空间位置计算。两条投影路径都参与训练；融合保留14×14空间网格，不先全局池化成单个向量。[S1]

### 5.2 Resampler 配方

| 字段 | 值 |
|---|---|
| 内部宽度 | 256 |
| 层数 | 2 |
| Heads / head dim | 4 / 64 |
| FFN | SwiGLU，intermediate=768 |
| Queries | 64 / 图 |
| 空间位置编码 | 原 `VisualResampler` 的可学习行/列位置编码，grid_size=14 |
| Dropout | 0.1，沿用原视觉适配器设置 |
| 输出投影 | Linear(256,768) |
| 相机/帧 embeddings | 在输出投影后加入，保持768维 |
| 状态投影 | 保持768维，保留名称 `state_projection` |

复用 [VisualResampler](../oat/perception/visual_resampler.py)，用新的独立视觉参数实例化。不能将 policy 的 `embed_dim`、`n_heads` 或 `ffn_dim` 改小来实现窄 Resampler。

按现有 block 结构估算，窄 Resampler 本体约2.25M参数；包含双尺度投影、融合归一化和256→768输出投影约2.7M。当前视觉投影与宽 Resampler 合计约19.3M。最终应报告实际参数统计，不把这项缩减当作整个策略的缩减比例或训练提速倍数。

## 6. 冻结、梯度与数值边界

```text
with no_grad():
    F16, F32 = frozen_backbone(preprocess(rgb))

# 下列计算必须保留训练图
M = trainable_fusion(F16, F32)
V = trainable_resampler(M)
V = trainable_output_projection(V)
```

- 只冻结 ConvNeXt 主干和既有 OAT，不冻结整个 observation encoder。
- 网络前向沿用 BF16 autocast；像素预处理与必要归一化归约使用 FP32。
- 不使用永久 `.half()` 改写原有权重保存约定。
- 新模块初始化不能递归重置预训练 ConvNeXt。
- `policy.train()`、EMA模型的 `.train()`、self-past rollout退出后，冻结主干必须仍为 eval。
- 保留现有 self-past 的 autocast cache 清理和退出 inference mode 后 clone 的处理。
- 训练主干特征提取采用 `no_grad()`；若采用 inference mode，则返回张量必须在退出该模式后转换成可被可训练投影保存用于反向的普通张量。
- Student 与 EMA 必须有相同结构；EMA更新和step计数沿用原工作区成功optimizer update的逻辑。

## 7. 公共接口与构造/恢复机制

新增 `ConvNeXtTokenObservationEncoder`，提供与原 encoder 一致的外部接口：

- `shape_meta`、`n_obs_steps`、`rgb_ports`、`state_ports`。
- `output_feature_dim() == 768`。
- `num_visual_tokens == 256`，`num_queries == 64`。
- `resampler.num_queries == 64`：现有 gate 直接使用此接口。
- `forward(obs) -> (visual, proprio)`。
- `export_config()` 与明确的冻结主干模式维护方法。

构造工厂只接受明确白名单类型：原 DINO 与新 ConvNeXt tokens。不要忽略 `_target_` 后总构造原 encoder，也不要让 Nano 假冒 `dino_encoder` 才能通过检查。

目前 [公共策略](../oat/policy/p2n_new_common.py) 的构造/恢复、rollout cleanup、策略名称和源码指纹存在 DINO 专用逻辑，实施时统一处理：

1. Fresh 根据 encoder 类型构造对应网络。
2. Restore 根据保存的 encoder 类型与结构构造，禁外部下载，随后 strict load。
3. 旧 artifact 缺少 encoder 类型且包含旧 `dino` 配置时，按原 DINO schema解释，保持既有state_dict key。
4. 用通用接口维护冻结主干模式，替换 `.dino_encoder.backbone.eval()` 的硬编码。
5. 保留 gate 的视觉池化、历史编码、summary mask/bias和validity合同。

## 8. 文件级实施清单

| 文件 | 改动 |
|---|---|
| `oat/perception/convnext_feature_encoder.py`（新增） | 本地预训练加载、元数据、预处理、冻结、双尺度输出 |
| `oat/perception/convnext_token_obs_encoder.py`（新增） | 双尺度融合、窄 Resampler、输出投影、camera/frame和state接口 |
| `oat/perception/obs_encoder_factory.py`（新增） | 白名单factory、旧schema兼容、新schema恢复 |
| `oat/perception/visual_resampler.py` | 复用参数化实现；无实际需要则不改逻辑 |
| `oat/perception/token_obs_encoder.py` | 仅在通用冻结模式接口确有需要时补兼容方法，保留DINO行为 |
| `oat/policy/p2n_new_common.py` | 构造/恢复分派、通用rollout cleanup、encoder标识与artifact信息 |
| `oat/policy/p2n_state_gate_new.py` | 核验接口兼容；不改变门控结构和公式 |
| `oat/workspace/train_p2n_new.py` | resume encoder合同检查、性能与参数报告 |
| `scripts/train_p2n_new.py` | 视觉配置选择、Nano本地权重preflight、CLI兼容 |
| `scripts/smoke_p2n_new.py` | 支持新视觉backend的实模型smoke |
| `tests/test_p2n_new_convnext_*.py`（新增） | 针对视觉路径、恢复、模式和集成合同的检查 |
| 两份真机YAML（新增，见§9） | 分别继承现有基础版与gate版 |

`oat/model/autoregressive/modern_transformer_cache.py` 无需为视觉替换改变算法或架构。ContextBatch的segment语义、mask和执行确认协议继续复用。

## 9. 配置与 launcher

### 9.1 两份新配置

| 拟新增配置，相对 `oat/config` | 继承来源 |
|---|---|
| `experimental/train_p2n_new_convnext_nano_real_robot.yaml` | `experimental/train_p2n_new_real_robot.yaml` |
| `experimental/train_p2n_state_gate_new_convnext_nano_real_robot.yaml` | `experimental/train_p2n_state_gate_new_real_robot.yaml` |

Variant、策略类与workspace保持原名，通过encoder类型区分视觉实现。两份新配置共用同一视觉配方，分别继承各自dataset与history/gate字段；不由“关闭gate”代替基础版。

拟定字段如下，**尚未实现，当前不可直接执行**：

```yaml
policy:
  obs_encoder_type: convnextv2_tokens
  convnext_model_name: convnextv2_nano.fcmae_ft_in22k_in1k
  convnext_path: null
  convnext_revision: null
  convnext_frozen: true
  vision_image_size: 224
  vision_feature_stages: [2, 3]

  visual_resampler_dim: 256
  visual_resampler_heads: 4
  visual_resampler_ffn_dim: 768
  resampler_depth: 2
  num_visual_queries: 64

  embed_dim: 768
  n_layers: 16
  n_heads: 12
  ffn_dim: 2048
  dropout: 0.1
```

`convnext_path`与revision在准备本地权重后提供。继承得到的旧DINO字段可以保留为空，但不能被Nano分支使用；若同时传入冲突的DINO/Nano权重，应明确报错。

保留77 demos、seed42、val_ratio0.05、真机runner=null和现有2001 epoch上限；CLI覆盖优先。不将旧运行中每卡batch128的覆盖写成通用默认，正式batch以当前硬件和同条件smoke确定。

### 9.2 拟定命令接口

在现有launcher增加 `--vision`，默认 `dinov3`；新增选择 `convnext_nano`、本地 `--convnext` 和 `--convnext-revision`。旧 `--dino` / `--dino-revision` 保持兼容。配置映射增加视觉维度，当前Nano正式交付仅要求上述两个真机配置。

拟定使用形式，**仅为未来接口示例**：

```bash
python scripts/train_p2n_new.py \
  --variant p2n_new --task real_robot \
  --vision convnext_nano \
  --convnext /path/to/pinned_convnext_nano_snapshot \
  --convnext-revision PINNED_REVISION \
  --output output/training/p2n_new_convnext_nano_real_robot_seed42 \
  --dry-run

python scripts/train_p2n_new.py \
  --variant p2n_state_gate_new --task real_robot \
  --vision convnext_nano \
  --convnext /path/to/pinned_convnext_nano_snapshot \
  --convnext-revision PINNED_REVISION \
  --output output/training/p2n_state_gate_new_convnext_nano_real_robot_seed42 \
  --dry-run
```

Dry-run只解析配置；preflight检查本地权重、OAT、schema、split和参数合同；GPU设备在实际执行前检查空闲。日志tags、policy name、输出目录及报告写入实际encoder，不能继续将Nano标成dinov3。

## 10. 训练、normalizer与optimizer

- 主干冻结；融合层、Resampler、输出投影与camera/frame embeddings使用 `obs_enc_lr=1e-4`。
- `state_projection` 保留原名称与 `policy_lr=5e-5` 分组；AR和history/gate也沿用policy LR。
- AdamW、weight decay、无decay参数规则、BF16、EMA和梯度累积沿用原训练合同。
- 按Parameter对象身份去重；冻结主干、OAT和固定normalizer不进入optimizer。
- 状态和过去动作继续使用原policy normalizer；OAT继续保留checkpoint内部的action normalizer。不能在视觉替换时改变动作归一化和tokenize语义。
- 保留原数据划分、历史padding、validity、self-past课程和生成温度/top-k。
- 保持双卡DDP，两个variant分别训练；不把两张卡的显存视作一个共享池。
- 初版不默认启用额外checkpointing或更大batch；性能比较时与DINO使用相同设置。

Fresh run从预训练Nano和指定冻结OAT初始化，AR/新增adapter重新初始化。这里不做旧DINO policy的部分权重迁移；若之后需要，定义独立的初始化流程并重置optimizer，不能伪装成resume。

## 11. 自包含 artifact 与恢复保护

新artifact至少记录：

- Encoder类型、schema版本、完整模型构造配置及权重。
- 权重model ID、revision、SHA256、timm版本和实际源码指纹。
- RGB范围、resize/interpolation/antialias、mean/std与augmentation。
- Feature stages、实际通道/grid、双尺度融合公式。
- Resampler维度/heads/FFN/queries/depth、输出投影与embedding布局。
- 原AR、OAT、history/gate、normalizer及执行schema。
- Student/EMA、optimizer、scheduler、成功update、self-past状态和RNG。

恢复流程必须仅依赖artifact中的结构、权重与配置；不先联网下载，也不要求原Nano或tokenizer文件仍存在。Fresh路径的权重覆盖检查与完整artifact的strict load均不能省略。

扩展 [resume检查](../oat/workspace/train_p2n_new.py) 和 [部署override保护](../oat/policy/p2n_new_common.py)：

- 对规范化encoder结构/预处理配置进行比较，而不是只比较variant或文件路径。
- Encoder类型、image size、feature stages、融合配置、Resampler参数不可在resume或加载override中改变。
- DINO↔Nano、基础↔gate之间拒绝resume，并提前给出具体不一致项。
- 旧DINO schema提供明确兼容解释；原DINO state_dict key与既有预测保持一致。
- 模型恢复后重置在线history/pending/cache，保留原有执行协议。

## 12. 性能验证与 self-past 边界

不能仅凭参数量或CNN理论计算量承诺总体训练加速。AR仍为16×768，视觉输出仍为256 tokens，self-past仍包含动作生成。

此前代码审计发现：self-past在概率大于0时先为全部有效previous windows生成，再按概率选用；每卡batch128、chunk4最多形成32次小批量生成。该执行方式会反复运行视觉编码与AR。本次保持其算法行为，单独观察视觉替换的收益。

此前运行的日志与时长属于前次审计记录，日志路径目前不保证可用；本计划不把它们作为新版本的现测基线。实施验收重新记录当前环境、代码版本和完整resolved config。

### 12.1 同条件测量

同一硬件、无其他训练干扰、相同数据/seed、每卡batch、world size、累积、BF16、checkpointing、帧数、query数、self-past chunk和温度/top-k。

分别短测：

1. `p=0`：正常训练前向/反向/更新。
2. 固定 `p=0.5`：达到目标概率后的完整训练步，包括self-past。
3. 完整 `predict_action` 与generated-history验证。

短测中固定self-past概率应通过明确的benchmark设置完成，不修改正式课程或用warmup-only结果代替后期速度。无需额外长期消融训练。

计时包含warmup并覆盖Adam状态建立之后的稳定步骤；GPU异步操作用CUDA events或明确同步处理。分段profiling和端到端吞吐分开运行，避免大量同步改变正常吞吐。

报告视觉主干、融合/Resampler、self-past、正常训练步骤、数据等待、端到端step/epoch与allocated/reserved峰值。记录批量、有效batch、测量区间与p50/p95。

### 12.2 后续独立优化

若self-past仍占主要时间，可另行实施“先选择实际需要替换的样本再生成”和更大的生成分块。这些改动不需要缩小AR，但需要验证validity、采样分布与RNG恢复，并独立报告收益；本版不默认实施。

## 13. 实施顺序与出口

| 阶段 | 工作 | 验收出口 |
|---|---|---|
| A | 固定权重来源、timm版本与视觉配置schema | 本地权重覆盖完整，缺失/冲突输入清晰报错 |
| B | 冻结feature encoder与双尺度融合 | 特征shape、预处理与梯度边界正确 |
| C | 窄Resampler、768输出及公共接口 | 两个策略均保持256视觉tokens和267/271 context |
| D | Factory、rollout模式、optimizer/EMA接入 | 冻结模式、参数组与self-past前后反传正确 |
| E | 两份配置、launcher、artifact和resume | Dry-run/preflight、离线strict restore与旧DINO兼容通过 |
| F | 两variant双卡短测与同条件性能测量 | 无DDP/数值错误，得到实测速度与显存报告 |
| G | 固定split的正式训练与任务验收 | Token loss、动作误差和独立闭环结果可比较 |

只有视觉路径和完整训练步实测后，才能给出本版本加速结论。离线loss或更快训练不直接代表闭环成功率更高。

## 14. 必要检查清单

- [ ] 两个variant都使用原modern AR结构、原OAT、原执行协议。
- [ ] F16/F32、融合grid、相机/帧顺序及最终visual/proprio形状正确。
- [ ] Nano/OAT无梯度；两尺度投影、Resampler、输出投影和AR有梯度。
- [ ] policy/EMA切换train、self-past退出后，冻结CNN仍eval。
- [ ] Gate仍使用64 queries接口，state validity和summary mask/bias不变。
- [ ] 状态投影保持policy LR，所有可训练视觉参数进入obs LR且无重复。
- [ ] OAT内部normalizer、policy normalizer和RGB预处理边界正确。
- [ ] 保存/恢复后确定性预测一致，恢复无需原预训练目录或网络。
- [ ] 错误encoder/结构的resume提前拒绝，旧DINO artifact继续可用。
- [ ] 两份真机配置、用户覆盖、日志标识和preflight正确。
- [ ] 两variant双卡训练、EMA、累积和checkpoint续训通过。
- [ ] 完整记录p=0与p=0.5的同条件速度、显存和生成指标。

## 15. 来源

- [S1：ConvNeXt V2-Nano 权重与特征尺寸](https://huggingface.co/timm/convnextv2_nano.fcmae_ft_in22k_in1k)。
- [ConvNeXt V2 官方仓库](https://github.com/facebookresearch/ConvNeXt-V2)。
- [当前观测编码器](../oat/perception/token_obs_encoder.py)、[Resampler](../oat/perception/visual_resampler.py)。
- [公共策略](../oat/policy/p2n_new_common.py)、[状态门控策略](../oat/policy/p2n_state_gate_new.py)。
- [训练工作区](../oat/workspace/train_p2n_new.py)、[launcher](../scripts/train_p2n_new.py)。
- [基础版真机配置](../oat/config/experimental/train_p2n_new_real_robot.yaml)、[门控版真机配置](../oat/config/experimental/train_p2n_state_gate_new_real_robot.yaml)。

[S1]: https://huggingface.co/timm/convnextv2_nano.fcmae_ft_in22k_in1k
