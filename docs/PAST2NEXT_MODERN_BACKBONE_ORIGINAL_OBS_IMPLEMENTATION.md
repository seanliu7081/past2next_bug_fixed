# Past2Next：Modern AR Backbone + 原版观测编码器实现计划

日期：2026-10-04

目标仓库：`/workspace/ysk/past2next_bug_fixed`

状态：**待实现。本次仅新增本 Markdown，不修改模型代码或训练配置、不下载权重、不启动训练。**

关联方案：[DINOv3 + modern AR](PAST2NEXT_DINOV3_SMALL_IMPLEMENTATION.md)、[ConvNeXt Nano + modern AR](PAST2NEXT_CONVNEXT_NANO_IMPLEMENTATION.md)。本方案是独立版本，保留前两条路线。

## 1. 目标与准确范围

保留现有 modern AR 动作 backbone，观测端直接复用原版 Past2Next 的完整 `FusedObservationEncoder`：两路 ResNet-18、SpatialSoftmax、状态编码及每帧特征拼接。仅在原 encoder 输出之后增加接入768维策略的线性投影与帧位置编码。

同一设计覆盖：

| Variant | 新入口拟用类 | 条件 |
|---|---|---|
| `p2n_new` | `P2NNewOriginalObsPolicy` | 原版融合观测 + 过去动作/差分 |
| `p2n_state_gate_new` | `P2NStateGateNewOriginalObsPolicy` | 上述条件 + 实测状态历史摘要及门控 |

Variant字符串沿用当前modern体系，明确区分新的policy target、`obs_encoder_type=original_fused`和context layout。普通版不构造无用gate模块。

“原版观测编码器”指完整融合路径，不是泛指任意ResNet18。已有flow版的每相机一个视觉token适配器不是相同的输出合同，不能直接拿来替代。

本次保留AR、OAT和CE目标，不转为DiT-X/flow。保留modern策略的动作历史、自生成历史课程和执行确认行为；不恢复旧策略中的其他动作建模设计。

## 2. 原版编码器的代码依据

当前 [Nut Washer launcher](../train_nut_washer_v3.sh) 的基础版使用 [train_past2next_scratch_all500.yaml](../oat/config/train_past2next_scratch_all500.yaml)；门控版经 [真机gate配置](../oat/config/experimental/train_past2next_state_history_gate_real_robot.yaml) 继承 [state-history配置](../oat/config/experimental/train_past2next_state_history.yaml)。这两条实际配方都使用112×112 crop。

较早的 `train_past2next_self_past.yaml` 使用76×76 crop，不能把这个旧默认当作当前Nut Washer配方。本计划将112×112明确写入新配置；用户明确指定的其他原版resolved recipe应通过显式覆盖记录。

完整结构来自 [FusedObservationEncoder](../oat/perception/fused_obs_encoder.py)、[RobomimicRgbEncoder](../oat/perception/robomimic_vision_encoder.py)、[ProjectionStateEncoder](../oat/perception/state_encoder.py)：

| 项目 | 原版行为，本方案保留 |
|---|---|
| 图像输入 | 原始128×128 RGB；不放大到224 |
| Crop | 训练随机112×112；验证、推理和self-past中心112×112 |
| Backbone | `ResNet18Conv`，无ImageNet预训练，随机初始化 |
| 归一化层 | 原BatchNorm替换为GroupNorm，沿用现有分组规则 |
| 相机权重 | `share_rgb_model=false`，两相机各自独立；同一相机各帧共享 |
| 空间池化 | SpatialSoftmax，32 keypoints、temperature=1、noise=0 |
| 每相机输出 | flatten后64维，经原VisualCore线性层及ObservationEncoder的ReLU |
| 状态编码 | `ProjectionStateEncoder(out_dim=null)`，归一化后直接拼接，不投影为768 |
| 融合 | 按原顺序拼接各相机特征和状态；每帧一个向量 |
| 训练状态 | ResNet和视觉输出层正常训练；不按DINO/Nano方式冻结 |

固定并记录robomimic/torchvision版本。当前安装的Robomimic中，`ResNet18Conv`默认`pretrained=False`，`ObservationEncoder`默认输出ReLU；构造验收需核对这些实际默认，防止依赖升级改变配方。

## 3. 保留的 modern 动作网络

| 项目 | 固定设置 |
|---|---|
| 实现 | 现有 `ModernAutoregressiveModel` |
| 层数 / 宽度 / heads | 16 / 768 / 12 |
| FFN | SwiGLU，intermediate=2048 |
| Norm / attention | 原RMSNorm、QK-Norm、causal AR和cache实现 |
| 动作位置编码 | 沿用当前learned slot，不新增RoPE |
| Dropout | 0.1，沿用modern配置 |
| OAT | 指定checkpoint的冻结EMA权重与其内部normalizer |
| 动作预测 | 8个OAT tokens → 16×7动作；执行前8步 |
| 目标 | Token teacher-forcing cross entropy |
| 动作历史 | 7步raw commands + 2个差分tokens，保留validity |
| Self-past | p=0.5，warmup1000、ramp4000，按成功optimizer updates推进 |
| Gate历史 | 8步实测状态、4个summary tokens、原门控MLP及summary-only bias |

沿用用户指定的tokenizer训练来源：

```text
/workspace/ysk/past2next_bug_fixed/output/training/nut_washer_v3_N77_gated_so3aug_20260924_081008_317043427/tokenizer/checkpoints/ep-1540_mse-0.000.ckpt
```

本次不重新验证checkpoint文件可用性。未来fresh preflight核对来源、EMA、horizon=16、action_dim=7、latent_horizon=8和codebook；不重训tokenizer。

保留backbone指动作网络结构与超参数；AR仍正常训练。旧DINO/Nano checkpoint不能直接作为本版本的resume。

## 4. 数据流、维度与 token 数

```text
每帧两路RGB
├─ Camera A: 原crop → 可训练ResNet18/GN → SpatialSoftmax → 64维
└─ Camera B: 原crop → 可训练ResNet18/GN → SpatialSoftmax → 64维
                          ↓ concat = 128维视觉特征
原始当前状态 → 原state normalizer → identity state encoder
                          ↓ concat
原 FusedObservationEncoder 输出 [B,2,D_obs]
                          ↓ encoder外的Linear(D_obs,768)
                          ↓ 2帧learned time embedding + fused类型embedding
融合观测tokens O [B,2,768]
                          ↓ concat原modern动作/差分条件 [+ summaries]
现有16×768 modern AR → 冻结OAT decode
```

`D_obs`必须调用原encoder的`output_feature_dim()`获得，不能写死为旧AR的embed_dim=256。

当前Nut Washer真机 [shape_meta](../oat/config/task/policy/real_robot/nut_washer_with_prev_window.yaml) 中，状态包括位置3、rot6d6、夹爪宽度1、task_uid1，共11维。因此：

```text
D_obs = 64 + 64 + 11 = 139
原encoder: [B,2,139]
接入投影: Linear(139,768)
```

Task UID也必须保留；不要漏掉它而写成138。其他任务按实际schema计算，不强行复用139。这里只支持当前RGB+state任务，text/task-residual等其他原版扩展不静默启用。

| Context部分 | 基础版 | Gate版 |
|---|---:|---:|
| 融合观测 | 2 | 2 |
| 过去动作 | 7 | 7 |
| 差分 | 2 | 2 |
| 状态历史摘要 | 0 | 4 |
| **合计** | **11** | **15** |

本版不构造Resampler、不复制观测为64 queries，也不再添加独立的proprio tokens。两路相机已按固定顺序编码在每帧融合向量内；无需另外生成camera tokens或camera embeddings。

相较DINO/Nano的267/271 context，序列长度改变是保留原融合编码方式的直接结果。动作Transformer支持可变memory长度，层数、宽度、heads和动作序列长度保持原样。

## 5. 精确复用与接入 adapter

新增轻量 `OriginalFusedObservationAdapter`，内部持有原 `FusedObservationEncoder`，不重写其视觉和状态计算。

- `fused_encoder`：原始encoder，直接接受原始obs，输出`[B,To,D_obs]`。
- `obs_projection`：在原encoder之后将`D_obs→768`；这是策略接入层，不改变原encoder本体。
- `frame_embedding`：`[To,768]`，保留两帧顺序信息；不能只拼无位置观测导致cross-attention视作无序集合。
- 提供`shape_meta`、`rgb_ports/state_ports`、`n_obs_steps`、`output_feature_dim()==768`和`export_config()`。
- 适配器返回`[B,To,768]`融合tokens，而不是伪造`(visual,proprio)`。
- 记录原视觉输出顺序、state字段顺序和融合顺序；验证形状同时验证内容顺序。
- 原encoder等价性检查在`obs_projection`之前进行。

不建立假的`resampler`、`num_queries`或`num_visual_tokens=256`字段绕过旧检查。报告字段使用`observation_tokens=2`、`context_tokens=11/15`、`fused_feature_dim=139`。

## 6. ContextBatch与gate接入

### 6.1 独立融合观测布局

新增语义明确的 `FUSED_OBSERVATION` segment。保留现有segment整数0–4不变，新增值5；扩展 `ContextBatch.validate()`，将该segment识别为有效当前观测。

新policy使用`context_schema_version=2`、`context_layout=original_fused_v1`；现有DINO/Nano入口继续使用原schema和布局。允许扩展segment枚举不等于允许跨布局resume。

新基础版显式构造：

```text
O    = adapter(raw_obs) + fused_type_embedding       # [B,2,768]
A    = modern_raw_action_features                   # [B,7,768]
Diff = modern_acc_and_jerk_features                 # [B,2,768]
memory = concat(O, A, Diff)                         # [B,11,768]
valid  = concat(current_observation_valid, past_valid, diff_valid)
```

保留modern历史归一化、padding清理、时间编码和差分validity规则。当前两帧沿用原数据集观测padding语义；新增bridge不改变采样锚点。

类型embedding只构造实际使用的四类：fused observation、raw action、acc、jerk。不要遗留未使用的视觉/状态独立投影或参数。新路径必须覆盖完整context构造，不能调用仍要求tuple输出和5行type embedding的旧`build_context()`。

### 6.2 门控版

历史编码器、summary数量、validity、rotation几何、gate MLP和attention bias语义保持modern版本：

```text
observed = mean(O, dim=time)                        # [B,768]
history  = mean(history_summaries, dim=tokens)       # [B,768]
fraction = mean(state_history_valid)                # [B,1]
log_gate = logsigmoid(history_gate(concat(observed, history, fraction)))
```

Gate MLP输入仍为`2×768+1=1537`，hidden=128，初始gate=0.9。观测摘要只来自当前两个融合tokens，不能混入raw/diff/history tokens。

原modern门控版本的`observation_pool`处理独立visual/proprio拼接，在本分支中用无参数Identity代替；不构造无用的1536→768层。这是融合观测所需的条件接入适配，不改变动作Transformer。

`open/closed`诊断模式仍按原规则设置gate值，同时填充完整context元数据。`log_gate`只作用于`HISTORY_SUMMARY`列，普通观测和过去动作不受gate抑制，仍受各自valid_mask约束；关闭gate时历史摘要不能绕过mask进入其他观测条件。

## 7. Normalizer与train/eval边界

原encoder内部负责RGB和当前state归一化。本分支的policy应直接传原始obs，不能沿用DINO/Nano先在policy中归一化state的路径。

| 输入 | 归一化职责 |
|---|---|
| RGB | 原RobomimicRgbEncoder，真机dataset固定byte endpoints映射到[-1,1] |
| 当前state/task_uid | 原ProjectionStateEncoder内部，使用训练集dataset normalizer |
| 过去动作/差分 | 原modern policy action normalizer |
| 实测state/action history | 原history encoder根据原始history和normalizer处理 |
| OAT encode/decode | 指定冻结checkpoint内部normalizer，不能替换 |

`set_normalizer()`既设置policy统计，也调用原`fused_encoder.set_normalizer(dataset_normalizer)`。只在训练episode真实帧拟合统计，随后冻结所有normalizer参数；resume从artifact恢复，不能重新拟合。

当前modern工作区在恢复checkpoint前无条件调用`dataset.get_normalizer()`。新workspace必须明确分流初始化：fresh才调用拟合并下发统计；resume从artifact恢复policy、原encoder内部、OAT及EMA的全部normalizer，不调用该拟合路径。可以提取可覆写的初始化hook，旧DINO/Nano路径保持现有行为；不能仅继承现有run循环并声称已经满足此合同。

Encoder入口预检要求所有state/RGB字段都有统计，避免原ProjectionStateEncoder“缺统计时跳过”的fallback被当作正常训练。不能遗漏RGB统计或把ImageNet mean/std应用到原版encoder。

训练模式必须与冻结DINO/Nano分开：

- 原ResNet、SpatialSoftmax和输出层参与反传，不包`no_grad()`、不强制eval。
- 训练随机crop，验证/self-past/predict_action临时eval并中心crop。
- self-past结束恢复全部原训练模式；OAT始终冻结eval。
- 保留autocast cache清理和inference tensor退出后的clone规则。
- EMA模型结构一致；验证显式eval；normalizer保持冻结。
- 初始化新增bridge不重新初始化原encoder，不改变原随机初始化机制；若比较原encoder等价性，需要相同权重与crop RNG。

## 8. 模块组织与现有代码复用

采用独立入口，复用modern网络、OAT生命周期和训练循环。当前Nano实现也有独立policy/workspace；其mixin假设Resampler存在且主干冻结，不能直接作为本分支父适配器使用。

| 拟新增/调整文件 | 职责 |
|---|---|
| `oat/perception/original_fused_obs_adapter.py`（新增） | 包装原encoder、768接入投影、frame位置、normalizer与导出 |
| `oat/policy/p2n_new_original_obs.py`（新增） | 两个policy类、融合context、gate观测汇总、train/rollout、optimizer/恢复 |
| `oat/workspace/train_p2n_new_original_obs.py`（新增） | 复用modern训练循环，分流fresh/resume normalizer初始化，扩展schema检查和报告 |
| `scripts/train_p2n_new_original_obs.py`（新增） | 两配置映射、dataset/OAT/encoder preflight与双卡启动 |
| `train_p2n_new_original_obs.sh`（可选新增） | 不覆盖用户参数的薄wrapper |
| `oat/model/common/context_batch.py`（兼容扩展） | 追加fused segment和观测有效性识别，不重编号旧值 |
| 两份真机YAML（新增） | 见§9 |
| `tests/test_p2n_new_original_obs_*.py`（新增） | 原encoder等价性、context、gate、恢复与normalizer检查 |
| `tests/p2n_new_original_obs_ddp_smoke.py`（新增） | 两variant的双卡训练与续训验收 |

原 `FusedObservationEncoder`、`RobomimicRgbEncoder`、`ProjectionStateEncoder` 和 modern AR 文件直接复用，不为本方案改变其算法。

如复用当前obs factory，应增加专用schema分支，不给fused类型注入64-query/Resampler默认值，不调用冻结backbone helper。也可以由新adapter独立白名单构造；正式实现选一种统一路径，不能双重构造两个encoder。

新policy生命周期可复用modern公共逻辑，但必须明确覆写tuple观测假设、预归一化、DINO专用rollout、策略命名、源码元数据和encoder导出。Gate context需走新的融合分支，不能再reshape成camera×queries布局。

## 9. 两份配置与未来启动方式

拟新增配置，相对`oat/config`：

```text
experimental/train_p2n_new_original_obs_real_robot.yaml
experimental/train_p2n_state_gate_new_original_obs_real_robot.yaml
```

分别继承当前modern基础版/gate版真机配置的task与训练设置，覆盖policy/workspace target及观测schema。原Nut Washer N77 dataset、seed42、val_ratio0.05和用户指定tokenizer继续沿用，不从旧pen_cabinet配置复制数据路径。

以下是拟定schema，**尚待实现，当前不可执行**：

```yaml
_target_: oat.workspace.train_p2n_new_original_obs.TrainP2NNewOriginalObsWorkspace

policy:
  _target_: oat.policy.p2n_new_original_obs.P2NNewOriginalObsPolicy
  obs_encoder_type: original_fused
  context_layout: original_fused_v1
  context_schema_version: 2
  original_obs_config:
    crop_shape: [112, 112]
    eval_fixed_crop: true
    use_group_norm: true
    share_rgb_model: false
    pretrained: false
    state_out_dim: null
    feature_dimension: 64
    spatial_softmax_num_kp: 32
    spatial_softmax_temperature: 1.0
    spatial_softmax_noise: 0.0
  embed_dim: 768
  n_layers: 16
  n_heads: 12
  ffn_dim: 2048
  dropout: 0.1
  expected_action_tokens: 8
```

原encoder自身没有暴露的参数由wrapper验证为原实现支持的固定值，不假装传给一个不接受该字段的构造器。导出同时记录实际resolved结构与依赖版本。

Gate配置改用`P2NStateGateNewOriginalObsPolicy`，保留原state-history配置、rot6d rows、4 summaries和gate设置。Variant仍分别为`p2n_new`、`p2n_state_gate_new`。

继承得到的DINO/Nano路径必须为空；Resampler字段在新resolved schema中移除或显式禁用，不构造任何相关参数。不能让旧默认64 queries参与新合同校验。

拟定launcher接口，**仅用于说明未来使用方式**：

```bash
python scripts/train_p2n_new_original_obs.py \
  --variant p2n_new --task real_robot \
  --output output/training/p2n_new_original_obs_nut_washer_seed42 \
  --dry-run

python scripts/train_p2n_new_original_obs.py \
  --variant p2n_state_gate_new --task real_robot \
  --output output/training/p2n_state_gate_new_original_obs_nut_washer_seed42 \
  --dry-run
```

不要求DINO/ConvNeXt预训练目录；OAT来源由所继承真机配置或明确`--tokenizer`覆盖。Dry-run只解析，preflight做CPU schema/参数构造检查；设备参数、batch、epoch等用户覆盖原样透传。

## 10. 训练与初始化配方

- 原ResNet18分支从随机初始化训练；不额外下载ImageNet权重或套冻结主干策略。
- AR、bridge、时间/type embedding与history/gate按新run初始化；指定OAT冻结。
- 建议原obs encoder使用`obs_enc_lr=1e-5`，沿用原版观测学习率；modern AR、bridge和history/gate使用`policy_lr=5e-5`。
- AdamW、BF16、cosine、EMA、梯度累积与成功update计数使用modern工作区；weight decay默认沿用modern值0.01，norm/bias按原分组规则为0。保留原encoder并不表示恢复整套旧训练配方。
- 以Parameter对象身份去重并明确分组；bridge位于adapter内部也应进入policy LR，不能仅靠`obs_encoder.*`前缀将其全部分给obs LR。
- Normalizer和OAT不进入optimizer；不用的旧type行/observation_pool/视觉投影不保留为可训练参数。
- 双卡DDP每卡均包含完整模型。ResNet现在有反向传播和Adam状态，不能仅依据context缩短就承诺显存一定更低。
- 每卡batch、累积和checkpointing沿用同条件测试设置；不能直接照搬以前某次运行的batch128作为默认。
- 两variant分别训练；训练时长上限和用户覆盖从原modern真机配方继承。

本版本不默认导入旧策略或旧encoder训练权重。后续若要求迁移旧encoder，可定义独立init流程；不能用跨架构resume或宽松加载处理。

## 11. Artifact、恢复与兼容性

Artifact保存完整原encoder、bridge、AR、OAT、history/gate和所有normalizer权重，以及：

- `obs_encoder_type=original_fused`、新的policy target、variant和context schema/layout。
- 实际RGB/state顺序、`D_obs`、crop、GN、SpatialSoftmax、ReLU、相机共享设置及初始化约定。
- 两帧位置编码、四类type embedding、11/15-token布局和新segment整数。
- Gate观测池化`mean_fused`与summary-only bias规则。
- 训练集划分、动作/状态单位、task_uid、tokenizer来源和执行协议。
- Student/EMA、optimizer、scheduler、self-past计数和全部RNG状态。
- robomimic/torchvision/torch等版本及实际涉及文件的源码hash。

Restore按保存结构构造并strict load，不依赖外部ResNet权重、原OAT路径或网络；normalizer加载后不可重新拟合。

Resume提前比较encoder/layout/crop/state/共享权重/桥接维度等有效配置，拒绝DINO↔Nano↔original-fused、基础↔gate以及旧小型AR↔modern AR之间的跨架构resume。相同variant字符串不能绕过这些检查。

旧DINO/Nano入口和checkpoint保持各自原schema/行为，不自动升级到新的fused布局。新segment扩展需有旧context、cached generation和checkpoint回归检查。

## 12. 速度与质量验收

本版本同时改变视觉计算方式和条件序列长度；这是保持原观测融合结构的整体方案，不将全部收益归因于ResNet模型参数量。

预期减少的是224输入的冻结视觉前向、Resampler以及长memory的cross-attention投影开销；新增的是可训练ResNet的反向与optimizer状态。实际速度/显存需测量，不能承诺倍数。

短测要求：

1. 同一硬件、数据、batch/累积、BF16、AR配置、OAT、self-past参数和生成设置。
2. 分别覆盖self-past `p=0`和固定`p=0.5`，不能只测warmup。
3. 报告原encoder forward/backward、self-past、AR训练、optimizer、数据等待及完整step时间。
4. 测量覆盖warmup和Adam状态建立后，记录allocated/reserved峰值、p50/p95和实际吞吐。
5. 同时记录两种context长度、裁剪大小和CNN冻结状态，明确两条视觉路径的差异。
6. 使用相同split检查CE、OAT解码动作误差、generated-history验证；闭环成功率通过独立机器人协议确认。

现有self-past“先全部生成再按概率选用”的执行开销可能继续存在。其样本选择/分块优化属于后续独立事项，不在本版中静默修改，也不为性能承诺提供依据。无需额外长期架构消融矩阵。

## 13. 实施顺序与针对性检查

| 阶段 | 工作 | 必须通过的出口 |
|---|---|---|
| A | 原encoder wrapper、normalizer、crop与尺寸核验 | 同权重eval时bridge之前与原encoder输出一致；相同crop RNG训练输出一致 |
| B | 139→768 bridge、帧位置、fused context | 原始状态只归一化一次；11-token基础context正确 |
| C | Gate融合观测汇总与segment扩展 | 15-token context、原summary有效性与门控只影响history列 |
| D | 原obs训练模式、optimizer、EMA与self-past | 两相机CNN及bridge有梯度、OAT无梯度；临时eval后正确恢复 |
| E | 两配置、launcher、workspace、artifact | 同架构strict恢复一致，错误resume明确拒绝 |
| F | 两variant双卡短测 | 无unused参数/NaN，累积/EMA/续训正确，得到同条件性能报告 |
| G | 真机任务正式训练与评估 | 固定split指标、动作生成和独立闭环验收 |

补充验收要点：

- 保留每相机64维特征、状态identity、task_uid和原字段顺序。
- 原encoder与adapter的随机初始化/预训练状态如实报告。
- Basic不实例化history/gate，gate不遗留未使用的1536→768观察池化层。
- New context至少有有效fused观测；padding在投影/attention前正确清理。
- Gate open/closed仍提供完整metadata；history被关闭时无旁路泄漏。
- 旧DINO/Nano context schema与segment数值语义保持，现代AR的cached/full预测一致性不变。
- 不通过假的Resampler、重复tokens或假proprio段满足旧接口。
- 保存部署模型后，删除外部初始化路径的依赖仍可加载并预测。

## 14. 本地参考

- [原Nut Washer训练入口](../train_nut_washer_v3.sh)。
- [原基础版配方](../oat/config/train_past2next_scratch_all500.yaml)、[原state-history配方](../oat/config/experimental/train_past2next_state_history.yaml)。
- [原融合encoder](../oat/perception/fused_obs_encoder.py)、[Robomimic视觉encoder](../oat/perception/robomimic_vision_encoder.py)、[状态encoder](../oat/perception/state_encoder.py)、[CropRandomizer](../oat/perception/crop_randomizer.py)。
- [真机normalizer](../oat/dataset/real_robot_dataset.py)、[Nut Washer schema](../oat/config/task/policy/real_robot/nut_washer_with_prev_window.yaml)。
- [modern AR](../oat/model/autoregressive/modern_transformer_cache.py)、[ContextBatch](../oat/model/common/context_batch.py)、[modern公共策略](../oat/policy/p2n_new_common.py)、[modern gate](../oat/policy/p2n_state_gate_new.py)。
- [现有Nano独立policy接入示例](../oat/policy/p2n_new_convnext.py)、[encoder factory](../oat/perception/obs_encoder_factory.py)。本分支不能继承其“视觉主干必须冻结、输出必须含Resampler”的假设。
