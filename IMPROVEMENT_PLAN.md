# Past2Next 改进计划与当前阶段

更新日期：2026-09-21。范围：`past2next_bug_fixed` 中的 LIBERO Past2Next 策略改进。

**当前阶段：基础修复与连续状态历史版本已实现，功能测试已通过；H8 版本已开始正式双 GPU 训练，成功率与速度收益尚待对照实验验证。**

## 1. 目标与已经确定的边界

目标是在任务成功率、方法新颖性和推理速度之间取得平衡。模型和代码结构可以调整，但始终保留三个核心要素：动作 tokenizer、离散 token 自回归 policy、过去动作与动态特征条件。

当前主线是验证：把过去动作和机器人实际状态响应按时间对齐，能否帮助模型生成更好的下一动作块。

- 使用 conjugate 版本的动作 tokenizer；当前 policy 实验先冻结并复用同一个 checkpoint。
- 训练采用离线示范数据和 self-past 历史替换，不在训练中执行生成动作，不引入 DAgger 或在线教师。
- 推理使用执行层确认的过去动作，并逐控制步记录实际状态。
- 图像保持两帧，首先增加成本较低的低维状态历史。
- `val_ratio=0.1` 是用户选定的划分，保留该设置。
- 独立保留各策略版本，方便复现和对照；state-history 的两个旧启动入口已合并为 [train_state_history.sh](train_state_history.sh)。

## 2. 已完成的工作

| 工作 | 当前状态与含义 | 主要代码 |
| --- | --- | --- |
| EMA 共享参数重复更新修复 | 每个独立 Parameter 每次只更新一次，避免 token embedding/output head 的共享权重被重复更新 | [ema_model.py](oat/model/diffusion/ema_model.py) |
| LIBERO 归一化数据隔离 | 新拟合的 normalizer 只统计训练 episode；tokenizer 与 policy 的新训练都受益 | [zarr_dataset.py](oat/dataset/zarr_dataset.py) |
| 禁止生成 BOS | 每一步采样/argmax 前屏蔽 BOS，覆盖普通预测和 self-past 内部生成 | [transformer_cache.py](oat/model/autoregressive/transformer_cache.py) |
| 起始历史与 self-past 进度修复 | 零填充、有效性标记、跳过不存在的 previous window；按成功 optimizer update 记录并恢复课程进度 | [历史数据集](oat/dataset/zarr_dataset_with_past.py)、[self-past policy](oat/policy/past2next_self_past.py) |
| 评估协议整理 | 修复旧入口的默认协议和目录清理；本主线使用 candidate evaluator，并保留各版本对应的 runner | [evaluate_candidate.py](scripts/evaluate_candidate.py)、[旧评估入口](scripts/eval_policy_sim.py) |
| 执行确认的动作历史 | 预测不会立即写入历史；执行层反馈实际执行前缀后才更新，支持部分执行与零执行 | [executed policy](oat/policy/past2next_executed_past.py)、[runner](oat/env_runner/executed_action_runner.py) |
| 连续状态历史与融合 | 数据采样、每控制步状态缓存、状态差分、相对旋转、时间编码和 AR 条件融合均已实现 | [state-history policy](oat/policy/past2next_state_history.py) |
| 统一训练入口 | 双 GPU、在线 W&B、环境评估、只保留评估轮次 checkpoint | [train_state_history.sh](train_state_history.sh) |

这些修复作用于当前代码及之后的训练。已有 checkpoint 的训练过程和已保存统计量不会被追溯修改。当前使用的历史 conjugate tokenizer 保留自己的 normalizer；其旧的全数据归一化统计，是后续建立严格数据隔离的两阶段基线时需要单独处理的事项。

## 3. 各版本的关系与对照组

| 版本 | 离线训练中的过去动作 | 推理中的过去动作 | 新增连续状态历史分支 |
| --- | --- | --- | --- |
| [past2next.py](oat/policy/past2next.py) | 示范记录 | 预测后提前更新，假定计划执行前缀全部执行 | 无 |
| [past2next_self_past.py](oat/policy/past2next_self_past.py) | 按课程混合示范历史与生成历史 | 同上 | 无 |
| [past2next_executed_past.py](oat/policy/past2next_executed_past.py) | 示范记录 | 执行确认的命令前缀 | 无 |
| [past2next_self_past_executed.py](oat/policy/past2next_self_past_executed.py) | 与原 self-past 相同 | 执行确认的命令前缀 | 无 |
| [past2next_state_history.py](oat/policy/past2next_state_history.py) | 保留离线 self-past | 执行确认的命令前缀 | 有 |

**当前直接对照组选 self-past + executed 版本。** 它与 state-history 候选保持相同的 self-past 训练和推理反馈规则，便于判断新增历史分支的贡献。入口为 [train_self_past_executed_live.sh](train_self_past_executed_live.sh)。

[train_baseline.sh](train_baseline.sh) 对应修复后的 `train_past2next_scratch` 配方，实际实例化的是 self-past policy，不是 expert-only 的 `past2next.py`。其训练时长和 rollout 默认设置也与当前长训练脚本不同，比较前必须对齐。

两个 live 启动脚本的 checkpoint 留存设置目前不同；正式对照前应对齐评估 epoch 网格和模型选择规则，并确保待比较的 checkpoint 均有保留。

原 self-past 与 self-past + executed 在架构配置匹配时可复用权重。state-history 增加了历史编码器和条件位置编码，需要重新训练 policy；动作 tokenizer 仍可复用。

## 4. 当前已实现的历史方案

```text
连续低维状态 + 相邻状态变化 + 对齐的过去动作 + 有效性标记
                              ↓
               时间位置编码 + 2 层 Transformer
                              ↓
                   4 个查询汇总为历史条件
                              ↓
与原有两帧图像/状态、过去动作和动作命令差分条件拼接
                              ↓
          离散 AR policy → 动作 tokens → 冻结 tokenizer 解码
```

状态包括末端位置、末端姿态和夹爪开度。位置与夹爪变化采用归一化状态的相邻差分；姿态使用 xyzw 四元数转旋转矩阵，再将绝对姿态及世界系相对旋转 `R[t] @ R[t-1].T` 表示为 6D。没有把四元数直接相减，也没有将这些差分除以时间间隔。

这里的“状态变化”描述测得的响应；“过去动作”描述下发的命令，两者不是同一个量。原有动作命令的一阶/二阶差分条件仍然保留，也不应称作实测速度或加速度。

| 设置 | H8（默认） | H16 |
| --- | --- | --- |
| 连续状态帧数 | 8 | 16 |
| 对齐的过去动作数 | 7 | 15 |
| 可计算的相邻状态变化 | 7 | 15 |
| 每路相机的图像帧数 | 2 | 2 |
| 新增历史汇总 token 数 | 4 | 4 |
| AR 条件总长度 | 15 | 23 |

时间对齐为：状态 `s[t-H+1] … s[t]`，动作 `a[t-H+1] … a[t-1]`；动作 `a[i]` 对应状态变化 `s[i] → s[i+1]`。episode 开始只有真实存在的状态有效，补齐位置带掩码，不跨 episode 取历史。

推理 runner 在一个动作块内部的每个实际控制步记录低维状态，因此每执行 8 步重新规划也不会漏掉中间状态。历史快照通过额外的小型 RPC 返回；这部分开销需要纳入后续耗时测量。

训练中的状态仍来自示范。self-past 被选中时，命令历史可能来自模型生成，不能把它与示范状态变化解释为一次真实执行后的因果转移。当前没有添加动力学监督、命令跟踪误差损失或状态条件 tokenizer。

实现细节见 [STATE_HISTORY.md](STATE_HISTORY.md)、[历史编码器](oat/model/state_action_history.py)、[数据集](oat/dataset/zarr_dataset_with_state_history.py)和[runner](oat/env_runner/state_history_runner.py)。

## 5. 当前训练设置与进度快照

下表是 **统一 bash 入口覆盖后的有效设置**；直接运行 YAML 不会自动应用这些覆盖。

| 项目 | 当前默认值 |
| --- | --- |
| GPU / batch | GPU 0、1；每卡 32，全局 64 |
| 训练长度 | 2001 epochs，epoch 标签 0–2000 |
| 图像 / 动作 | 两帧图像；预测 16 步，每轮通常执行前 8 步 |
| 历史 | H8 状态，7 步过去动作 |
| AR / 历史编码器 | AR 宽度 256、8 层、8 头；历史宽度 128、2 层、4 头 |
| Self-past | 最大替换概率 0.5；1000 次 optimizer update warmup，4000 次 ramp |
| Policy / vision 学习率 | 均为 `1e-5` |
| 数据 | LIBERO-10 N500，训练/验证比例 90/10，划分 seed 42 |
| 日志 / EMA | W&B online，修复后的 EMA 开启 |
| 离线验证 | 每 epoch；包含 expert-history 与 generated-history 验证 |
| 环境评估 | `lazy_eval=false`，corrected 协议；每 100 epoch，100 trials、10 个并行环境 |
| Checkpoint | 只在评估轮次保存，全部保留；无额外 snapshot、无 `latest.ckpt` |

默认保存 `ep-0000_sr-….ckpt`、`ep-0100_sr-….ckpt`，直到 `ep-2000`，共 21 个。epoch 0 的评估发生在第一轮训练结束后。checkpoint 间隔随 `ROLLOUT_EVERY` 自动调整。

当前复用的 conjugate tokenizer：

```text
/workspace/past2next_clean/output/20260827/070913_train_oattok_so3aug_libero10_N500/checkpoints/ep-1300_mse-0.001.ckpt
```

**2026-09-21 07:05:16 UTC 的本地检查记录：**

- H8 正式训练已启动，检测到 launcher 和两个训练进程。
- 运行目录：[state_history_seed42_20260921_065955_803707596](output/training/state_history_seed42_20260921_065955_803707596)。
- [保存配置](output/training/state_history_seed42_20260921_065955_803707596/.hydra/config.yaml)确认从头训练 policy，复用上述 tokenizer。
- [logs.json](output/training/state_history_seed42_20260921_065955_803707596/logs.json)最新记录为 epoch 0、`global_step=1944`；尚无验证/rollout 成功率或 checkpoint 文件。
- 本地未发现 H16 的正式训练记录。

这是一份采样时刻的进度记录，后续以该运行目录的新日志为准。运行产物通常不随代码提交到 Git。

## 6. 已验证与尚未证明的内容

之前完成的相关验证包括 200 项 CPU 测试，以及完整模型配置、现有 conjugate tokenizer 和少量真实 LIBERO 数据的前向、反向、预测检查。脚本合并后也完成了语法及默认/覆盖参数的 dry-run 检查。本次编写文档没有重新跑训练或整套测试。

| 已有证据 | 对应检查 |
| --- | --- |
| H8/H16 因果采样、边界掩码、训练集归一化 | [dataset tests](tests/test_state_history_dataset.py) |
| 相对旋转、q/-q 一致性、动作对齐、补齐隔离 | [encoder tests](tests/test_state_action_history.py) |
| Self-past 反向传播、混合精度、优化器覆盖、EMA、checkpoint 恢复 | [policy tests](tests/test_state_history_policy.py) |
| 双进程 DDP 的梯度和权重同步 | [CPU gloo tests](tests/test_state_history_ddp.py) |
| 每控制步缓存、部分/零执行、异步 worker 与重置 | [runner tests](tests/test_state_history_runner.py)、[闭环模拟测试](tests/test_state_history_closed_loop.py) |

这些检查支持代码行为与接口正确，不能证明成功率已经提高。尚需收集同条件的任务成功率、推理延迟、吞吐和显存结果。历史 015/043/046 的分数也不能直接当作此次新架构的受控对照，见[历史比较的限制](COMPARISON_015_043_046.md)。

## 7. 下一阶段实验计划

建议按以下顺序推进，具体实验顺序可根据 H8 结果调整。前三项聚焦当前主线；后面的模型变化属于待实验结果支持后再选择的候选方向，尚未确认实施。

| 优先级 | 工作 | 要回答的问题 / 完成条件 |
| --- | --- | --- |
| P0 | 跟踪当前 H8 训练，核查首个验证、rollout 和 checkpoint | 完整训练—评估—保存流程是否正常；获得第一批真实闭环结果 |
| P1 | 与 self-past + executed 做受控对照 | 在相同 tokenizer、数据、预算、评估协议下，新增历史分支是否提高成功率，增加多少推理成本 |
| P2 | 运行 H16，并补充动作历史长度对照 | 更长历史是否有用；收益来自更长状态历史还是更多过去动作 |
| P3 | 状态/状态变化/时间编码消融 | 分别保留绝对状态、加入状态变化、使用完整时间编码，确认每个成分的贡献 |
| P4 | 分段耗时测量和历史编码成本实验 | 测量视觉编码、历史编码、AR 生成、token 解码和 runner 传输；比较汇总 token 数、编码宽度及更简单的融合结构 |
| P5 | 多随机种子和独立最终评估 | 检查稳定性，报告任务级结果和不确定性，确认收益不依赖单次训练或 checkpoint 选择 |
| P6 | 视结果决定进一步结构改进 | 候选包括门控融合、轻量时间编码、状态条件/残差式 tokenizer；需单独设计、实现和验证 |

**长度对照的限制：** 当前代码要求 `H=past_n+1`。H8→H16 同时把过去动作从 7 步改为 15 步，AR 总条件长度也从 15 改为 23。它不能单独证明“更长状态历史”的贡献。可增加 past_n=15 的无状态历史基线；若要独立改变状态与动作长度，需要新的对齐设计。

**消融的实现状态：** 历史编码宽度、层数和汇总 token 数已有配置字段；“只用状态”“关闭状态差分”“关闭时间编码”等功能隔离开关尚未实现，不能把它们当成现成的 bash 参数。

**速度方向的起点：** 当前 AR 已有 SDPA、KV cache 和 token-prefix 解码能力。先测量实际瓶颈，再比较 token 预算或编码结构，避免把已有机制当成新增贡献。减少 token 数也必须同时测量解码误差和任务成功率。

**Tokenizer 分两步处理：** 首轮 policy 架构对比固定现有 conjugate tokenizer，以控制变量。之后用修复后的训练集归一化重训 conjugate tokenizer，并为相关 policy 重新训练，建立严格数据隔离的两阶段结果；不要把 tokenizer 变化和历史分支变化混在一次对比中。

## 8. 对照实验与结果记录规则

比较时固定 tokenizer checkpoint/统计量、训练/验证 episode 划分、全局 batch、更新预算、图像裁剪、AR 配置、self-past 课程、EMA 和推理 token 数。更换模型 seed 时，数据划分 seed 仍固定为 42。GPU 数和每卡 batch 的组合应保持可比。

环境评估固定 protocol、任务集合、初始状态/episode seed、每任务 trials、步数上限、温度、裁剪和执行步数。训练中的 100 trials 用于开发监测；正式比较建议采用预先确定、任务均衡的更大评估集，例如每任务 50 次、共 500 次，并在模型选择后使用独立的最终评估计划。

至少记录以下指标：

- **成功率：** 总体及各任务成功率、实际 trial 数、跨 seed 波动或置信区间。
- **速度：** 固定硬件/精度/batch 的 policy 延迟 p50/p95，以及包含历史快照传输的端到端耗时；GPU 计时需同步并预热。
- **资源：** 峰值显存、训练吞吐、参数量和实际 AR 条件长度。
- **离线诊断：** expert/generated-history token loss、动作重构误差；这些指标用于诊断，不替代闭环成功率。

建议每个结果条目保存：代码 commit、配置、policy/tokenizer checkpoint、数据划分、训练 seed、评估协议及 episode 计划、checkpoint 选择规则、硬件和精度。新颖性的证据来自可解释的设计和受控消融，当前新增分支本身尚不足以形成研究结论。

## 9. 常用入口与后续更新

```bash
cd /workspace/past2next_bug_fixed

# 仅检查配置
bash train_state_history.sh --dry-run

# 新的 H8 训练；不是自动续训命令
bash train_state_history.sh

# 新的 H16 训练
STATE_HISTORY_STEPS=16 bash train_state_history.sh

# 覆盖训练时长和评估间隔，checkpoint 间隔自动同步
NUM_EPOCHS=2001 ROLLOUT_EVERY=100 VAL_EVERY=1 bash train_state_history.sh
```

当前 H8 已在运行，以上训练命令用于有意启动新的实验。原脚本名 `train_state_history_live.sh` 和 `train_state_history_eval_ckpts_live.sh` 已停用；统一使用 `train_state_history.sh`。

- [x] 基础修复与数据/评估约定。
- [x] Self-past + 执行确认版本。
- [x] 连续状态历史、状态变化、时间融合、H8/H16 支持。
- [x] CPU 功能/集成检查与统一双 GPU 启动入口。
- [x] H8 正式训练已启动。
- [ ] 收集并核查 H8 的首个 rollout、验证指标和 checkpoint。
- [ ] 完成同条件 baseline/H8/H16 对照。
- [ ] 完成功能消融和端到端耗时测量。
- [ ] 完成多 seed 和独立最终评估。
- [ ] 根据结果选择下一项模型改进，更新本文件及对应证据路径。
