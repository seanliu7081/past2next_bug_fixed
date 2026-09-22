# State-history gate 独立实验

本版本在 C（`Past2NextStateHistoryPolicy`）上新增一个可学习的历史门控，独立使用新 policy、AR helper、配置和启动入口。原有 C 文件与训练入口保持原样。实现和检查不会自动启动训练；正式训练由用户运行下面的命令。

目标是检验：模型是否能根据当前观测，调整对新增连续状态历史分支的依赖。成功率或速度收益需要后续受控实验确认。

## 门控作用在哪里

条件顺序沿用 C：当前两帧观测、动作命令差分、过去动作、最后的历史汇总 tokens。默认最后 4 个 tokens 来自连续状态/动作历史编码器。

门控在 AR 的条件 cross-attention 中，为这些末尾历史 tokens 的 attention logits 加上 `log(g)`；其他条件的 bias 为 0：

```text
当前两帧观测特征的均值 ─┐
4 个历史汇总 token 均值 ├─ 拼接 → LayerNorm → Linear(128) → GELU → Linear(1)
有效状态帧比例 ─────────┘                              ↓
                                                 g = sigmoid(logit)
                                                      ↓
attention logits[..., 最后4个历史tokens] += log(g)
```

每个样本使用一个共享标量 `g`，广播至各 attention heads 和查询位置。它调整历史 tokens 相对于其他条件的权重；不改变条件序列长度，也不减少 AR 解码步数。实现使用数值稳定的 `logsigmoid` 计算可学习 log gate。

`g=0.9` 表示对历史 tokens 的**未归一化 attention 权重乘以 0.9**，不是“历史占 90% 的注意力”。经过 softmax 后的实际份额还由其他 attention logits 决定。

| 模式 | 行为 | 用途 |
| --- | --- | --- |
| `learned`（默认） | 根据观测和历史预测 `0 < g < 1` | 训练自适应门控 |
| `open` | 固定 `g=1`，即 log bias 为 0 | 同结构的完全开放对照 |
| `closed` | 屏蔽末尾历史汇总 tokens | 检查新增历史分支的影响 |

**`closed` 只关闭新增的历史汇总 token 通路。** 原有两帧观测中的状态、过去动作以及动作命令差分条件仍然有效；它不等于关闭全部历史，也不等于让模型只看图像。历史编码计算不会因此自动省去，因此不能把 `closed` 的运行时间当作无历史编码器的速度。

新建模型时，门控 MLP 的末层权重初始化为 0，bias 初始化为 `logit(history_gate_init)`；默认 `history_gate_init=0.9`，起点接近完全开放，训练后可随输入变化。`open`/`closed` 的 gate MLP 固定，不学习。

`get_history_gate_metrics()` 可读取最近一次 batch 的 gate mean/min/max，便于诊断。原有 workspace 没有修改，因此这些新指标**不会自动出现在 W&B**。

## 新文件与配置

- [新 policy](oat/policy/past2next_state_history_gate.py)：门控预测、条件构造及 C/gate 权重加载。
- [继承配置](oat/config/experimental/train_past2next_state_history_gate.yaml)：继承现有 state-history 配方，替换 policy target，新增 gate 参数和日志 tags。
- [新训练入口](train_state_history_gate.sh)：负责正式训练的有效参数覆盖。

配置新增字段：

```yaml
policy:
  history_gate_mode: learned
  history_gate_hidden_dim: 128
  history_gate_init: 0.9
```

除门控外，默认保持 C 的 H8、7 步过去动作、两帧图像、4 个历史汇总 tokens。H16 同时增加状态与过去动作长度，仍不能单独隔离状态历史长度的贡献。离线 self-past 的状态来自示范，生成动作与示范状态也仍然不是一次真实执行的因果配对。

## 由用户启动训练

进入仓库：

```bash
cd /workspace/ysk/past2next_bug_fixed
```

先检查配置。此命令允许 tokenizer、`INIT_CHECKPOINT` 和数据集路径不存在，明确只做 Hydra 配置解析，不加载或验证权重与数据，不创建训练进程或 W&B run：

```bash
bash train_state_history_gate.sh --dry-run /path/to/tokenizer.ckpt
```

数据集默认位于仓库的 `data/libero/libero10_N500.zarr`。如果放在其他位置，用 `DATASET_PATH` 指定实际 Zarr 目录；相对路径按调用者工作目录解析为绝对路径。下面先设置该变量，后续训练命令会继续使用它；请把示例路径替换为真实路径：

```bash
export DATASET_PATH=/path/to/your/libero10_N500.zarr
```

从头训练新的 gate policy：

```bash
DATASET_PATH=/path/to/your/libero10_N500.zarr \
  bash train_state_history_gate.sh /path/to/tokenizer.ckpt
```

真实启动会检查 tokenizer 文件、可选初始化权重文件和数据集目录是否存在，以及 Python 是否可执行；数据集内容和权重兼容性仍由实际加载验证。此次检查环境中，默认 tokenizer checkpoint 与默认数据集目录均不存在，因此正式训练前需要提供这两项真实输入。

也可用环境变量指定 tokenizer；位置参数优先：

```bash
TOKENIZER_CHECKPOINT=/path/to/tokenizer.ckpt bash train_state_history_gate.sh
```

指定输出目录，或用 H16 / 另一训练 seed：

```bash
bash train_state_history_gate.sh /path/to/tokenizer.ckpt /path/to/new-run

STATE_HISTORY_STEPS=16 SEED=43 \
  bash train_state_history_gate.sh /path/to/tokenizer.ckpt
```

`SEED` 仅改变训练/model seed，数据划分 seed 始终为 42。真实训练拒绝已存在的输出路径，不会覆盖旧 run。相对路径按调用者工作目录解释。默认 tokenizer 仍指向原来的历史 checkpoint；若此机器没有该文件，必须提供实际路径。

模式和初始化值可由环境变量设置：

```bash
GATE_MODE=open bash train_state_history_gate.sh /path/to/tokenizer.ckpt
GATE_MODE=closed bash train_state_history_gate.sh /path/to/tokenizer.ckpt
GATE_MODE=learned GATE_INIT=0.9 GATE_HIDDEN_DIM=128 \
  bash train_state_history_gate.sh /path/to/tokenizer.ckpt
```

默认使用 `CUDA_VISIBLE_DEVICES=0,1` 和 `/venv/oat/bin/python`，可用 `CUDA_VISIBLE_DEVICES`（恰好两张卡）和 `TRAIN_PY` 覆盖。默认每卡 batch 32、全局 batch 64、2001 epochs、online W&B、每 epoch 离线验证、每 100 epoch 进行 corrected rollout（100 trials、10 个并行环境）。

保存所有评估轮次 checkpoint，不做 top-k 删除，没有额外 snapshot，也不生成 `latest.ckpt`。默认 epoch 标签为 0、100、……、2000，共 21 个；epoch 0 指完成第一轮训练之后。`NUM_EPOCHS`、`ROLLOUT_EVERY`、`VAL_EVERY` 可覆盖，保存间隔自动跟随 `ROLLOUT_EVERY`。

直接使用 YAML 时，不会自动获得这些 bash 覆盖；如需同一配方，请使用新启动脚本。

## 从已有 C 或 gate 权重初始化

可选择加载兼容的 C 或 gate checkpoint 的 EMA 权重，开始一个新的实验：

```bash
INIT_CHECKPOINT=/path/to/C-or-gate.ckpt \
  bash train_state_history_gate.sh /path/to/tokenizer.ckpt /path/to/new-run
```

这是 **weight initialization，不是续训**：`training.resume=false`，优化器、学习率进度、epoch 和 self-past 课程从头开始。源 checkpoint 必须含 EMA 权重；匹配的已有 tokenizer、观测归一化统计量等模型状态也随 checkpoint 加载。

- **C → gate：** 仅支持具有历史编码器的 C/state-history 架构；缺失的新 gate 参数按当前 `GATE_INIT` 初始化，加载时给出提示。
- **gate → gate：** 加载已有 gate 参数，因此 `GATE_INIT` 不会覆盖训练好的 gate 权重。
- H8/H16、AR/视觉配置、历史编码器配置及门控宽度等必须与待加载权重兼容；其他缺失项、未知项和形状不匹配仍会报错。旧的无状态历史 baseline 不做自动迁移。

tokenizer 路径需要与源 policy 使用的 tokenizer 匹配。不要用一个 tokenizer 构建模型，再期望 INIT_CHECKPOINT 中保存的 tokenizer 权重自动转为另一个版本。

## 建议对照与检查边界

比较 `learned`、`open` 和 `closed` 时，固定 tokenizer、H、训练预算、self-past 课程、数据划分、模型初始化来源、评估任务/种子与 checkpoint 选择规则；从头训练和加载 C 权重初始化应分别比较。检查闭环成功率、任务级差异、p50/p95 推理时间以及包含历史传输的端到端时间。

当前复用 tokenizer 的旧归一化统计量不会被门控修复。若需要严格数据隔离结论，仍应按原改进计划重新训练 tokenizer 和相应 policies。

本次验证通过 37 项功能测试，包含 CUDA 和双进程 DDP 检查；启动脚本语法、Hydra 默认/覆盖配置、模拟启动参数及输入检查也已通过。原有 185 个文件的哈希保持一致，改动仅新增文件。没有启动正式训练或 W&B run。

这些检查支持代码与配置行为正确，不证明闭环成功率提高。默认真实数据和 tokenizer 未就位，尚未验证用户实际训练输入的完整加载及训练—评估流程。本入口不会自动寻找或恢复旧 run。
