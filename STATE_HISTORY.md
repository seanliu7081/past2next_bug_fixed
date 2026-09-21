# 连续状态历史版 Past2Next

这是独立的新实验入口，现有 policy、dataset、runner、配置和启动脚本均不需要修改。

## 实现内容

保留冻结的动作 tokenizer、离散 token 自回归预测、原有两帧图像/状态观测、过去动作及动作命令差分条件。新增历史分支：

```text
连续状态历史 + 状态变化 + 对齐的过去动作 + 有效性标记
                         ↓
          时间位置编码 + 2 层 Transformer
                         ↓
              4 个可学习查询汇总历史
                         ↓
追加到原有图像/状态/动作条件 → AR 生成动作 tokens → tokenizer 解码
```

默认历史编码宽度 128、4 个注意力头，输出 4 个历史条件 token。H=8 时 AR 条件总长度为 2+2+7+4=15。所有历史信息都在当前决策时刻及之前，历史编码器内部允许双向注意力。

状态字段为 `robot0_eef_pos`、`robot0_eef_quat`、`robot0_gripper_qpos`。位置和夹爪开度使用训练集 normalizer，变化量是归一化状态的相邻差分，没有除以时间间隔，因此不称为实测速度。四元数按 xyzw 解释，绝对姿态和世界系相对旋转 `R[t] @ R[t-1].T` 转为 6D，避免直接相减以及 q/-q 跳变。夹爪动作命令与测得的夹爪开度分别作为特征，不直接相减。

## 时间对齐

`state_history_steps=H` 表示 **H 帧状态**，对应 `past_n=H-1` 个过去动作：

```text
状态：s[t-H+1], s[t-H+2], ..., s[t]
动作：a[t-H+1], a[t-H+2], ..., a[t-1]
每个 a[i] 与 s[i] → s[i+1] 的变化对齐。
```

默认 8 帧状态、7 个过去动作；也支持 16 帧状态、15 个过去动作。图像始终保留两帧。episode 开始只有 s[0] 有效，没有有效历史动作；补齐位置为零并带掩码，不能产生虚假的状态差分。

训练数据同时为 `obs` 和 `prev_obs` 提供连续状态历史。推理 runner 在动作块内**每个实际控制步**缓存低维状态；执行确认只把实际执行的动作前缀写入动作历史。提前结束的环境不会把未执行后缀写入历史。图像 transport 保持原来的两帧接口，另加一次小型状态快照 RPC。

训练仍为离线 self-past：一部分历史命令来自模型预测，状态历史来自示范记录。这些预测命令没有在训练时执行；新分支没有加入在线采集、DAgger 或动力学监督目标。

## 启动

从本目录运行：

```bash
# 只解析配置，不启动训练、环境或 W&B
bash train_state_history.sh --dry-run

# 默认 8 帧状态历史
bash train_state_history.sh

# 16 帧状态历史，同时 past_n 自动变为 15
STATE_HISTORY_STEPS=16 bash train_state_history.sh
```

脚本默认使用 GPU 0、1，每卡 batch size 32，总 batch size 64；在线 W&B、`lazy_eval=false`、修复后的 EMA、全新 policy。默认 `NUM_EPOCHS=2001`、`ROLLOUT_EVERY=100`、`VAL_EVERY=1`，可通过同名环境变量覆盖，例如：

```bash
NUM_EPOCHS=2001 ROLLOUT_EVERY=100 VAL_EVERY=1 \
CUDA_VISIBLE_DEVICES=0,1 bash train_state_history.sh
```

checkpoint 频率自动跟随 `ROLLOUT_EVERY`，只在 rollout 评估轮次保存，并全部保留。默认保存 epoch 标签 `0、100、200、…、2000`，共 21 份；epoch 0 的评估和保存发生在完成第一个训练 epoch 后。不做 top-k 淘汰，也不额外保存 `latest.ckpt` 或 snapshot。

脚本复用已有 conjugate tokenizer：

```text
/workspace/past2next_clean/output/20260827/070913_train_oattok_so3aug_libero10_N500/checkpoints/ep-1300_mse-0.001.ckpt
```

也可显式指定 tokenizer 和输出目录：

```bash
bash train_state_history.sh /path/to/tokenizer.ckpt /path/to/new_output
```

无需重转 Zarr。policy 的 normalizer 继续只统计训练 episode；冻结 tokenizer 的已有权重及其 normalizer 保持原样。`val_ratio=0.1` 保留用户设置。

新 policy 增加了历史编码器参数并延长条件位置编码，应从头训练；现有旧 policy checkpoint 不能直接严格加载为此架构。新版本自己保存的 checkpoint 支持正常恢复和评估，保存配置会指向新 runner。若用现有评估脚本，应使用 `scripts/evaluate_candidate.py` 并保留 checkpoint 中的 runner target。

H8/H16 对比会同时改变状态长度和动作历史长度，属于联合历史长度实验。正式实验还需对比成功率、推理耗时及显存；目前没有性能提升结论。

## 新文件

| 文件 | 作用 |
| --- | --- |
| `oat/model/state_action_history.py` | 状态、相对旋转、动作对齐及时间编码 |
| `oat/policy/past2next_state_history.py` | 新分支与 AR/self-past/执行确认集成 |
| `oat/dataset/zarr_dataset_with_state_history.py` | 当前及 previous-window 的连续状态采样 |
| `oat/env_runner/state_history_runner.py` | 每控制步采集和向量环境快照 |
| `oat/config/experimental/train_past2next_state_history.yaml` | 独立训练配置 |
| `train_state_history.sh` | 双 GPU、在线 W&B，保留所有评估轮次 checkpoint |

## 验证范围

新增 CPU 测试覆盖 H8/H16 因果采样、episode 边界、训练集归一化、旋转几何、掩码、动作对齐、混合精度反向传播、优化器参数覆盖、EMA 和 checkpoint 恢复。双进程 gloo 测试覆盖 expert 与 generated self-past 更新；异步模拟环境测试覆盖完整块、部分执行、零执行、重置和状态/动作对应关系。

另用完整模型配置、现有 conjugate tokenizer 和两段真实 LIBERO 数据做了 CPU 前向/反向/预测检查，确认 tokenizer 不变、图像仍为两帧。没有启动正式训练或真实 LIBERO rollout；训练吞吐和成功率仍需实测。
