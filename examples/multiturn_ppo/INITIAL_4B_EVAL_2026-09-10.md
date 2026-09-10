# Qwen3-4B-Instruct-2507 的 SWE 训练前评测

用户要求先验证已有 4B 的初始能力，并将后续实验限制在物理 GPU 0–3。
本轮保留同一套 SWE agent、提示、补丁规则、独立评分和 8 轮预算，仅替换模型。
`trainer.val_only=true`、`val_before_train=true`、`resume_mode=disable`，不执行 PPO 更新。
运行目录名称沿用通用启动器的 `training-` 前缀，不表示实际执行了训练。

## 独立验证集：1/6 成功

运行 `ppo-eval-4b-8turn-val6-20260910-01` 正常退出，结果核验通过。
六条任务与此前 Qwen3-1.7B 的初始验证相同，初始消息完全相同，实际请求 temperature 都为 0.7。
两者均使用 8 turns、768 response tokens、12288 context，以及每题一条 rollout。
4B 共生成 39 次模型调用，四条非空补丁，没有补丁规则拒绝。

| 初始模型 | 同一 6 条独立验证任务 | PPO 更新 |
| --- | --- | --- |
| Qwen3-1.7B | 0/6 | 该初始验证之前为 0 |
| Qwen3-4B-Instruct-2507 | 1/6（16.7%） | 0 |

成功实例为 `agronholm__exceptiongroup.0b4f4937.func_basic__lw7117cv`。
模型依次查看项目目录和实际源码，在第 6 轮修正 `src/exceptiongroup/_exceptions.py` 的构造函数，
第 7 轮提交。唯一修改为：

```diff
-        return super().__new__(cls, __exceptions, __message)
+        return super().__new__(cls, __message, __exceptions)
```

独立评分容器中，F2P 60/60、P2P 29/29，pytest exit 0；`reference_control=false`。
这是真实模型候选补丁，未使用标准答案、Git history 或测试元数据作为模型输入。
另一题 `combine_module__a6ta2od9` 达到 F2P 40/63、P2P 26/26，但仍严格记为失败、奖励 0。
`pr_95` 没有生成补丁，缺少 `suppress` 导出导致测试导入失败、pytest exit 4；未算作成功。

初始对照引用 `capo-swe-1p7b-20260910-01` 的 step 0 验证，不能混入其训练后验证。
六个同仓库任务、单次采样只能说明观察到了成功，不能据此估计整个 SWE-bench 的性能或归因于参数量单一因素。

## 32 条训练池任务的能力诊断

运行 `ppo-eval-4b-8turn-pool32-20260910-01`，使用同一 4B 原始权重和前四卡。
将原训练池作为本次只评测入口的 validation 数据；原来的六题仅作为未使用的 train 占位输入，
保持入口要求的两个数据文件不重叠。没有用独立验证题更新模型，也没有更改原数据文件。

此运行保持评测 temperature=0.7。此前 1.7B 在 32 条训练池上的采集 temperature=1.0，
因此训练池结果用于判断能否产生正奖励，不视为严格同温度的模型对照；上方六题是同温度对照。

8 轮评测 exit 0、核验通过：**0/32**，234 次调用、16 条非空补丁、0 条补丁规则拒绝。
16 条任务主动提交、16 条耗尽轮数；没有模型参数更新。
已检查的失败包括错误重写源码，以及查阅源码后尚未完成修改便耗尽预算。

鉴于一半任务耗尽上限，继续在同一 32 条任务上做 24 turns / 32768 context 的只评测诊断，
run 为 `ppo-eval-4b-24turn-pool32-20260910-01`，仍使用 GPU 0–3、原模型、temperature=.7。
它同时增加轮数和上下文，并重新采样，不能把结果差异严格归因于单一参数。
24 轮评测 exit 0、核验通过：**0/32**，348 次调用、19 条非空补丁、2 条补丁规则拒绝。
26 条主动提交，6 条耗尽轮数，平均 10.875 轮。三题的 F2P 部分通过分别为 6/23、1/39、2/26，
对应 P2P 为 61/66、47/50、63/63；均未完整修复，终局奖励仍为零。
两条被拒绝的轨迹分别创建了根目录 `test_repr.py`、`test_subgroup.py`，触发当前
`forbidden_test_or_config_change` 规则，连同源码修改一起被拒绝。
这两题包含 agent 流程限制的影响，不能仅归因于模型；本轮未重放去掉临时测试文件后的补丁，
也没有证据将它们改计为成功。

| 4B 评测 | 交互预算 | 结果 | 调用数 | 非空补丁 | PPO 更新 |
| --- | --- | --- | ---: | ---: | ---: |
| 独立验证 6 题 | 8 轮 / 12K context | 1/6 | 39 | 4 | 0 |
| 训练池 32 题 | 8 轮 / 12K context | 0/32 | 234 | 16 | 0 |
| 同一训练池 32 题 | 24 轮 / 32K context | 0/32 | 348 | 19 | 0 |

**结论：4B 有可观察的初始修复能力，但当前训练池仍没有形成正奖励。**
不能将独立验证题中的一次成功计入训练池，或据此直接启动全量 PPO。
放宽轮数和上下文后仍全零，说明限制不只是 8 轮预算；已检查轨迹中还存在错误修改、
整文件覆盖和未完成修复便提交的问题。9B 就绪后宜先用同一任务与评分口径检查初始能力。
三次评测均已正常退出，GPU 0–3 已释放；GPU 4–7 未用于本轮实验。

## 复现与核验

在 A800 部署后的 Agent Lightning 仓库根目录执行：

```bash
# 六条独立验证任务。
AGL_TRAIN_TAG=ppo-eval-4b-my-val6 AGL_GPUS=0,1,2,3 AGL_TRAIN_PORT=18381 \
  AGL_TRAIN_MODEL=/media/ubuntu/D1/zsj/GOPD/G-OPD-main/models/Qwen3-4B-Instruct-2507 \
  SMITH_MAX_TURNS=8 SMITH_MAX_TOKENS=768 SMITH_CONTEXT=12288 \
  bash examples/multiturn_ppo/run_capo_ppo.sh --steps 1 \
  trainer.val_before_train=true trainer.val_only=true trainer.resume_mode=disable trainer.save_freq=-1

# 原 32 条训练池只做评测；必须待前一个任务释放 GPU 0–3 后运行。
AGL_TRAIN_TAG=ppo-eval-4b-my-pool32 AGL_GPUS=0,1,2,3 AGL_TRAIN_PORT=18381 \
  AGL_TRAIN_MODEL=/media/ubuntu/D1/zsj/GOPD/G-OPD-main/models/Qwen3-4B-Instruct-2507 \
  SMITH_MAX_TURNS=8 SMITH_MAX_TOKENS=768 SMITH_CONTEXT=12288 \
  bash examples/multiturn_ppo/run_capo_ppo.sh --steps 1 \
  --train-file /media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/exceptiongroup/val.jsonl \
  --val-file /media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/exceptiongroup/train.jsonl \
  data.train_batch_size=4 data.val_batch_size=32 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 critic.ppo_mini_batch_size=4 \
  trainer.val_before_train=true trainer.val_only=true trainer.resume_mode=disable trainer.save_freq=-1

python examples/multiturn_ppo/verify_eval.py \
  --run /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-eval-4b-my-val6
```

24 轮诊断复用第二条命令，更换为新的 tag，设置 `SMITH_MAX_TURNS=24`、`SMITH_CONTEXT=32768`，
并追加 `agentlightning.rollout_timeout_seconds=2400`；其他参数不变。

本轮实际 run tag 为上述各段记录的 `20260910-01` 名称，日志共同前缀为
`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-`。
`evaluation-summary.json` 保存逐题 instance ID、轨迹 ID、补丁路径、测试计数和终止原因。
核验器确认全部预期题目齐全、独立 grade 与上报奖励一致、所有调用标记 model version 0、
metrics 仅 step 0，且没有梯度更新指标、PPO audit 或训练检查点。

本地新增 `verify_eval.py` 仅核验日志；运行器单卡默认值从 GPU 4 改为 GPU 0。
没有改变 PPO 算法或本次 SWE agent 行为，没有启动 4B 训练。
