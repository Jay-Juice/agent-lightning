# Qwen3-8B：与 4B 相同六题验证集的训练前评测

用户明确要求检查此前 4B 得到 1/6 的独立验证集，而不是 32 条训练池任务。
本次已完成 Qwen3-8B 的六题评测：**0/6**；此前 Qwen3-4B-Instruct-2507 为 **1/6**。
两组均为原始模型、每题一次尝试、8 轮预算、零 PPO 更新。
此前 [8B 分层排查](SWE_8B_FRAMEWORK_AUDIT_2026-09-11.md) 的 0/32 属于训练池诊断，不能替代本结果。

## 对照条件与核验

| 项目 | 两组共同配置 |
| --- | --- |
| 数据 | exceptiongroup/val.jsonl 的同一六题，数据快照逐项相同 |
| 交互预算 | 最多 8 轮，每次最多 768 output tokens，总 context 12288 |
| 采样 | temperature=0.7、top_p=1、top_k=-1，每题一条 rollout |
| 思考模式 | enable_thinking=false |
| 工具观察上限 | 4000 字符 |
| 评分 | 独立容器运行完整 F2P+P2P，全部通过才记成功 |
| GPU | 物理 0、1、2、3 |
| 模型更新 | val_only=true、resume_mode=disable；0 次 PPO 更新 |

六题初始 messages 逐题完全一致。4B 的 39 次请求和 8B 的 45 次请求实际都使用
temperature=0.7、max_tokens=768、enable_thinking=false。
配置对比仅有模型相关路径、运行输出路径，以及权重传输 bucket 从 2048 增至 4096 MiB 的差别。
后者是让 8B 的大 embedding 张量可以完成启动传输，不改变解题预算、采样和评分。
agent、sandbox、训练器及 PPO 实现的已有来源哈希与原 4B 记录一致；通用启动器仅已有默认 GPU 变更，
两次均显式指定 GPU 0–3。新增诊断脚本不参与模型解题。

本次 `run.exit=0`，`verify_eval.py` 核验通过；45/45 次调用的完整历史、prompt token IDs、
模型响应与 agent 收到的内容一致。45 次均为 finish_reason=stop，无单次输出长度截断，
有 4 次工具观察达到字符上限。metrics 仅 step 0，model version 均为 0，无更新检查点。

## 逐题结果

表中省略共同前缀 `agronholm__exceptiongroup.0b4f4937.`。
“被拒”是原评测结果，后面的排除复现脚本诊断不回写原成绩。

| 任务后缀 | 原 4B | 本次 8B | 8B 的直接失败表现 |
| --- | --- | --- | --- |
| func_basic__lw7117cv | 成功；F2P 60/60、P2P 29/29 | 失败；0/60、29/29 | sed 未改到源码，随后五轮重复同一条报 SyntaxError 的测试命令 |
| combine_module__a6ta2od9 | 失败；40/63、26/26 | 被拒 | 仅新增 test_script/reproduce_issue.py；源码编辑未匹配 |
| combine_module__fm0c4bll | 失败；0/84、0/5 | 被拒 | 仅新增 test_script.py；源码 sed 未匹配，复现程序又未覆盖实际缺陷 |
| func_basic__4njcbj1q | 失败；0/72、17/17 | 被拒 | 仅新增 test_script.py；重复未匹配的源码编辑 |
| combine_module__mkpnzqei | 失败；0/83、0/6 | 失败；0/83、0/6 | output=[] 插在错误位置，实际 format 中仍报 NameError |
| pr_95 | 失败；0/89、0/0 | 失败；0/89、0/0 | 未修复 suppress 的公开导出，测试导入失败；还插入了一行缩进错误的 import |

8B 共 45 次调用，5 题耗尽 8 轮，1 题第 5 轮主动提交。
最终 2 条非空导出源码补丁、3 条被拒、1 条无修改；没有整题成功。

原 4B 成功的 `func_basic__lw7117cv`，修复是交换 `super().__new__` 的两个参数。
8B 第 3 轮的 sed 正则没有匹配源码，因此命令虽然返回 0，文件没有改变。
第 4–8 轮则反复在单行 Python 的分号后使用 try/except，每次均得到 SyntaxError。
该题的失败不涉及补丁规则拒绝，也没有单次生成被截断。

## 三条被拒结果的额外核查

首次重放诊断在 `test_script/reproduce_issue.py` 的文件名白名单检查处 exit 1，
不能算作完成。此前诊断只允许文件名为 test_*.py；本次将它扩展为任一路径段以 test_ 开头的 .py，
仍要求文件在 HEAD 中不存在，仍禁止排除已跟踪测试/配置，保留路径集合及源码类型检查。
**只改诊断脚本，没有更改正式 agent 的补丁导出或评分规则。**

第二次重放 exit 0：三条轨迹的命令退出码和修改文件集合均与原记录一致。
排除新建复现脚本后，三题的源码补丁均为空，重新评分仍为 **0/3**：

| 任务后缀 | 排除新建脚本后的 F2P | P2P |
| --- | --- | --- |
| combine_module__a6ta2od9 | 0/63 | 26/26 |
| combine_module__fm0c4bll | 0/84 | 0/5 |
| func_basic__4njcbj1q | 0/72 | 17/17 |

另对 fm0c4bll 的复现测试“看似成功”做了解释器检查：python、python3 和评分使用的绝对 Python
均为 testbed 的 Python 3.10.16，均导入 /testbed/src/exceptiongroup。
该测试只构造 ValueError 子异常，构造器返回仍继承 Exception 的 ExceptionGroup 子类，
因此没有检出 BaseExceptionGroup 基类缺失的问题，不是切换到另一个 Python 环境运行。

## 结论边界

原 4B 在这六题上有一次真实完整修复；本次 8B 在完全相同交互预算下没有观察到完整修复。
这不证明 8B 普遍弱于 4B：每题仅一次采样、任务只有六个、模型版本不同，且未测试 8B 的
100 轮预算或 thinking 模式。也不能把该子集结果称为完整 SWE-bench 性能。
本次没有启动正式训练；结束后 GPU 0–3 均恢复为 14 MiB、0% 使用率。

## 复现与运行产物

服务器仓库目录：`/media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main`。
使用已有部署源码，启动前核对前四卡空闲、端口可用；长任务运行在独立 screen 中。

```bash
AGL_TRAIN_TAG=ppo-eval-8b-my-val6 AGL_GPUS=0,1,2,3 AGL_TRAIN_PORT=18381 \
  AGL_TRAIN_MODEL=/media/ubuntu/D1/zsj/GOPD/G-OPD-main/models/Qwen3-8B \
  SMITH_MAX_TURNS=8 SMITH_MAX_TOKENS=768 SMITH_CONTEXT=12288 \
  bash examples/multiturn_ppo/run_capo_ppo.sh --steps 1 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  trainer.val_before_train=true trainer.val_only=true trainer.resume_mode=disable trainer.save_freq=-1
```

实际运行目录位于 `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/`：

- `training-ppo-eval-8b-8turn-val6-20260911-01/`：本次验证。
- `training-ppo-eval-4b-8turn-val6-20260910-01/`：原 4B 对照。
- `swe-rejected-8b-val6-replay-20260911-02/`：完成的三题重放；-01 为未完成诊断。

本次验证目录包含 evaluation-summary.json、validation-comparison.json、trace-integrity.json、
interpreter-check.json、完整轨迹、逐题 patch 与 grade，以及 rejected-replay-v2.log/.exit。
完整模型和实验产物留在服务器。诊断脚本的本地修改经 WSL rsync dry-run 预览后仅同步该文件，
未使用 --delete；部署后 Ruff lint/format 检查及三题实际重放通过。
