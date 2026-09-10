# SWE 零奖励复核：不能归因于模型单一因素

2026-09-10，针对“为什么训练池 32 题一个都没有成功”进行离线复核。
生产代码基线为 `c989a1d`。本轮仅新增诊断脚本与记录，没有修改 PPO、agent 或奖励，
没有调用模型或使用 GPU。后续 GPU 实验仍仅允许物理 GPU 0–3。

**结论：0/32 是当前模型、任务子集和受限交互配置下的真实观测；不能据此确认只有模型能力不足。**
评分正负对照正常，保存的多轮输入与 token 完整，但任务筛选和预算与作者配置存在明显差异。
持续全零的二值奖励不足以支持直接启动全量 PPO。此前直接归因于模型的表述过强，应以本复核为准。

## 数据不是为 4B 挑选的容易题

仓库原始说明 [Coding Agent / Data Preparation](../../docs/75-example-coding-agent.md) 记载：
先用 Qwen3.5-9B 对每个候选任务尝试四次，删除四次全对的题，保留约 5000 道有对有错的题，
再加入约 1000 道四次全错的题。该描述是发布数据的总体筛选流程；没有逐题 probe 标签，
不能声称当前 32 题都属于“9B 四次全错”的部分。

本地缓存子集 manifest 的选择条件是 `All exceptiongroup tasks in the published example train/validation splits.`
即发布数据中同一个 exceptiongroup 仓库的全部 32 train / 6 val，选择原因是镜像已缓存。
这些样本相互关联，不是从整个 SWE benchmark 独立随机抽取的 32 题，也不是按 4B 可解性挑选的课程。

这里“训练池”只是数据用途。此次 4B 在评测前没有用这 32 题更新参数，不能期待其训练池准确率已高于验证集。
4B 在六题验证集的 1/6 成功也不意味着这 32 题必然成功。

## 当前短测预算与作者部署模板不同

原始代码依据为 [job-template-openai.yaml](../swe_smith/job-template-openai.yaml)、
[train_smith_agent.py](../swe_smith/train_smith_agent.py)；本地入口为
[run_smith_training.sh](run_smith_training.sh)、[smith_docker_agent.py](smith_docker_agent.py)。

| 配置 | 原仓库部署模板 / FSDP 入口 | 此前 4B 短测 | 4B 扩轮诊断 |
| --- | ---: | ---: | ---: |
| 最大交互轮数 | 100 | 8 | 24 |
| 每次生成 token 上限 | 12288 | 768 | 768 |
| 工具输出字符上限 | 6000 | 4000 | 4000 |
| 模型 context 上限 | 81920 | 12288 | 32768 |

表中作者值是当前仓库配置，不代表已核实其难度 probe 使用完全相同预算。
此前是为了短程验收采用更小预算；扩轮诊断仍保留了 768 token 的单次输出限制。
因此不能把这次结果描述成作者原配置的能力复现，也不能声称增加到 24 轮已排除了预算影响。
原 agent 的无环境变量默认值又不同，比较时应以部署模板实际覆盖值为准。

奖励口径也不同：本地固定完整 F2P+P2P、终局二值奖励；原 agent 默认 `SMITH_F2P_ONLY=1`，
部署模板还配置成功轨迹的长度惩罚。本轮没有为提高分数而更改本地口径。

## 32 题逐题正负对照

运行 [audit_swe_controls.py](audit_swe_controls.py)，每题建立独立新容器，分别评测原始缺陷代码
和回退缺陷注入后的已知正确代码，沿用生产 `grade()`、同一镜像、完整 F2P+P2P。
正确代码只供离线诊断，未传入任何模型输入。

| 对照 | 整题成功 | 审计异常 |
| --- | ---: | ---: |
| 原始缺陷代码 | 0/32 | 0 |
| 已知正确代码 | 32/32 | 0 |

这排除了“评分器无论什么补丁都打零”，但不是评分流程每个边界条件均正确的证明。
五题缺陷对照虽然启动 pytest，却在错误报告过程中崩溃，89 个测试状态全部缺失：

- `combine_file__e14uhohy`
- `combine_module__6vp5bpr2`
- `combine_module__kxbg804z`
- `combine_module__muc4qt7x`
- `func_pm_remove_assign__m5pepyrs`

任务镜像使用 Python 3.10.16 / pytest 8.3.4；pytest 的异常处理本身涉及被修改的 exceptiongroup。
例如 `task-000-bug/test-output.txt` 中，pytest 已记录两个失败，随后 traceback 格式化进入
`/testbed/src/exceptiongroup/_formatting.py`，在 `len(exc.exceptions)` 抛出 TypeError。
这使诊断反馈不完整，五题的原始 P2P 元数据也未在缺陷对照中重现。
它不能被简写成“所有测试断言都失败”，也不证明一个正确修复被误判为零；对应正确代码全部通过。

## 70 条轨迹、621 次模型调用完整性

运行 [audit_swe_trace_integrity.py](audit_swe_trace_integrity.py)，按本地轨迹重建每轮历史，
使用同一模型 tokenizer 重新套 chat template，并比较网关保存的实际 prompt token IDs。
另行核对全部 70 条轨迹的初始系统消息与原始 `SYSTEM_PROMPT` 完全相同。

| 4B 评测 | 调用数 | 完整历史一致 | prompt tokens 一致 | 返回内容一致 | 达到输出 token 上限 | 格式错误 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 轮 / 验证 6 题 | 39 | 39 | 39 | 39 | 0 | 1 |
| 8 轮 / 训练池 32 题 | 234 | 234 | 234 | 234 | 16 | 25 |
| 24 轮 / 训练池 32 题 | 348 | 348 | 348 | 348 | 11 | 41 |

两个训练池运行分别有 13、11 次格式错误发生在 `finish_reason=length` 的响应上。
其余错误包括输出零个或多个 bash block，而协议要求恰好一个。
增大输出预算可能减少截断，但部分截断响应已经出现无意义重复，不能保证增加预算就会修复任务。
两个训练池运行各有 31 次工具输出达到 4000 字符上限，被明确省略中段；这与偷偷丢历史不同，
但仍会限制模型可见信息。没有发现请求历史错接、遗漏已经保存的观察、或 tokenizer 输入身份不一致。

按最终可观察结果分组如下；分类有优先顺序，**不是互斥的因果归因**：

| 最终结果 | 8 轮训练池 | 24 轮训练池 |
| --- | ---: | ---: |
| 没有导出补丁 | 16 | 11 |
| 补丁规则拒绝 | 0 | 2 |
| 有补丁，但评分没有解析到逐项测试状态 | 10 | 15 |
| 有补丁、有测试状态，但 F2P 通过数为 0 | 5 | 1 |
| 部分 F2P 通过，整题仍失败 | 1 | 3 |
| 整题成功 | 0 | 0 |

没有测试状态包括坏补丁造成导入失败、SyntaxError，以及目标库破坏 pytest 的错误报告。
这些不能一律算作基础设施故障，也不能单凭这个分类认定模型正确。

**零终局奖励不等于零部分进展。** 8 轮的 `func_pm_remove_cond__ov98s1zs` 已达到
F2P 7/15、P2P 74/74；24 轮三题分别达到 F2P 6/23、1/39、2/26，
P2P 61/66、47/50、63/63。生产评分要求整题成功，故仍为 0。
F2P 指缺陷代码原本失败、应由修复转为通过的测试；P2P 指需要保持通过的回归测试。

## 两条被拒补丁的反事实重放

运行 [audit_swe_rejected.py](audit_swe_rejected.py)，在新容器按顺序执行保存的模型动作。
两条轨迹的命令退出码与原记录全部一致、修改路径集合一致。
仅从候选补丁中排除新建的根目录 `test_*.py`，保留源码修改，再在另一个新容器评分。
没有放行已存在的测试或配置修改；没有修改原始奖励。重放不是对原候选补丁内容逐字节一致的证明。

| 实例后缀 | 排除的新文件 | 只评源码的结果 | 实际错误 |
| --- | --- | --- | --- |
| `combine_module__nfx0d1jk` | `test_repr.py` | 失败，F2P 0/4、P2P 0/85 | 源码中写入字面量 `\\n`，触发 SyntaxError |
| `combine_module__l5god4ul` | `test_subgroup.py` | 失败，F2P 0/40、P2P 0/49 | 源码出现 `def subgroup:`，触发 SyntaxError |

因此这两条没有发现被测试文件规则隐藏的成功修复。
提示建议创建复现脚本、又禁止修改测试文件，临时脚本位置不够清楚，仍值得后续改进。

## 对训练的含义与下一步

0/32 在这种同仓库、经难度筛选、单次采样且预算受限的子集上可以发生，不能由一次零分推导真实成功率为零。
但先后两轮仍无正奖励，说明当前组合没有提供可用的成功样本，不是“正常所以继续全量训练”的信号。
这些 4B 运行没有任何 PPO 更新，故其零分不能归因于 PPO 更新把模型训坏；这也不证明所有 PPO 实现均无问题。
全零二值回报下，critic 初始化误差仍可能形成非零优势和参数更新，但更新不等于获得了成功修复的学习信号。

下一步应先保持 4B、任务、温度和完整测试评分不变，扩大生成预算，明确临时复现脚本目录，
做可区分预算因素的对照；之后再以相同设置比较 9B。
不能直接断言换大模型必然解决，也不应把所有限制同时修改后将提升归功于单一因素。
必要时从发布训练数据中另选有可观察初始成功的任务作为课程，独立验证集不用于训练。
本轮只完成原因审计，没有启动新的预算对照或全量训练。

## 日志与复现

全部日志留在 A800，不下载模型或运行产物。诊断运行前部署源代码，使用已配置的运行环境：

```bash
source /media/ubuntu/D1/zsj/agent-lightning-runtime/admin/activate-agent-lightning-d1.sh
cd /media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main
export CUDA_VISIBLE_DEVICES=
LOG_ROOT=/media/ubuntu/D1/zsj/agent-lightning-runtime/logs

# 输出路径必须是新目录；将 my-audit / my-replay 替换为新的名称。
python examples/multiturn_ppo/audit_swe_controls.py \
  --dataset /media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/exceptiongroup/train.jsonl \
  --output "$LOG_ROOT/my-audit" --workers 4

python examples/multiturn_ppo/audit_swe_trace_integrity.py \
  --run "$LOG_ROOT/training-ppo-eval-4b-8turn-val6-20260910-01" \
  --run "$LOG_ROOT/training-ppo-eval-4b-8turn-pool32-20260910-01" \
  --run "$LOG_ROOT/training-ppo-eval-4b-24turn-pool32-20260910-01" \
  --output "$LOG_ROOT/my-audit/trace-integrity.json"

python examples/multiturn_ppo/audit_swe_rejected.py \
  --run "$LOG_ROOT/training-ppo-eval-4b-24turn-pool32-20260910-01" \
  --output "$LOG_ROOT/my-replay"
```

实际正负对照与完整性结果位于 `$LOG_ROOT/swe-zero-recheck-20260910-01/`：
`summary.json`、`trace-integrity.json`、逐题 `task-*-{bug,reference}/test-output.txt`。
拒绝补丁重放位于 `$LOG_ROOT/swe-rejected-replay-20260910-01/`：
`summary.json`、逐轨迹 `source-only.patch`、`replay.json`、`test-output.txt`。
三个脚本实际运行均 exit 0，Ruff lint 与格式检查通过。
