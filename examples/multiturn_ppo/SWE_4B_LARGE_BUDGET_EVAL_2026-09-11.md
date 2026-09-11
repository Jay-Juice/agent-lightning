# Qwen3-4B：放宽交互预算的训练前对照

用户要求继续使用已有 Qwen3-4B-Instruct-2507，以更大的预算检查初始修复能力。
本次先评测原六题独立验证集，再评测原 32 条训练池；两者均 val-only，每题一次尝试，
不执行 PPO 更新。所有 GPU 工作仅使用物理 GPU 0–3。

**最终结果：验证集 0/6，训练池 0/32，两次运行均正常退出并通过核验。**
放宽上限没有带来完整修复；这不证明模型完全没有能力，也不构成预算对性能的统计因果结论。
本轮还核实了至少一题的描述没有覆盖全部注入缺陷，详见下文。

## 预算与实施

| 配置 | 原验证集 | 原训练池最近一次 | 本次两组 |
| --- | ---: | ---: | ---: |
| 每题最多交互轮数 | 8 | 24 | 100 |
| 单次输出 token 上限 | 768 | 4096 | 12288 |
| 总上下文 token 上限 | 12288 | 32768 | 81920 |
| 工具反馈字符上限 | 4000 | 4000 | 6000 |
| temperature | 0.7 | 0.7 | 0.7 |
| 每题独立尝试 | 1 | 1 | 1 |
| thinking | 关闭 | 关闭 | 关闭 |

本次四项交互上限对齐本地上游 SWE-smith 部署模板及训练配置；并不声称完整复现作者所有设置。
原始模型权重、初始提示、动作解析、补丁限制、评分容器及完整 F2P+P2P 二值标准保留。
同时增加多个上限且重新采样，不能把结果差异归因于单一参数，也不能把单次采样波动称为显著变化。
100 是单题交互上限，不是 PPO 更新次数；模型主动提交或触及上下文上限仍会结束。

新增 `run_swe_large_budget_eval.sh` 支持 validation / train-pool 两种只评测入口。
`smith_docker_agent.py` 将原来固定的观察上限与请求等待时间改为可配置，未设置时仍为 4000 字符、240 秒。
本次请求等待上限 600 秒，rollout 等待上限 7200 秒；vLLM 显存比例从 0.3 提高到 0.5，以容纳更长历史。
权重传输 bucket 使用 4096 MiB。未改变模型精度、PPO loss、学习率、KL 或奖励。
`train.py` 记录新增环境参数；输入审计读取实际观察上限，并按原始输出超过该上限统计截断。

本地编辑后通过 WSL rsync dry-run 预览，再仅同步四个代码/脚本文件，未使用 --delete，未同步 models/。
部署后的 Ruff lint/format、Bash 语法检查通过，44 项已有多轮/PPO、消息保存与 SWE 回归检查通过。

## 已完成的六题验证结果

原 8 轮验证为 **1/6**，本次为 **0/6**。六题全部在第 4–8 轮主动提交，
共 39 次调用，单次输出最长 405 tokens，最长 prompt 4585 tokens。
没有任何题耗尽轮数、上下文或单次生成额度；不能用这些上限提前截断来解释本次六题失败。
初始消息与原 4B 六题逐题一致，数据快照相同；实际 39 次请求均为 temperature=0.7、
max_tokens=12288、enable_thinking=false。36 条有动作的观察与 6000 字符渲染结果完全一致。

消息历史、prompt token IDs、返回内容均 39/39 一致，finish_reason 全部为 stop。
有 3 次动作格式错误，分别返回多个 bash 块；5 次工具输出超过观察上限。
最终为 2 条非空导出补丁、1 条被拒、3 条无补丁。唯一部分修复为
`combine_module__a6ta2od9`：F2P 40/63、P2P 26/26，与旧验证的部分通过数相同。

原来成功的 `func_basic__lw7117cv`：本次模型能指出构造参数顺序错误，
但首次编辑输出了两个 bash 块而被拒，随后 sed 未匹配源码，却因退出码为 0 而认定修复成功并提交。
最终源码未修改，F2P 0/60、P2P 29/29。这里表现为执行及验证失败，不能说模型连问题都无法定位。

被拒的 `combine_module__fm0c4bll` 新建 test_exception_group.py，同时改了源码。
重放所有动作、仅排除新建复现脚本后，仍为 F2P 0/84、P2P 0/5、未解决；
命令退出码与修改文件集合一致。原成绩不变，没有把正确修复隐藏在拒绝记录中。

## 32 条训练池

最终 **0/32**，与原 24 轮 / 4096 输出的 0/32 相同；共 269 次调用、123990 个输出 tokens。
31 题主动提交，1 题因上下文预算结束；调用次数为 5–23，10 题超过 8 轮，**没有题超过 24 轮**。
最终 24 条非空导出补丁、1 条被拒、7 条无补丁；有 3 题部分 F2P 通过。

最后完成的是 `pr_69`：前 3 轮查找并读取 `_catch.py`，第 4–9 轮连续重复生成转义表达式，
每次都用满 12288 tokens，缺少完整动作而收到 length 格式反馈。第 10 次循环前因历史加预留输出
超过上下文上限而停止，因此是 **9 次模型调用**，不是完成了 10 轮生成。
该题输出 74093 tokens，占训练池总输出约 59.8%，源码没有改变，F2P 0/3、P2P 86/86。
它解释了末尾长时间等待，也说明新增输出额度可能被无效重复消耗。

269/269 次调用的完整历史、prompt token IDs、响应内容核对一致。
263 次 finish_reason=stop、6 次 length；后者均来自 pr_69。共 25 次格式错误，其中 19 次没有输出截断。
实际最长请求为 64407 prompt tokens，27 次工具输出超过 6000 字符。
244 条有动作的工具观察与 6000 字符渲染结果一致；32 题初始消息和数据快照与旧训练池评测相同。

已完成任务中观察到三个部分修复：

| 任务后缀 | 本次 F2P | 本次 P2P | 上次 24 轮 / 4096 输出的 F2P |
| --- | --- | --- | --- |
| combine_module__kgxoavlo | 2/39 | 50/50 | 0/39 |
| combine_module__3hhws1dx | 24/28 | 61/61 | 0/28 |
| combine_module__1ebqjjwn | 2/26 | 63/63 | 0/26 |

3hhws1dx 的补丁为 ExceptionGroup 增加 exceptions 属性，修复了部分行为；
剩余四个失败测试涉及 BaseExceptionGroup 的拆分及旧式 format_exception_only 调用，仍未完整修复。
这些是局部改善的观测，不能由单次随机采样推断为增加预算的因果收益。

### 额外发现：至少一题的描述未覆盖全部注入缺陷

对 3hhws1dx 的可信离线检查发现，问题描述仅说明 BaseExceptionGroup.exceptions 返回顺序反转，
但原始 Bug Patch 实际包含两处独立修改：

1. `_exceptions.py` 将 `tuple(self._exceptions)` 改为 `tuple(reversed(self._exceptions))`。
2. `_formatting.py` 将旧式 format_exception_only 包装器的 `format_exception_only(value)` 改为
   `format_exception_only(__exc)`，导致参数类型错误；描述未提及这一修改。

模型只在 ExceptionGroup 子类上覆盖属性，因而还漏掉 BaseExceptionGroup 的两项相关测试；
另外两项格式化测试则对应描述未提到的独立缺陷。因此不能说模型已经完整修好了描述中的问题，
也不能把该题的四项失败都解释为模型没有理解异常顺序问题。
这是已核实的一题描述覆盖不全，不证明其他 31 题均存在同样问题，更不证明评分器错误。
原始缺陷差异仅在单独诊断容器读取，保存在 description-bug-audit.patch；没有提供给 actor，
没有人工修改模型候选、数据、原始成绩或奖励规则。

训练池唯一被拒补丁为 `func_pm_ctrl_invert_if__pwlyksge`。
重放并仅排除新建复现脚本后仍为 F2P 0/9、P2P 0/80、未解决；
重放的退出码和修改文件集合一致，原奖励未更改。

## 判断与验收

本次验证集 6 题及训练池 31 题在预算内主动提交；其余一题被无效长输出耗尽上下文。
所以这轮没有支持“只需增加交互轮数就能恢复成功”的证据。
可观察问题包括源码替换未匹配却提交、错误整文件覆盖、多个动作块、重复生成，以及至少一题的描述覆盖不全。
后续应将动作执行与提交核验、样本描述质量、重复采样分别检查，不能继续把所有失败笼统归于模型参数量。

两次 `run.exit=0`，`verify_eval.py` 均通过，所有请求 model version=0，metrics 仅 step 0，
没有 PPO 梯度更新、PPO audit 张量或更新检查点。总计 308 次调用的输入/输出审计全部通过。
没有启动正式训练。结束后物理 GPU 0–3 均为 17 MiB、0% 使用率；GPU 4–7 未使用。

## 复现与产物

在服务器仓库 `/media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main` 中依次执行，
两次运行之间确认前一次已退出并释放 GPU 0–3。每次使用新的 tag：

```bash
AGL_TRAIN_TAG=large-4b-val-unique AGL_GPUS=0,1,2,3 AGL_TRAIN_PORT=18381 \
  bash examples/multiturn_ppo/run_swe_large_budget_eval.sh validation

AGL_TRAIN_TAG=large-4b-pool-unique AGL_GPUS=0,1,2,3 AGL_TRAIN_PORT=18381 \
  bash examples/multiturn_ppo/run_swe_large_budget_eval.sh train-pool
```

实际运行目录位于 `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/`：

- `training-ppo-eval-4b-100turn-12k-val6-20260911-01/`
- `training-ppo-eval-4b-100turn-12k-pool32-20260911-01/`
- `swe-rejected-4b-100turn-val6-20260911-01/`
- `swe-rejected-4b-100turn-pool32-20260911-01/`

验证目录包含配置、provenance、完整轨迹、候选补丁、逐题 grade、evaluation-summary.json、
trace-integrity.json 和 budget-diagnostics.json；两组最终核验文件均已生成。
完整权重、数据和运行产物保留在服务器。
