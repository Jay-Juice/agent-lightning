# P2 首次在线试验：初始验证评分分类阻断

检查时间：2026-09-18 01:28–01:35，北京时间。本文是只读排查结果；本次未修改训练代码、未启动新训练、未停止 PI。

## 实际结果

- B run：`training-capo-swe-v2-mini128-zero-4b-gpu47-20260918-01`。
- B 和 pair 的 `run.exit` 都是 1，B 于00:53退出；A 尚未启动。
- 初始验证执行470/470；69条reward=1、373条reward=0、28条reward=null。
- `metrics.jsonl` 为0字节；故障记录为 `reliability-failure-step-0.json`；没有发生在线PPO更新，也没有训练权重checkpoint。
- 69/470=14.68%是已确认成功占全部任务比例，不能当成完整有效验证分数；28条尚未完成归类。不能用69/442替代470题口径。
- GPU4–7已释放；GPU0–3的PI仍运行；磁盘约494 GiB空闲。

## 直接原因

`examples/multiturn_ppo/swe_reliability.py:49` 的 `grading_status()` 仅按退出码判断：除patch rejection外，只有0/1返回completed，其余全部requires_review。
agent对requires_review不发送reward；`episode_contract.validate_completed()`拒绝这类episode，初始验证后抛出 `ValueError: Unresolved grading status: requires_review`。

这是新增可靠性层的分类实现不完整。方案P0明确要求“候选代码引发的语法、导入、测试失败属于模型失败”，但当前只实现了对不明结果的阻断，未实现证据支持的候选失败归类。原CAPO更新循环尚未执行，不能将这次退出归因为PPO数值不稳定、Critic初始化或学习率。

| 待复核退出码 | 数量 | 已看到的证据 |
|---|---:|---|
| 4 | 23 | 17条语法/缩进错误，错误路径均与候选patch修改路径重合；6条其他收集/导入错误 |
| 3 | 2 | exceptiongroup异常格式化报错；pandas/xdist内部异常 |
| 124 | 2 | 固定600秒测试预算超时，pdfminer与exceptiongroup |
| 137 | 1 | alive-progress评分进程被杀；仅凭137不能认定OOM，需容器事件证据 |

例：oauthlib第4题候选工作区的`oauthlib/oauth1/__init__.py`有未闭合三引号；tenacity第45题候选工作区有不匹配右括号。这里的“候选工作区错误”不等于证明由候选新引入，也可能是原bug未修好。

124镜像空补丁/参考补丁审计通过，验证的是审计样本的环境可用性，并未覆盖任意候选补丁引起的失败分支。此前单测及worker恢复验收也不足以证明完整在线路径通过。

## 次要诊断问题

`reliability_metrics.rollout_diagnostics()`从reward事件取终止原因；缺reward的28条都被统计为unknown。
episode outcome其实包含这些原因：完整470条共submitted417、turn_budget50、format_errors3。
当前日志只计submitted405、turn_budget34、format_errors3、unknown28，主动提交率被低估为86.17%，按完整outcome应为88.72%。应优先读经过校验的outcome，并在reward存在时校验二者一致，不能将无奖励直接等同未知终止原因。

5169次模型调用没有记录HTTP失败或空输出；7/470条含length截断，格式终止3/470。以上是训练前行为诊断，不能说明后续优化稳定。

## 建议执行顺序

1. 固定这28份原始候选patch与任务，保留hash和原始日志，在新目录做CPU/Docker复核，不重新采样模型。对疑似运行时/资源问题使用同任务、同镜像、同预算的参考补丁对照和固定候选重放；137收集容器OOM/kill证据。
2. 补齐评分分类：有可信证据的候选源码语法/导入/测试失败记0；确认为评分基础设施失败则仅有界重试同一patch；真正不明结果仍阻断。不能把所有3/4/124/137统一放行，也不能删除这些任务或关闭可靠性校验。
3. 修正行为原因统计，并增加坏语法、导入失败、runner崩溃、候选超时/容器OOM和基础设施失败的定向回归；让已保存的470条轨迹通过完整episode contract重放，核对奖励覆盖与真实action，不能只修第一个报错。
4. 若评分源码发生变化，按现有签名规则重做对应环境审计，使用新tag重跑20步A/B。保持LR、KL、gamma/lambda等不变，A/B唯一配方差异仍为head初始化。
5. 两臂各完成20步、初始及step20完整验证，且完成首个在线checkpoint恢复验收后，再决定全量。

## 证据位置与检查方式

日志根：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/`。
上述B run内的 `checkpoints/episode-audit/val-step-0-cb3afc9fd4524b0a845eaaf128e778a3.json` 保存470条outcome；`agent/<rollout_id>/grade.json`、`grading-attempt-0/test-output.txt`、`model.patch`与`traces/<rollout_id>.json`保存逐题证据。

从Windows经 `C:\Windows\System32\OpenSSH\ssh.exe -o BatchMode=yes -l ubuntu A800` 执行只读 `tail`、`cat run.exit`、`find checkpoints`、`nvidia-smi`、`df -h`，以及stdin Python汇总上述JSON、核对测试错误路径与patch修改路径。没有改写原始评分、补丁或日志。
