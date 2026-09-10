# Qwen3-8B 与框架分层排查

2026-09-11。用户最终指定使用服务器已有的 Qwen3-8B，暂停 Qwen3.5-9B 新环境配置，
并停止中途启动的 Qwen2.5-7B 试验。所有 GPU 操作仅使用物理 GPU 0–3。

**结论：确实发现了一个 8B 权重同步配置限制，已通过增大传输缓冲区解决。
跑题后的零分则没有发现由丢历史、网关改写或补丁规则漏算成功造成。
三个实际失败历史脱离 AGL/veRL/vLLM、直接交给原始 Transformers 模型后，仍生成完全相同的错误命令。**
这支持“当前模型在本任务与交互方式下的修复、纠错能力不足”，不支持“PPO 把模型训坏”。
仍不能把模型、提示、工具接口、轮数预算等因素完全分离，也不是对框架所有路径无缺陷的证明。

## 环境与中途停止的工作

- 实际 8B：`/media/ubuntu/D1/zsj/GOPD/G-OPD-main/models/Qwen3-8B`。
  五个权重分片齐全，index 中总权重字节数 16,381,470,720；`Qwen3ForCausalLM`，36 层，hidden size 4096。
- 4B 对照：`/media/ubuntu/D1/zsj/GOPD/G-OPD-main/models/Qwen3-4B-Instruct-2507`。
- 使用原 `agent-lightning-d1` 环境：PyTorch 2.9.0+cu129、Transformers 4.57.6、vLLM 0.12.0、veRL 0.7.1。
  安装尝试前后这些版本一致，未升级原环境或驱动。
- Qwen3.5-9B 分片完整，但原 Transformers 无法识别 `qwen3_5`，原 vLLM 无对应实现。
  [官方模型说明](https://huggingface.co/Qwen/Qwen3.5-9B)也提供了其专门的部署要求。
  已开始的独立 `qwen35-eval-v0171` 环境依赖安装按用户要求中止，没有用于任何评测。
  D1 上保留部分安装与下载状态；`qwen35-eval-setup-20260911/cancelled-by-user.json` 是有效状态，
  其中 `install.exit` 的 0 不能视为安装验收通过。
- Qwen2.5-7B 任务 `ppo-eval-7b-24turn-4096-pool32-20260911-01` 在模型初始化阶段按用户要求停止。
  没有 agent 轨迹或成绩；其全部带该 run tag 的进程均已退出。
- 还查到 Qwen3-14B、Qwen3-30B-A3B-Instruct-2507，但本轮未运行它们。

## 找到并修正的规模适配问题

8B 第一次启动 `ppo-eval-8b-24turn-4096-pool32-20260911-01` 在任务开始前 exit 1：

```text
Weight model.embed_tokens.weight(torch.Size([151936, 4096]), torch.float32)
is too large to fit in the bucket ... (2048 MB)
```

该张量占 2,489,319,424 字节，约 2.32 GiB，大于原 2 GiB 传输缓冲区。
veRL 的这个传输实现不把单个大张量切块，因而需要更大的 bucket。
实际代码读取的是 `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes`；
错误提示中省略了 `checkpoint_engine`，不能照抄报错中的旧配置路径。

重试只把该配置设为 4096，权重同步成功；没有改变权重、精度策略、PPO loss、GAE 或奖励。
新增 [run_swe_capacity_eval.sh](run_swe_capacity_eval.sh) 保存这一配置以及此次评测入口。
原通用训练入口没有被整体重写；以后用 8B 训练也应显式使用足够大的传输 bucket。

## 完成的同预算评测

两组均使用同一 32 条训练池任务、同一原始提示和 Docker agent、24 turns、4096 response tokens、
32768 context、4000 字符工具观察、`enable_thinking=false`、终局完整 F2P+P2P 二值评分。
原训练池通过 val-only 入口评测，六题验证集只是未使用的 train 占位输入，不进行参数更新。
实际 temperature=0.7；后端覆盖了模型各自 generation_config 的默认值，top_k=-1、top_p=1、
repetition_penalty=1。不是拿两个模型各自不同的采样默认值直接比较。

| 模型与运行后缀 | 成功 | 模型调用 | 输出 token | 主动提交 / 轮数耗尽 | 非空导出补丁 | 规则拒绝 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen3-8B，`8b-24turn-4096-pool32-20260911-02` | 0/32 | 677 | 69072 | 7 / 25 | 19 | 9 |
| Qwen3-4B-Instruct-2507，`4b-24turn-4096-pool32-20260911-01` | 0/32 | 272 | 47373 | 31 / 1 | 23 | 0 |

两组均 exit 0、`verify_eval.py` 验证通过，model version 0、metrics 仅 step 0，无 PPO 更新或检查点。
此前 4B 的 24-turn / 768-token 运行也是 0/32。
此次没有观察到扩大单次生成上限带来的整题成功；单次随机采样、并发批次差异及模型版本差异，
不允许据此给出严格统计显著性或仅参数量的因果结论。

## 消息、执行和奖励核对

两组新增共 **949 次调用**，逐项重建每轮完整历史，并使用原 tokenizer 重算 prompt token IDs：
历史、prompt token IDs、网关响应与 agent 接收内容均 **949/949 一致**。
两个运行全部 `finish_reason=stop`，此次没有单次输出长度截断。

| 观测结果 | 8B | 4B |
| --- | ---: | ---: |
| 格式错误 | 0 | 30 |
| 命令返回非零 | 227 | 14 |
| 工具观察达到字符上限 | 51 | 28 |
| 无补丁 | 4 | 9 |
| 规则拒绝 | 9 | 0 |
| 有补丁、无逐项测试状态 | 13 | 18 |
| 有补丁、有测试状态、F2P 通过数为 0 | 5 | 4 |
| 部分 F2P 通过 | 1 | 1 |

表中最后五项按最终结果分类，不是互斥的原因归因。
8B 的部分通过为 `combine_module__1ebqjjwn`：F2P 4/26、P2P 60/63；
4B 的部分通过为 `combine_module__nfx0d1jk`：F2P 1/4、P2P 85/85。
两者均未满足整题修复标准。

8B 有 366 次“某命令此前出现过”；重复运行测试本身可能合理，不能全算无效循环。
更严格地统计相邻两轮 **action 与 output 都相同**，得到 126 次，涉及 16 道题。
部分任务连续 18 次重复相同失败命令。其反馈明确包括：

- 在 `python -c` 中把 `def`、`try`、`with` 放在分号之后，导致 SyntaxError。
- 反复用同一条 `sed` 写入未闭合字符串，每次得到相同 SyntaxError。
- 错误改写代码造成缩进错误，仍未完成修复便提交或耗尽轮数。

8B 的评分输出中，末尾可识别异常包含 5 个 SyntaxError、7 个 IndentationError。
这些是实际坏补丁导致的错误，不能解释成评分器没有启动或框架基础设施统一失败。

评分正确性还参考 [前次逐题正负对照](SWE_ZERO_REWARD_RECHECK_2026-09-10.md)：
32/32 已知正确代码通过、0/32 原始缺陷代码通过。本轮未改评分实现、镜像或原数据。
exceptiongroup 缺陷可能影响 pytest 的错误报告，这个已知反馈局限仍然存在。

## 被规则拒绝的 9 条补丁

通过 [audit_swe_rejected.py](audit_swe_rejected.py) 重放保存动作，九条轨迹的命令退出码均与原记录一致，
修改文件集合一致。仅从补丁中排除新建 `test_*.py` 复现文件，仍不放行已跟踪测试/配置的修改，
然后在新容器中按完整测试评分。

**只评源码仍为 0/9**。其中 `combine_module__nfx0d1jk` 达到 F2P 1/4、P2P 85/85，
`combine_module__l5god4ul` 达到 F2P 5/40、P2P 37/49；均未整题成功。
没有发现被规则隐藏的正确修复。该反事实诊断不修改原成绩，也不证明重放补丁与原候选逐字节一致。
本轮仅扩展诊断脚本，允许排除新建子目录中的复现测试；生产 agent 的规则不变。

## 脱离框架的真实模型对照

### 网关与初始生成

[audit_live_transport.py](audit_live_transport.py) 在独立 CPU 进程用 TestClient 执行真实生产网关路由，
后端连接正在运行的 8B vLLM 服务。三个真实 SWE 初始提示各做直接调用、直接重复调用、经过网关调用。
三组网关输出和 prompt token IDs 都与直接调用一致，网关保存事件也一致。
这没有在正在运行的评测网关创建额外任务；额外模型探针不计入 SWE 分数。

其中一个直接重复调用发生输出差异，因此脚本的严格重复一致性断言 **exit 1**，
不能把整个探针称为通过。后续用原始 Transformers / BF16 / SDPA 对照：
三个输入 token 都一致，一条生成逐 token 一致，另外两条分别在公共前缀 21、28 tokens 后分歧。
首个分歧位置下，vLLM 选词在 HF teacher-forced logits 中分别是并列第一、第二，
距 HF 最优 logit 分别为 0、0.25。一个提示的直接 vLLM 重复调用本身也在 28 tokens 后分歧。
这与相近候选在不同数值路径下分叉相符，不是证明权重损坏的证据；仅检查了少量提示，不能推广成全部数值行为一致。

### 直接复现真实失败历史

更有针对性的 [audit_hf_failure_history.py](audit_hf_failure_history.py) 使用原始 8B 权重，
完全绕过 AGL、veRL、vLLM，把三个循环任务最后一轮的完整历史直接输入 Transformers。
每条都有 48 条历史消息；prompt tokens 分别为 7381、6124、9214，均与实际失败调用完全相同。

| 任务后缀 | 独立 HF 是否给出同一条错误 action |
| --- | --- |
| `combine_file__f9ib0lv6` | 是，仍在单行 Python 的分号后定义函数和 try/with |
| `lm_rewrite__oqs13swc` | 是，仍在分号后使用 with |
| `func_basic__l78q257t` | 是，仍用相同 sed 写入未闭合字符串 |

**3/3 再次生成完全相同的错误 action**，脚本 exit 0。
HF 采用 greedy，原评测 temperature=0.7；这是一项针对失败历史的复现，不是新的整题成功率测量。
它表明已进入这些历史状态后，错误循环可以由原模型本身产生，不依赖框架破坏输入或输出。
尚未测试从第一轮开始、整个 SWE episode 都用独立 HF 的成功率。

## 对后续工作的判断

本轮没有找到把正确补丁统一判零或把正常模型输出改坏的框架错误。
8B 在当前条件下主要暴露了纠错和执行策略问题；只把 4B 换成 8B 不足以让此训练池形成正奖励。
单次输出上限已经不构成这两轮的截断，但 8B 有 25/32 题耗尽 24 轮，不能排除更长轮数的作用。
同时，16 题已存在连续无进展循环，不能保证只加轮数就能解决。

继续全量二值 PPO 仍缺少成功样本依据。应先做 agent 使用方式/思考模式/轮数的独立对照，
或者从发布训练数据选择有可观察初始成功的任务作为课程；保持独立验证集不参与训练。
本轮未更改 thinking 模式、提示、工具执行或奖励，也未启动正式训练。

## 复现与产物

在已部署的仓库根目录，依次运行，前一任务释放 GPU 0–3 后再启动后一任务：

```bash
AGL_GPUS=0,1,2,3 AGL_TRAIN_PORT=18381 AGL_TRAIN_TAG=capacity-8b-unique \
  AGL_TRAIN_MODEL=/media/ubuntu/D1/zsj/GOPD/G-OPD-main/models/Qwen3-8B \
  bash examples/multiturn_ppo/run_swe_capacity_eval.sh

AGL_GPUS=0,1,2,3 AGL_TRAIN_PORT=18381 AGL_TRAIN_TAG=capacity-4b-unique \
  AGL_TRAIN_MODEL=/media/ubuntu/D1/zsj/GOPD/G-OPD-main/models/Qwen3-4B-Instruct-2507 \
  bash examples/multiturn_ppo/run_swe_capacity_eval.sh
```

实际日志共同根目录为 `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/`：

- `training-ppo-eval-8b-24turn-4096-pool32-20260911-02/`
- `training-ppo-eval-4b-24turn-4096-pool32-20260911-01/`
- `swe-rejected-8b-replay-20260911-01/`

两个评测目录均有 `evaluation-summary.json`、`trace-integrity.json`、配置、provenance、完整轨迹和逐题 grade。
8B 目录另有 `live-transport.json`、`hf-generation.json`、`hf-generation-v2.json`、`hf-failure-history.json`。
尝试追加 vLLM logit 探针时服务已退出，`logit-probes.json` 为空，不作为任何数值结论依据。
所有模型权重、数据和完整运行产物留在服务器。

CPU 回归检查命令：

```bash
CUDA_VISIBLE_DEVICES= pytest -q tests/verl/test_multi_turn_ppo.py tests/verl/test_capo_ppo.py \
  tests/server/test_triplet_preserve.py tests/examples/test_swe_smith_agent.py
```

结果 44 passed；诊断 Python 文件 Ruff lint/格式与启动器 Bash 语法检查通过。
这些检查验证其各自覆盖的路径，不等于全项目正确性证明。
