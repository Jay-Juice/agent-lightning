# SWE PI 实现审查与双实验结果

核对时间：2026-09-14 23:56 至 2026-09-15 00:04，Asia/Shanghai。

本次只读审查运行状态、代码与既有轨迹，并运行 CPU 测试；没有修改训练源码、重启作业或改变训练参数。新增本报告及外层 research 中的只读审计脚本。

## 结论

PI 的主要接线正确：pre-action 状态通过独立 event 与逻辑调用 ID 配对；Actor 原输入保留，Critic 使用独立张量；compute_values 与 update_critic 切到 Critic view；GAE、奖励和 Actor PPO 核心不变。配置层面可以作为同训练配方的 baseline/PI 对照。

但不能称为已经完全通过设计验收：发现一处 PI 独有的上下文边界错误，以及一处实际显著削弱 PI 内容覆盖的整文件序列化问题。两者均未修改当前作业。早期结果也没有证明 PI 有收益。

## 运行身份与最新结果

| 项目 | Baseline | PI |
|---|---|---|
| 工作树 | agent-lightning-speedtrial-20260913 | agent-lightning-p1-20260914 |
| 核对本地提交 | ed12734 | f18cd5a |
| GPU | 4–7 | 0–3 |
| tag | capo-swe-pythonfull-v7-4b-gpu47-20260914-01 | capo-swe-pythonfull-v7-p1-4b-gpu03-20260914-02 |
| 已完成 PPO step（00:03:53） | 47/784 | 3/784 |
| 训练前验证 | 70/470，14.89% | 70/470，14.89% |
| step 20 验证 | 73/470，15.53% | 尚未到达 |
| step 40 验证 | 72/470，15.32% | 尚未到达 |
| 累计训练成功 | 348/1504，23.14% | 18/96，18.75% |
| 前 3 步训练成功 | 18/96，18.75% | 18/96，18.75% |

进度不同，不能拿两组累计训练成功率判定优劣。Baseline 验证相对初始仅多 2–3 条，随机采样下尚不能证明稳定提升。PI 尚无训练后的完整验证。

两组当前 run.exit 均不存在，结合活跃日志及 GPU 工作状态判断仍在运行。23:56 扫描两组当前主日志，未发现 Traceback、OOM、RuntimeError、AssertionError、ValueError；已落盘数值指标没有 NaN/Inf。

Baseline step 40 checkpoint 文件总计约 91.39 GiB；actor/critic 各有 4 份 model、4 份 optimizer、4 份 extra_state 分片。仅核验文件存在与大小，未加载恢复。PI 尚无当前正式运行的 checkpoint。

PI 旧 tag `...20260914-01` 已失败退出，不能与当前 `-02` 混合。旧 controller 日志显示 Docker 删除容器时返回 500（未收到容器退出事件），导致验证缺失奖励；这次失败不能归因于 critic PI 数值。当前 `-02` 已完成完整初始验证与真实更新。

## 配置对照

逐项比较两个 resolved-config.json，仅输出非敏感差异。模型、actor/critic 学习率、PPO/GAE、batch、microbatch、精度、offload、采样、数据、训练长度、验证及保存频率一致。环境软件版本一致，任务轮数、输出/上下文预算、观测限制、编辑与提交协议一致。

主要共同值：Qwen3-4B-Instruct-2507；6248 train / 470 val；4 epochs / 784 steps；batch/minibatch 32；microbatch 2；LR 1e-6 / 1e-5；gamma/lambda 1；KL 0.001；无 warmup；32 turns、4096 response、65536 context、32000 observation chars；32 agents；初始及每 20 步验证，每 40 步保存。

除 PI 配置和运行路径/端口外，实际差异是 checkpoint 保留数量：baseline 为 3 组，PI 为 1 组。这不改变当前优化目标，但影响后续恢复和模型选择的可用范围。

datasets.json 按每个 split 排序键规范化序列化后的 SHA256，两组完全相同：

- train（6248）：`21e9172cb45b007cdf75c81c4918e0fec3c53a05a85bf9fee4ff2f7d5aca1e74`
- validation（470）：`15a1effb0b71a9cc30bb692b0f27909bf78d5caa5c7ac31f5a985da1c6e2ad79`

这两个是运行数据 JSON 的规范化哈希，不是原始 val.jsonl 文件的字节哈希。

470/470 个任务的初始 Actor messages 完全一致。第 1 步两组实际 instance_id 集合相同（32/32）。data_id_list 是运行内生成的 UUID，不能用它判断任务是否相同。未证明全部训练步的排列或随机生成轨迹完全相同；异步随机采样下也不应期待逐 token 重现。

远端当前源文件与各自启动 provenance 一致：baseline 66 项、PI 75 项，均无差异。PI 相对 baseline 的 Git diff 没有修改 vendored CAPO loss/GAE、评分器和公共 prompt。

## 发现 1：PI 为空时仍会额外拒绝合法 Actor 上下文（P1）

位置：`agent-lightning-p1-20260914/examples/multiturn_ppo/smith_docker_agent.py:293`、`:313`。

Baseline 允许 `P + 4096 <= 65536`，即 P 最大为 61440。PI 另扣 32 safety margin，把 prompt ceiling 设为 61408；serializer 即使返回空 PI，之后仍无条件要求完整 Actor prompt 小于等于 61408。

因此 P 在 61409–61440 时，baseline 仍会采样，而 PI 会抛 `RuntimeError("Privileged Critic prompt exceeded its pre-action budget")`。这违背“优先保留完整 H，没空间就截 PI”的退化规则，并可能因缺失奖励中止整步训练。

已作最小边界复现：P=61420，baseline 接受，PI 返回空字符串后仍触发错误。当前主运行尚未观察到该异常；这是已确认的可达边界缺陷，不能说它已经影响了现有分数。

建议：完整 Actor history 合法时允许无 PI 的 Critic view；safety margin 仅约束新增 PI，不额外收紧 Actor 接受域。可以先保存 pre-action state，实际 response 得到后再按设计计算可用预算，复用同一份过去状态，不能重新采未来状态。

当前实现还使用固定 response 最大长度预留空间，而设计建议使用实际 response 长度；这是保守编码选择，不是未来信息泄漏，但会比设计少利用一部分上下文空间。

## 发现 2：整文件全收或全丢，造成明显的 PI 信息损失（P2）

位置：`agent-lightning-p1-20260914/agentlightning/privileged_state/serialize.py:20`、`:87`。

serializer 为文本变化构造完整当前文件；超预算时丢弃整条 body，没有 generic diff/hunk 或分段退化路径。Manifest 只保留路径、类型、大小，不包含内容 hash 或 mode；因此某些不同的文件状态可能形成相同的 Critic 文本。

对已落盘的 127 个训练 rollout、1533 次模型调用做只读审计（包括尚未更新的第 4 步已完成 rollout，不等于全部已经训练）：

- 326 次状态存在文本文件变化；
- 其中 207 次没有任何文件正文进入 Critic，占 63.50%；
- 仅 119/1533 次包含 `[CURRENT FILE:]`；
- 212/1533 次报告序列化截断，13.83%；
- 原始 runtime delta 非空为 0，不能据此断言采集器坏了；独立 smoke 已有 process/socket 正例。

真实例子：`src/pydicom/util/codify.py` 当前正文 17402 字符被整份省略，PI 仅 69 tokens；`readability/readability.py` 25678 字符被省略，PI 仅 64 tokens。并非每次都把 4096 token 空间用满后才丢失文件细节。

这是信息表示与覆盖不足，不能直接判为 PPO 数学错误或答案泄漏；但会削弱“Critic 看到了当前实际代码变化”的实验含义。设计允许 budget 下省略内容，因此应准确描述为当前压缩策略不足，不能夸大成全部样本缺失 PI。

建议先对保存的 S0 / delta 进行固定 1K、2K、4K、8K coverage audit，考虑通用 diff/hunk 或带明确省略标记的分段编码，再以独立版本实验比较。不要无标记地把新编码混入正在运行的 PI 曲线。

## 已核验的正确部分与局限

- 1533 次成功调用均能唯一配对 pre-action event；snapshot end、event timestamp、model-request event 顺序正确，turn/sequence 一致，落盘 gzip 原始字节哈希与 event 相同。
- 模型请求 event 本身在响应完成后记录；“采集发生在请求前”的结论同时来自 agent 源码顺序，不能仅靠响应后的 event 时间戳证明。
- 470 条验证无 privileged_state event，Actor messages 没有 PI marker。
- PI 第 1 步保存的 360 行实际训练样本：Actor input token 与对应 Gateway prompt+response 完全一致；Critic response IDs、response attention、response mask 一致；Critic position IDs 正确；有效总长不超过 65536。
- 只有 compute_values、update_critic 经 critic_batch_view；old/ref log-prob 和 Actor update 使用原 view。未另做固定轨迹、同一权重的数值 A/B 重算，因此不要声称完成了所有 logits/gradients 数值等价验收。
- CPU 测试：`tests/test_privileged_state.py`、`tests/verl/test_privileged_critic.py`、`tests/verl/test_agl_rollout_manager.py`，28 passed。运行时禁用 GPU、字节码和 pytest cache，临时文件只写 `/tmp/pi-review-20260915-001`。
- Collector 在 rollout 容器内扫描 /testbed 和该容器 /proc，普通遍历不跟随符号链接；未见主动读取 gold patch / reward / grader 的 PI 路径。上述代码和样本检查不等价于完整恶意边界攻击审计。
- 当前独立 smoke 记录证明至少完成一次真实 actor/critic 更新；此次没有完成正式 PI checkpoint 保存/恢复的验收。

## 对当前结果的解释

前 3 步两组在线修复都是 18/96。PI 的 critic loss / explained variance 暂时也没有显示优势：第 3 步 baseline 为 1.21 / -3.78，PI 为 4.99 / -11.53。但两者轨迹不同且训练太早，不能据此判定方法无效或 critic 接线错误。

现有证据支持“PI 主通路已接通、训练配方基本严格一致”，不支持“实现完全无误”或“PI 已超过 baseline”。应先处理已确认的边界问题，明确 PI 编码版本；效果比较至少要等 PI 完成第 20 步全量验证，并与 baseline 同步数结果比较。

只读审计脚本：`research/review_pi_20260914.py`。使用固定日志路径，不调用模型、不控制容器、不写远端状态。运行中的轨迹集合会增长，重复执行的计数可能不同。
