# SWE baseline V2：评分分类修复与重启验收

## 已查明的结果

原B实验`capo-swe-v2-mini128-zero-4b-gpu47-20260918-01`在初始验证后退出，在线PPO更新0步，A未启动。
问题是评分可靠性层只接受退出码0/1，没有实现方案中要求的“有证据的候选失败”归类；不能归因于PPO数值崩溃。

28份原始候选补丁均已在相同任务、镜像、测试集合、600秒、4GiB/2CPU约束下复评分：28个参考补丁全通过，28个候选全复现原退出码（23×4、2×3、2×124、1×137），没有重新采样Actor。137重放有精确容器ID和时间窗口内的Docker OOM事件；该候选将有限循环改成无限生成。

因此原始固定补丁集的完整复核结果为69成功、401失败，69/470=14.68085%。这是离线复核结果；原始run仍保持退出1、原来的28条reward=null，未改写日志或冒称完成在线更新。

复评分输出：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-v2-grading-recheck-20260918-01`，run.exit=0。
逐题记录见工作树`research/V2_FIXED_PATCH_REGRADING_2026-09-18.md`。

## 修复内容

- 普通0/1结果仍直接交付。对2/3/4/5/124/137等歧义退出，执行同任务参考对照和一次固定候选重放。
- 仅在reference全测试通过、两次candidate均失败且退出码一致、task/patch/test spec/image/资源预算一致时归为`candidate_failed`，最终奖励必须为0。
- 137要求两次候选的容器cgroup `oom_kill`计数增量；124要求真实执行达到原600秒上限，且对照与重放满足资源证据要求。不能仅凭退出码认定OOM或将未知基础设施问题记0。
- 参考失败、候选重放变成成功、两次结果不一致、证据不完整继续requires_review；没有挑选幸运重试结果。
- Docker/HTTP传输错误每阶段最多重试一次，同一任务与patch不变，不重新生成模型轨迹。
- 参考补丁只在agent交互容器关闭后由隔离grader使用，不进入模型上下文。
- 终止原因统计优先使用校验过的`swe-v2 rollout_outcome`，并核对reward reason；缺奖励不再被当成未知终止原因。
- 新增评分模块SHA纳入环境审计signature；容器标签包含episode及评分阶段，便于准确关联。

模型交互预算仍3600秒，单次评分仍600秒；外层安全超时由5400增为8400，覆盖candidate/reference/replay及每阶段最多一次传输重试：3600+6×(600+120)+300=8220秒。它不增加模型turn/token预算。Actor/Critic LR、KL、gamma/lambda、batch/minibatch及20步边界不变。

## 已完成的测试

- 实际Linux环境：112项pytest通过、26个subtests通过；覆盖受控分类、参考失败、成功重放不替换奖励、证据不匹配、未知退出、评分/动作契约、终止统计及启动配置。
- bash -n通过；生产改动Ruff检查通过（保留显式if/else的SIM108风格豁免）。Windows上的16项Linux信号测试跳过，不拿其结果冒充Linux验收。
- A/B正式入口config-only退出均0；归一化实验路径后仅`critic_head_init`有差异，外层timeout均8400。
- 全470条真实轨迹走正式triplet转换、CompletedRollout构建、episode contract及validation metrics聚合，全部通过。28条奖励使用有证据的内存审计覆盖层，所有原始trace哈希前后不变，未用于训练。

轨迹重放：`audit-v2-episode-replay-all-20260918-01`，accepted470/held0/passed=true。
5169次真实调用，1368833输出token；submitted417（88.72%）、turn_budget50、format_errors3；7题含length截断。

## 重启前最后验收

- 新版124镜像审计：`swe-full-python-envs-reliable-v2-20260918-02`，124/124通过，ready=true，run.exit=0。
- 真实生产自动分类验收：`audit-v2-grading-live-20260918-01`，已完成，summary.passed=true、run.exit=0，6/6均candidate_failed/reward0。每题调用未经替换的`grade_fixed_patch(full_python_agent.grade)`，固定candidate→reference→candidate；两个timeout均实际执行两轮600秒，OOM案例两次自身容器oom_kill增量均为1，参考通过。全部维持600秒/4GiB/2CPU。
- `run_reliable_ab_pair.sh`已增加上述live6及470轨迹重放的启动门槛，核对证据及源码SHA；原有四卡worker恢复证据继续保留。

所有门槛已通过，已启动新pair `20260918-02`，GPU4–7先B再A，各20步，初始及step20均完整470题验证。GPU0–3的PI保持运行。启动日志已确认 WORKER_ENVIRONMENT_GRADING_AND_470_EPISODE_GATES_PASSED，随后 FULL_PYTHON_READY 确认6248训练题、470验证题和124镜像检查通过。四卡worker、FSDP和推理服务已启动；日志确认Critic head在fresh模型、FSDP包装前置零。B组已进入完整470题初始验证，检查时至少5条rollout完成，真实trajectory与sandbox文件持续产生。这里的succeeded是rollout执行成功计数，不是修复成功数；尚无新PPO更新或最终修复率。

修复代码已本地提交`8e0e473`并部署；尚未push到GitHub。

## 路径与运行方式

本地权威源码：`D:\ai project\RL学习\agent-lightning-main\agent-lightning-baseline-v2-20260917`。
WSL：`/mnt/d/ai project/RL学习/agent-lightning-main/agent-lightning-baseline-v2-20260917`。
SSH：`C:\Windows\System32\OpenSSH\ssh.exe -o BatchMode=yes -l ubuntu A800`。
远端部署：`/media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917`。

部署使用`research/p2-grading-sync-files.txt`精确清单，WSL `rsync -anvz`先预览，再`-avz`；不使用--delete，不修改PI部署目录。

所有验收通过后，已在远端部署目录执行以下命令（screen启动返回0）：

```sh
screen -dmS agl-ab-v2-gpu47-20260918-02 bash examples/multiturn_ppo/run_reliable_ab_pair.sh \
  20260918-02 \
  /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-checkpoint-roundtrip-v2-20260918-03 \
  /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/swe-full-python-envs-reliable-v2-20260918-02 \
  /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-v2-grading-live-20260918-01 \
  /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-v2-episode-replay-all-20260918-01
```

## 当前实验与后续判断

- Pair：`ab-pair-20260918-02`；screen：`agl-ab-v2-gpu47-20260918-02`。
- B：`training-capo-swe-v2-mini128-zero-4b-gpu47-20260918-02`，正在初始验证。
- A：`training-capo-swe-v2-mini128-default-4b-gpu47-20260918-02`，等待B正常完成后自动启动；B失败则pair停止。
- 两组均从Qwen3-4B-Instruct-2507新启动，仅critic_head_init不同，各20步，初始与step20全470题验证。
- 关注完整评分交付、Actor/Critic数值与梯度、KL、修复率、提交率和输出长度；不能仅凭进程存活判定训练稳定。
- 四卡worker checkpoint恢复已验收；完整在线trainer/dataloader/rollout恢复仍待首个完整线上checkpoint验证。

## 2026-09-18 04:40（北京时间）定时巡检

- Pair与B进程存活，run.exit尚未产生，A尚未启动；trainer日志持续更新。实际读取远端metrics.jsonl和有界日志，未改运行中的源码、配置或进程。
- B完整初始验证已通过：470/470有轨迹及奖励，58/470=12.3404%；425题提交（90.43%），3题格式终止（0.638%），7题含长度截断，0失败/空模型响应，0基础设施重试。相比旧固定补丁离线复核69/470低11题，但这是新的随机rollout，不能归因于训练或直接证明退化。
- 已完成3/20次PPO任务批更新，各32题；训练奖励依次9/32、7/32、7/32。Actor/Critic各实际累计10次minibatch optimizer更新，跳过0次，已记录指标没有NaN/Inf。
- Actor KL loss依次0.002671、0.003961、0.007704，clip fraction约0.195%、0.248%、0.235%；Actor grad norm为5.14、3.30、8.05，Critic为14.96、5.25、9.84。Critic explained variance为0、-0.00967、0.02878，仍需后续观察，尚不能认为value已经拟合充分。
- 三批提交率90.63%、87.5%、90.63%，格式终止均0，平均输出2654、3010、3291 tokens/题，暂无明显输出塌缩证据。训练集批次奖励与初始验证不能直接比较为学习收益。
- 当前第4批31/32完成；尾部任务708fe71bd812489c857c01c724e04514约483秒前启动，约140秒前结束交互并进入grading-attempt-0，尚在单次600秒评分预算内。GPU4–7此时0%是尾部CPU评分等待的快照，不能据此认定GPU故障或训练卡死。
- 前三步总耗时36.47、23.68、25.13分钟，其中首步rollout约21.12分钟、后两步约4.62/5.15分钟；Actor+Critic更新约11.31/14.16/14.91分钟。可在A/B结束后评估微批次/调度加速，当前保持对照一致，不削减600秒评分预算。暂不外推精确结束时间。
- D1约492GiB可用；尚无global_step checkpoint（仅episode-audit）。GPU0–3的PI screen及工作进程继续运行，本轮未操作。
- 决策：继续原20步B→A，不改超参数、不重启、不启动全量。待完整step20验证与A组对照，以及线上checkpoint恢复验收后决定配方与下一阶段。
## 2026-09-18 06:12（北京时间）定时巡检

- B已完整记录5/20步，第6步Critic更新完成、Actor更新进行到64/89 forwards；pair/B无run.exit，日志在28秒内更新，A尚未启动。上轮第4批尾部评分已结束，无卡死证据。
- 新完成step4/5的训练奖励为13/32、10/32；Actor KL loss为0.01329、0.00985，clip fraction为0.296%、0.233%；Actor grad norm为3.85、2.50，Critic为18.56、12.65。已读取指标均有限，optimizer skipped仍0。
- step4/5提交率90.625%、84.375%，格式终止0/32、1/32，平均输出2802、3119 tokens/题；Critic explained variance为0.0423、0.0144，仍接近零，不能提前断言Critic已有效拟合或训练带来泛化提升。
- step4/5耗时44.60/29.71分钟。GPU4–7当前99%利用率、显存约58.8–59.5GiB；D1空闲489.39GiB。GPU0–3的PI继续运行，未操作。尚无global_step checkpoint，符合save_freq=20设置。
- 本轮只读远端日志与资源、更新本地记录；无异常需要恢复，继续同一A/B，不改超参数、不启动新run。等待step20完整验证与checkpoint后推进下一阶段。
## 2026-09-18 07:43（北京时间）定时巡检

- B已完成10/20步，正在第11批rollout（26/32完成）；pair/B screen存活、run.exit不存在，trainer日志1秒内更新，A尚未启动。没有新完整验证或checkpoint，不能判断泛化收益。
- step7–10训练奖励分别6/32、5/32、5/32、9/32；Actor KL loss为0.01900、0.01939、0.02424、0.03198，呈上升趋势，继续重点跟踪，但目前无数值发散依据。clip fraction为0.249%、0.304%、0.235%、0.318%，Actor grad norm为2.71、2.80、3.51、3.22，Critic为13.26、5.11、6.83、7.03；这四步记录均无NaN/Inf、无跳过optimizer更新。
- step7–10提交率90.625%、100%、90.625%、81.25%，格式终止0、0、1、0题；平均输出2993、2847、2191、3517 tokens/题，尚无持续长度塌缩证据。Critic explained variance约0.0670、0.0254、0.0106、0.0689，仍偏低，保留到A/B完成后判断是否调Critic配方。
- step7–10耗时21.65、21.18、19.09、25.12分钟；GPU4–7正处rollout阶段，显存约24.9–25.8GiB、利用率瞬时0–61%；D1空闲490.12GiB。GPU0–3的PI仍运行，未操作。
- 决策：继续原对照，不以不同训练批次奖励波动判定退化，不为追求较低KL立即修改正在运行的配置。等待20步同口径验证，按原计划开展A与在线恢复验收。
## 2026-09-18 09:15后：step12退出120修复与受控重启

当前权威状态：旧pair `ab-pair-20260918-02`与B均run.exit=1，约08:44退出；实际完成11次PPO更新，第12批采样结束后被episode contract拒绝。A从未启动。没有global_step checkpoint，不能恢复11步参数；旧日志、轨迹保持原状。

唯一待复核样本为`9390865148bd4f2ebf953fe8bd4a8ccb`，任务`mozillazg__python-pinyin.e42dede5.lm_rewrite__zhlzdi5y`。候选将带连字符CLI子命令改成下划线，10个F2P均失败、2个P2P通过，pytest输出完整但Python退出码为120。分类器歧义列表漏掉120，未启动参考/候选重放，留下requires_review。诊断中参考12/12通过(exit0)，同一候选再次exit120且同样10失败/2通过，复现约2秒，非超时/OOM。这次中断不是已观察到的PPO数值发散。

离线诊断目录：`audit-v2-exit120-diagnosis-20260918-01`。旧step11 Actor KL loss 0.04493、clip fraction0.372%、Actor/Critic grad norm6.40/11.84，EV0.06665，均有限且无跳过更新。KL上涨值得后续对照，但不是本次异常的直接原因。

修复为120增加受控评分分支，保持reference全通过、两次candidate相同退出码/相同任务补丁镜像预算等证据，还额外要求两次候选测试状态逐项一致、包含真实FAILED、测试ID集合与全通过参考一致。仅返回candidate_failed/reward0；成功重放、空测试、测试状态不一致与未知基础设施退出继续待复核。没有放宽奖励或跳过样本。

实际Linux回归：118 pytest passed、26 subtests passed；相关bash语法检查通过。本地源码权威；WSL rsync按`research/exit120-sync-files.txt`先dry-run、再部署7个明确文件。没有修改PI、共享环境或远端训练产物。

新A/B计划tag后缀`20260918-03`，仍GPU4–7、B→A、各20步，学习参数完全保持原样；两组保存频率统一由20改为5，仍保留2份，以免再次在首个保存点前丢掉多小时更新。完整在线恢复尚待首个checkpoint验收。

恢复入口：`research/run_v2_exit120_recovery.sh`。将先并行运行环境审计`swe-full-python-envs-reliable-v2-20260918-03`和7案例生产验收`audit-v2-grading-live-20260918-02`（原6案例+本次120）；两者均exit0后，pair再次检查源码hash及既有四卡恢复、470轨迹证据，再启动新训练。任一验收失败则不启动，禁止绕过门槛。脚本输出`recovery-v2-exit120-20260918-01/run.log`。跟踪最新恢复入口/新tag，禁止自动重启旧02。
恢复流程已实际启动：screen `agl-recovery-v2-exit120-20260918-01`，远端执行`screen -dmS agl-recovery-v2-exit120-20260918-01 bash research/run_v2_exit120_recovery.sh`返回0，两个审计都有持续输出。新增120案例生产路径已在9.96秒完成，candidate→reference→candidate结果120→0→120，最终candidate_failed/reward0、limits_valid=true、worker_exit=0。原两条timeout等案例和124环境仍在重新验收，因此此时不能称新03训练已启动。两组config-only退出均0，均20步、save_freq5、test_freq20、Actor/Critic各保留2；除名称/输出路径外仅critic_head_init不同。后续巡检先读恢复目录及03 pair，不要再次运行恢复脚本。
源码修复与此前本分支记录已推送至GitHub专用分支`experiment/swe-baseline-v2-reliability-20260917`，已验证修复提交`fcb224e`；未合并main，未上传数据、checkpoint或原始日志。

## 2026-09-18 10:56（北京时间）：验收完成，新03已训练

- 恢复流程两项前置验收均exit0：124/124环境通过；7/7真实评分案例全部candidate_failed/reward0，包含本次120和真实timeout/OOM。新pair `ab-pair-20260918-03`约09:47开始，所有启动证据门槛及6718任务分支检查通过。
- 当前B：`training-capo-swe-v2-mini128-zero-4b-gpu47-20260918-03`。完整初始验证470/470有奖励，62/470=13.1915%；提交408/470=86.81%，格式终止3/470=0.638%。这是新采样的初始性能，不能把较旧02高4题归因于训练收益。
- 第1步PPO已完成，训练奖励8/32，Actor KL loss0.001941，Actor/Critic grad norm3.06/15.37，优化器跳过0、已记录指标均有限。第2批32/32采样已完成，已进入训练计算，GPU4–7约99%利用率；日志末次写入235秒前与较长前向计算一致，未据此判卡死。
- A尚未启动，B正常完成后脚本自动接续。screen为`agl-recovery-v2-exit120-20260918-01`（它承载03 pair，并不存在另一个agl-ab-v2-gpu47-20260918-03 screen）；恢复/pair/B均无run.exit，正常运行。
- 保存周期5、保留2份的配置已生效；当前未到首个checkpoint。D1空闲489.03GiB，GPU0–3的PI仍运行，未操作。本轮没有重启、改学习参数或源码。
- 下一步等待B第5步完整checkpoint以及后续20步验证；在整个A/B运行中不争抢4–7卡做额外恢复测试，待可用窗口验证完整在线恢复，再决定全量。
## 2026-09-18 12:29（北京时间）：首个线上checkpoint写入完成

- B03已完成5/20步，进入第6批rollout，A尚未启动；pair/B持续运行，无新评分异常或退出。
- `checkpoints/global_step_5`已写入：Actor/Critic分别12个非空pt文件，各覆盖4 rank的model、optim、extra_state。`data.pt`6868B，`reliability-state.json`标记step5/epoch1，initial_output_tokens=2941.083；latest_checkpointed_iteration.txt=5。完成标记存在且训练已进入下一批，确认写入完成；尚未加载验证，不冒称全链路恢复验收通过。
- step2–5训练奖励3/32、7/32、11/32、9/32。step5 Actor KL loss0.01094，Actor/Critic grad norm8.46/7.87，Critic explained variance0.0590，optimizer skipped均0；提交率90.625%，格式终止0。前四步已记录数值均有限，暂未观察持续输出塌缩；仍等待step20完整验证判断性能。
- 保存后D1空闲384.35GiB。后续B保留两份时预计约293GiB，接近或低于A启动所需295GiB，应在交接时核查；不降低空间保护、不擅删恢复点。B本身写入下一份及轮换仍有余量。PI会改变共享磁盘余量，估算不能替代实时检查。
- GPU0–3的PI未动；本轮只读元数据和日志，未停止训练、改学习参数或运行额外GPU任务。下次继续检查B进展、checkpoint轮换与磁盘，A/B结束后再做完整在线恢复验收。
## 2026-09-18 14:01（北京时间）：B继续运行，A空间预警

- B03完成9/20步，第10批31/32完成，最后任务1b6bfbb73baf4e0d94a0717753e03aa8的trajectory在4秒内更新，仍在生成；未见卡死或新评分异常。A尚未启动。
- step7–9训练奖励4/32、6/32、7/32；Actor KL loss0.03428、0.01876、0.02619，clip fraction约0.26%；Actor grad norm4.29/3.51/2.91，Critic11.18/8.62/7.64，无跳过更新、记录数值均有限。Critic EV为-0.0723/-0.0240/0.0007，拟合能力仍弱；尚不凭三个不同批次调整配方。格式终止均0，平均输出2875/2904/2223 tokens，提交率81.25%/93.75%/93.75%。
- B step5仍完整91.39GiB。PI新增step80 checkpoint（13:20），当前PI40/80均保留。D1实际空闲303.05GiB，B第二份checkpoint写入后预计约211.66GiB，会低于A启动295GiB门槛；B自身后续正常轮换暂有空间。
- 已只读核验具体清理候选：旧baseline run `training-capo-swe-pythonfull-v7-4b-gpu47-resume40-20260915-01`下global_step_80为91.390GiB，canonical路径且无symlink；保留step120有Actor/Critic各12个非空pt及data.pt。旧run目录字符串被PI进程继承环境引用，但进一步核查step80精确路径无argv/environ/open-fd引用，不能把前者误报为step80仍在使用。
- step80在此前清理清单中曾暂时保护为历史对照，未在已删除15项中，因此已向用户请求只删除此一个历史点的授权；当前尚未获答复、没有删除。若获准，执行前必须再次核对精确路径、无符号链接、引用及保留点；若未获准，保留checkpoint并允许A空间门槛阻止启动，不能降低保护或另删PI。
- 本轮未改参数、重启、停止或删除任何运行/产物。下一轮优先检查用户对step80的答复与新B进度；审批等待不影响B正常继续。
## 2026-09-18 17:34（北京时间）：B进入step20完整验证

- B03已进入第20步后的470题验证，当前160/470 rollout执行完成；这不是修复成功数，不报告未完成集合的修复率。trainer日志持续更新，无run.exit；A尚未启动。
- metrics.jsonl最新完整训练记录为step19，因为正式入口在第20步更新后先验证、再记录整步metrics和保存checkpoint。不能将当前尚无step20记录误判为只更新了19步；最终20步指标/验证/保存仍待结束。
- step18/19训练奖励8/32、5/32，Actor KL loss0.06065/0.08917，Actor grad norm3.52/5.02、Critic8.55/9.49，EV0.1407/0.0014；已记录指标均有限且optimizer skipped为0。step19有1/32格式终止，暂无凭此单批次宣布行为崩溃的依据。
- 保留的实际权重为step10/15，各24个Actor+Critic rank文件；step5仅残留标记/dataloader，权重已正常轮换删除，不能视为完整恢复点。step20 checkpoint按代码顺序在验证后保存；评分异常会在异常分支尝试保存恢复点。
- D1空闲211.12GiB，A启动仍会受295GiB空间门槛阻止；旧baseline step80清理申请尚未收到授权，未删除任何文件。B当前验证不受此A启动门槛影响。
- 决策：等待完整验证后比较62/470初始结果，不能把训练未发散当成性能有效；不改超参数、不打断当前验证。下一轮若B结束且A因磁盘被拒绝，分别记录B退出与pair退出，不能称B训练崩溃。
## 2026-09-18 18:34（北京时间）：B正常完成20步，A因空间未启动

- B03 run.exit=0，20步训练、470题最终验证和step20保存完成。完整权重仅保留step15/20，各24个Actor/Critic rank文件，data.pt与reliability-state.json存在。
- 验证从62/470=13.1915%升至73/470=15.5319%，净增11题、2.3404个百分点。同一470个instance_id逐题核对：51题两次成功、22题新增成功、11题从成功变失败、386题两次失败、0未知奖励。这是单次随机评测的改善，不证明显著收益，也不证明零head优于默认head。
- 提交率408/470=86.81%升至434/470=92.34%；格式终止3降至2，长度截断题13降至3；平均输出2941.08降至2047.65 tokens/题（约-30.38%）。结合修复率和提交率上升，未见此前那种输出塌缩，但不能只凭输出缩短宣称能力或推理效率提升。
- 全部已记录指标无NaN/Inf；step20 Actor KL loss0.08355，clip fraction0.326%，Actor/Critic grad norm3.39/6.65，optimizer skipped均0。Critic EV0.0121，value拟合仍弱，应在A组对照及后续实验中评估。
- pair尝试启动A时被reliable_launch磁盘检查拒绝：211.57GiB<295GiB，pair/recovery因此退出1；A运行目录未创建。当前磁盘211.47GiB。必须区分B正常完成与A启动前被保护拦截，不重跑已完成B。
- 旧baseline step80清理仍未获明确答复，未删除。下一步先解决已提出的磁盘清理授权，再单独启动相同配置的A（无需重复整个pair）；完整在线恢复验收也仍未完成。尚不直接启动全量或更改学习参数。
## 2026-09-18：用户要求继续后，清理旧step80并启动A

- 用户在已明确说明step80清理方案及A启动受阻后要求“下一步是怎么做，你继续”，本轮按该方案继续。通过本地脚本`research/cleanup_old_baseline_step80.py`精确限定旧baseline step80路径，WSL rsync dry/apply部署后先执行只读dry-run，再--apply。
- dry-run核验路径canonical、无符号链接、无具体checkpoint的活动引用；step40/120、B15/20、PI40/80六个保留点均检查4 rank的model/optim/extra_state及data.pt。apply前与dry-run清单完全一致才删除；删除后六个保留点文件大小及mtime全部不变，所有日志保留。
- 已删除`training-capo-swe-pythonfull-v7-4b-gpu47-resume40-20260915-01/checkpoints/global_step_80`，98129658766B=91.3904GiB，磁盘从227057598464B增至325187407872B（约302.85GiB）。证据位于远端`cleanup-old-baseline-step80-20260918-01/{plan,result}.json`。不再将此清理记为待授权，也不扩大清理范围。
- 新增`examples/multiturn_ppo/run_reliable_a_only.sh`，复用原pair全部恢复/环境/7案例/470轨迹门槛，并额外要求同pair B run.exit=0；只运行A。源码训练/评分实现及超参数未改。bash -n通过，WSL dry-run/apply部署此一个启动脚本。
- 已实际执行screen启动，返回0；screen为`agl-ab-A-v2-gpu47-20260918-03`，外层日志`ab-arm-A-20260918-03/run.log`，A tag仍`training-capo-swe-v2-mini128-default-4b-gpu47-20260918-03`。新入口已通过WORKER_ENVIRONMENT_GRADING_AND_470_EPISODE_GATES_PASSED和空间门槛，正在全任务分支检查；尚不能仅凭screen启动称A已经更新。
- B03已结束且保持不动，不重跑旧pair；旧pair/recovery的exit1属于历史A空间拒绝。后续巡检跟踪A独立入口及A run.exit。
- A仍Qwen3-4B、20步、初始和最终470验证、save_freq5、保留2份、GPU4–7；完成后先比较两组各自前后验证及稳定性，完成线上恢复验收，再决定全量或有依据的单因素超参数调整。GPU0–3的PI全程未操作。
A启动核验已完成：全124镜像对应6718任务分支检查通过，四卡Ray/FSDP/vLLM已加载；A实际进入470题初始验证，检查时21个真实trajectory.jsonl已产生、run.exit不存在。worker1249203/1249204环境明确AGL_CRITIC_HEAD_INIT=default、CUDA_VISIBLE_DEVICES=4,5,6,7。默认head路径不会打印zero初始化专属日志，因此不能要求出现CRITIC_HEAD_INITIALIZATION零头记录。

已逐项比较A实际resolved-config.json与已完成B的实际配置，差异精确仅5项：critic_head_init及4个实验名/输出路径字段；20步、batch32、save_freq5、test_freq20、resume_mode=disable等均一致。当前尚无A初始最终分数及优化更新，不宣称A有效。90分钟巡检已增加最新接续状态，跟踪A独立screen，不重跑B或再次删除step80。
## 2026-09-18 20:47（北京时间）：A早期Critic差异与继续计划

- 只读核验A独立screen及run.exit、metrics.jsonl、trainer.log和GPU状态。A已完成2/20步，第3批32/32 rollout执行结束，正在训练计算；GPU4–7利用率99–100%，A及外层均无run.exit。GPU0–3的PI未操作。当前无A checkpoint，尚未到step5；D1空闲295.08GiB，后续继续关注保存与轮换空间，不降低保护门槛。
- A初始验证71/470=15.1064%，470/470有奖励，提交424/470、格式终止4题、平均输出2799.87 tokens/题。B初始62/470、最终73/470；验证为随机采样，初始差异不能归因于Critic，也不能直接以A初始与B最终判断训练收益。
- A前两步训练奖励均9/32，Actor KL loss为0.003035/0.006975，Actor grad norm为4.459/3.234；已记录指标全部有限，Actor/Critic optimizer skipped均0。

| 指标 | B step1 | B step2 | A step1 | A step2 |
|---|---:|---:|---:|---:|
| value MSE | 0.357654 | 0.042975 | 14.698040 | 6.263215 |
| value explained variance | 0 | -0.006027 | -132.172607 | -36.676891 |
| Critic裁剪前grad norm | 15.369 | 4.170 | 252.259 | 172.590 |
| raw advantage std | 0.47931 | 0.20658 | 3.64126 | 2.50260 |

- 默认head的早期value误差与优势噪声明显更大，符合本轮对照要检查的现象；各组rollout不同且目前仅两步，不能由此断言A最终效果更差。Critic grad norm来自clip_grad_norm_返回值，为裁剪前范数；两组critic.grad_clip=1.0，不能将252/172解释为未经裁剪更新或已经发散。A误差正在下降，继续原配置。
- 后续顺序：完成A的20步及最终470题验证，比较各自前后修复率、提交率、输出、KL及Critic趋势；空闲窗口完成在线trainer/dataloader/rollout断点恢复验收；再按证据选择延长训练的配方。如A/B性能差异仍落在随机波动范围，应补重复评测/对照，不把单次11题净增当作显著收益。当前不改学习率、KL、gamma/lambda或microbatch，不重跑B。

## 2026-09-18 23:13（北京时间）：A完成7步，首个checkpoint已保存

- A03已完整记录7/20步，正在第8步Actor更新（16/73 forwards）；screen存在、run.exit不存在，日志49秒内更新。未改运行参数、源码或进程，GPU0–3的PI未操作。
- step5/6/7训练成功数为3/32、8/32、3/32；value MSE为2.9484、2.5189、1.5603，较早期下降，但EV为-115.34、-12.17、-18.75，仍不能说Critic拟合充分。裁剪前Critic grad norm为314.81、154.06、80.86；Actor KL为0.02590、0.02129、0.01656，各步optimizer skipped均0。训练批次不同，不能把奖励波动直接当作泛化退化。
- step5/6/7提交率81.25%、100%、87.5%，格式终止1、0、0题，平均输出3793、2082、2738 tokens/题。暂无持续输出塌缩证据；最终470题验证仍未开始，A初始71/470仍是唯一完整验证结果。
- global_step_5已有Actor/Critic合计24个pt文件，合计91.36GiB，data.pt=6868B及reliability-state.json存在，训练已进入后续步骤。本轮仅检查文件元数据，未加载，完整在线恢复仍未验收。
- 磁盘空闲207.85GiB；写入第二份约91.4GiB后预计剩116.5GiB，接近后续写入保护所需余量，继续监控实际轮换及PI的共享磁盘影响，不降低门槛或扩大清理范围。
- 决策：继续A完整20步对照。B保持已完成状态，不重跑；两组结束后比较各自前后验证并完成在线恢复验收，再选择后续训练配方。

## 2026-09-19 00:45（北京时间）：A完成13步；发现PI独立退出及磁盘波动

- A03仍无run.exit，screen存活，完成13/20步，下一批32/32 rollout完成、GPU4–7约99%利用率；日志持续更新。所有已记录数值有限，最近三步optimizer skipped均0。
- step11/12/13奖励6/32、5/32、6/32；value MSE为0.82933、0.34795、0.32405，EV为-3.3166、-2.5096、-1.2807，较早期改善但仍弱。Critic裁剪前grad norm76.71、26.34、9.50；Actor KL为0.02852、0.04010、0.04159，clip fraction约0.27–0.32%。提交率87.5%、100%、96.875%，格式终止2、0、0题，平均输出2799、2141、2126 tokens；暂无持续行为塌缩证据。
- A step5和step10各有24个非空Actor/Critic pt文件（各91.36GiB权重/优化器状态）。未加载测试，完整在线恢复仍待验收。继续同配置至20步最终验证，不重跑B。
- D1空闲117.78GiB，Ray持续报告文件系统超过99%占用、若需spill可能失败。当前A仍计算，不将预警本身认定为已发生对象创建失败。余量接近写入保护要求，继续关注checkpoint轮换，未删文件或降低门槛。
- 额外只读发现GPU0–3已空闲，PI screen消失。PI tag training-capo-swe-pythonfull-v7-p2fixed-4b-gpu03-20260916-01的run.exit=1，最后metrics为step91；尾部明确下一批31成功1失败，随后ValueError: Incomplete episode fee00eea0f684005a81116cd3b4b1882: failed。该成功/失败是rollout执行状态，不是修复率。日志00:19曾显示磁盘仅9.23GB；尚未追溯失败episode，不能断言磁盘是直接原因，更不能称PPO数值发散。本任务没有停止、修改或重启PI，也不占用0–3卡。

## 2026-09-19 02:18（北京时间）：A完成17步，checkpoint轮换正常

- A03完成17/20步，第18批29/32 rollout执行完成，screen存活、run.exit不存在、日志1秒内更新；GPU4–7处于rollout阶段。尚未开始最终470题验证，无新完整修复率。
- step15/16/17奖励8/32、10/32、5/32；value MSE为0.29250、0.15539、0.21574，EV为-0.17040、-0.03663、-0.29474，较早期显著改善但仍不足以证明value有效拟合。Critic裁剪前grad norm25.38、6.31、20.81；Actor KL为0.04117、0.03779、0.04003，clip fraction约0.21–0.30%。已记录指标全部有限，最近三步optimizer skipped均0。
- 最近三步提交率90.625%、87.5%、87.5%，格式终止0、1、0题，平均输出2930、3031、3143 tokens。不同批次奖励有波动，未见持续行为塌缩，不提前判断训练收益。
- 完整权重目前step10/15，各24个Actor/Critic pt、约91.36GiB；step5权重已由保留策略正常轮换，只剩目录不能视为恢复点。D1空闲117.54GiB，与上轮接近，继续关注step20保存的瞬时占用与保护门槛，不扩大清理。GPU0–3仍空闲，本任务未操作PI或新增GPU任务。
- 继续相同A配置至20步完整验证；B保持完成状态。完整在线恢复验收安排在A结束后的空闲窗口。本轮仅SSH只读检查和记录，无源码、超参数、环境或进程变更。

## 2026-09-19 03:59（北京时间）：A最终验证完成与下一阶段建议

- A最终470/470有奖励，69/470=14.6809%，初始71/470=15.1064%，净减2题（-0.4255个百分点）。按同一val batch sample_idx配对：两次成功52、新增17、丢失19、两次失败382。B为62→73/470，净增11题（+2.3404个百分点）；最终B比A高4题（0.8511个百分点）。验证为单次随机采样且初始相差9题，不能宣布B显著优于A，也不能将两组净增差直接当作确定因果收益。
- A提交424→428/470（90.21%→91.06%），格式终止4→2，长度截断11→10，平均输出2799.87→2425.33 tokens。B提交408→434/470（86.81%→92.34%），格式终止3→2，长度截断13→3，输出2941.08→2047.65。两者均无此前持续输出塌缩的证据；B行为指标改善更明显。
- A step20 value MSE=0.212918、EV=-0.07297、Critic裁剪前grad norm=15.9218、Actor grad norm=5.6963、Actor KL loss=0.092104；B对应MSE=0.113545、EV=0.012116、Critic grad norm=6.6536、Actor grad norm=3.3879、KL=0.083551。A早期value噪声大后缓解；B避免了此启动问题，但两者value拟合仍弱，不宣称已解决长期学习稳定性。
- 最后一题28860b0f3b0b42bc9043424ee009e613经candidate timeout124→reference通过0→同candidate再次真实600秒timeout124，最终受控candidate_failed/reward0；评分完整交付，没有丢弃、放宽奖励或采样取成功。该复核是本轮验证尾部拖延原因。
- 03:59检查时A尚在保存step20（24个文件尚未全部落盘），run.exit不存在，暂不宣称进程已正常结束。等待完成后再启动任何GPU任务；不得把正在轮换中的旧目录当成新完整恢复点。
- 决策建议：优先选B零初始化配方继续，依据是更平稳的早期Critic及无明显行为退化，而不是已证明显著修复收益。当前Actor LR1e-6、Critic LR1e-5、KL损失系数0.001、gamma=lambda=1、无预热、task batch32、call minibatch128保持；观测KL约0.08–0.09不是配置系数。不要同时改学习率、KL和折扣来混淆本轮结论。
- 下一步先完成真实在线checkpoint恢复验收（模型/优化器/RNG、trainer步数、dataloader下一批、rollout权重版本及一次更新），再从B20延长训练，在累计step40/80做470题验证；稳定时继续既定全量目标784步（4epoch），总目标不因恢复重新计数。当前本来就在完整6248题训练集上采样，20步只是短程预算，不是另一套小数据集。
- 新运行前须解决空间：全量可靠入口要求295GiB空闲；当前空间不足且历史曾瞬间接近耗尽。保留A20/B20和全部证据，不降低保护门槛。具体新清理方案需按授权范围单独核对，不凭本建议直接删除A/B旧checkpoint。
- 性能优化与学习参数分开：后续可试microbatch2→4，但须保留有效task/call batch，先验长序列显存和梯度累积语义再采用；不在这次结果分析中声称已提速或已启动全量。

04:00最终核验：A run.exit=0，step15/20各24个非空pt（91.36GiB），data.pt与reliability-state.json均存在，step20/epoch1标记匹配；所有已记录数值有限、optimizer skipped均0。两组现在均已正常结束20步及完整验证。D1实际空闲116.60GiB，尚未启动恢复验收或全量训练。

## 2026-09-19：用户授权两组双四卡全量续训，启动与清理记录

- 用户明确要求A/B都从20步续训、各占四卡，并允许清理失败旧实验的checkpoint。这覆盖此前只用4–7卡的限制；A现在使用0–3，B使用4–7。启动前八卡均仅14MiB显存，无本任务旧screen，端口均空闲；不重启已失败PI。
- 已用research/cleanup_failed_for_ab_resume.py先dry-run、再--apply，仅删除旧失败baseline training-capo-swe-pythonfull-v7-4b-gpu47-20260914-01/global_step_40和旧失败PI training-capo-swe-pythonfull-v7-p2fixed-4b-gpu03-20260916-01/global_step_40（路径均在各自checkpoints下）。释放196259317916B=182.78GiB，空闲116.60→299.38GiB。旧baseline120、PI80、A15/20、B15/20六点文件大小/mtime均未变，全部日志保留；证据cleanup-failed-for-ab-resume-20260919-01/{plan,result}.json。不得重复执行清理。
- 审计安装的veRL发现previous_saved_paths为内存列表，跨进程恢复后不会自动接管之前的保存点。因此新增显式retention_across_resume，仅在新的完整Actor/Critic四rank文件、data.pt及匹配完成标记都存在后保留最近两份；保留旧目录元数据/日志。共享checkpoint_lock_path串行化两组大文件保存，进入锁后检查112GiB写前余量，避免同时写入挤占空间。PPO/GAE/奖励计算未改。
- run_full_python_ppo.sh改为尊重明确AGL_GPUS；checked入口只允许0–3或4–7完整分组；run_training.sh支持各自独立Ray目录。新源码提交017a500。本地修改后按research/ab-resume-sync-files.txt通过WSL rsync -anv预览、-av部署10个指定文件，无--delete。
- 实际CPU回归tests/verl/test_checkpoint_retention.py、test_reliability_checkpoint.py、test_reliability_control.py共13 passed；四个相关bash脚本bash -n通过。CPU恢复预检首次因未设置PYTHONPATH在导入阶段失败，无任务或产物写入；指定本部署根PYTHONPATH后成功。
- research/prepare_ab_continuation.py从A/B真实step20的data.pt构造StatefulDataLoader，均_num_yielded=20，重建的第21批32个data_id完全相同；预检证据ab-continuation-preflight-20260919-01/{A,B,completed}.json。训练器将在线核验第21批data_id及rollout_weight_version=20，输出RELIABILITY_RESUME_NEXT_BATCH_OK；目前预检通过不等于完整线上恢复验收。
- 两组config-only均exit0。与原配置比对，核心学习/采样/奖励/任务预算完全不变；改变仅恢复路径、总步数20→784、运行名/端口/审计目录，以及恢复核验/保留锁字段。Actor LR1e-6、Critic LR1e-5、KL系数.001、gamma=lambda=1、无预热、task batch32、call minibatch128、microbatch2均保留。仍每20步470题验证、每5步保存、保留2份；恢复前完整验证保持开启。
- 新日志tag：A training-capo-swe-v2-full-default-4b-gpu0123-resume20-20260919-01，B training-capo-swe-v2-full-zero-4b-gpu4567-resume20-20260919-01。新日志目录独立，checkpoint根复用各自原mini128 run的checkpoints目录，后续正常保留策略会轮换旧点，不能在新日志run/checkpoints查找。累计目标784步=4epoch，包含已有20步。
- 04:20实际执行screen -dmS agl-full-v2-A-gpu03-20260919-01和agl-full-v2-B-gpu47-20260919-01，分别运行bash research/run_ab_continuation.sh A/B，均返回0。端口18531/18541；Ray目录ray-ab-a-0919/ray-ab-b-0919。外层日志在ab-continuation-preflight-20260919-01/launch-A.log及launch-B.log。
- 04:22检查：两screen存活、已通过295GiB空间门槛，正在全124镜像/6718分支检查，训练run目录尚未创建，因此此时仅称续训已发起，不称模型恢复或step21更新成功。后续确认加载step20、模型/optimizer/RNG、恢复验证、首批数据及优化更新后才称完整在线恢复通过。
- 已更新自动巡检提示为新A/B双四卡任务与已完成清理，避免旧提示重启PI/旧pair。保留工具当前保存的ACTIVE与RRULE interval120分钟（文件实际值），未将其口头称为已验证90分钟。稳定时继续当前784步任务，不另开重复全量运行；40/80等验证节点评估趋势。

04:27启动核验完成：A/B实际provenance分别CUDA_VISIBLE_DEVICES=0,1,2,3和4,5,6,7，均记录Setting global step to 20及对应Actor/Critic model、optimizer、rng、lr_scheduler加载日志。Ray会合并重复rank日志，打印条数不能当作实际rank数量。两组均已进入恢复后的470题验证，A执行完成22/470、B2/470，各产生49/32个真实trajectory文件，run.exit均不存在；这是执行完成数，不是修复成功数。step21数据校验和首次恢复后优化更新仍待验证结束，不提前称全链路验收完成。续训已实际运行，后续跟踪新tag。

## 2026-09-19 06:46（北京时间）：修复pydicom离线下载后恢复A/B

- 06:29巡检确认首轮全量续训01两组均run.exit=1，screen消失，八卡空闲。两组已恢复step20模型、optimizer/RNG/scheduler，并通过在线 `RELIABILITY_RESUME_NEXT_BATCH_OK step=21 rollout_weight_version=20`；step21/22更新完成且数值有限、无跳过更新。第23批因评分requires_review被保护性中止，并非PPO数值发散。未保存21/22，所以本次从最近完整step20恢复，损失两次未落盘更新。
- 两组同一任务 `pydicom__pydicom.7d361b3d.pr_1920` 的原候选和官方reference都真实600秒timeout124，卡在test_fetch_data_files：官方测试先删除693_J2KR.dcm缓存，再下载全部79个数据文件；容器network=none且旧缓存不完整。无OOM，不能把未知评分记0。
- 新增pydicom_http_fixture，仅匹配固定镜像前缀和该下载测试。79份公开官方数据共59,340,134 bytes，逐一核对固定上游hashes.json的SHA256；metadata也固定哈希。容器内只对原始URL的allowlist提供回环HTTPS，临时证书信任仅给评分子进程，不启用外网、不关闭TLS验证、不改生产代码/测试/断言/原始URL及模拟网络故障分支。原始候选、参考、测试节点、600秒预算及奖励标准均保持。
- 本地编辑，通过WSL rsync先dry-run再apply同步精确源码及fixture目录，无--delete。数据作为远端运行状态且本地已gitignore，不提交GitHub。源码可复现下载脚本research/fetch_pydicom_fixtures.py保留。
- 验证：test_pydicom_http_fixture.py及test_full_python.py共31 passed；test_swe_reliability.py共34 passed（含新增fixture证据不一致必须拒绝归因），共65 passed。真实冻结补丁复核audit-pydicom-downloads-20260919-01/summary.json passed=true：A/B均46项测试有状态、F2P0/4、P2P42/42、pytest_exit1、reward0；官方参考46/46通过、exit0、reward1。测试本体约1.4秒；各候选含准备约5秒。原补丁SHA及任务一致性核验通过。
- 01恢复时的同step20随机重测A66/470、B63/470，不代表新增训练的退化；原20步最终仍为A69/470、B73/470，不能将不同随机重测直接视为因果收益。01 step22两组reward均7/32；A value MSE .27968、EV -.06599、KL .07632；B .06471、.09517、.09289。
- 06:46启动前复核验收源码哈希、端口18531/18541空闲、八卡均14MiB、磁盘297.94GiB；未新增清理、不降低295GiB门槛。第一次预检调用系统python不存在，在脚本执行前失败，无任务启动；换用已有环境绝对Python路径后预检及启动成功。
- 本轮唯一一轮训练恢复：分别 `AGL_RESUME_TAG_OVERRIDE=capo-swe-v2-full-default-4b-gpu0123-resume20-20260919-02 bash research/run_ab_continuation.sh A` 与 `AGL_RESUME_TAG_OVERRIDE=capo-swe-v2-full-zero-4b-gpu4567-resume20-20260919-02 bash research/run_ab_continuation.sh B`。screen为agl-full-v2-A-gpu03-20260919-02和agl-full-v2-B-gpu47-20260919-02，启动命令均exit0；外层日志ab-continuation-recovery-20260919-02/launch-A.log、launch-B.log及launch.json。实际训练状态随后核验，不凭screen声称更新已完成。
- 新日志tag为training-capo-swe-v2-full-default-4b-gpu0123-resume20-20260919-02及training-capo-swe-v2-full-zero-4b-gpu4567-resume20-20260919-02。checkpoint根仍在各自原mini128 run。目标累计784步=4epoch，数据6248/470、Actor LR1e-6、Critic LR1e-5、KL系数.001、gamma=lambda=1、无预热、task batch32/call minibatch128/microbatch2均未改；每20步验证、每5步保存、保留最近两份，串行保存及空间保护保持。继续等待新02恢复验证和首批核验，禁止重启失败01。

06:48实际状态纠正：上述02启动命令exit0仅表示screen创建成功。两组均在check_full_python_ready中被旧审计源码哈希拦截（Changed source: full_python_agent.py），训练run目录未创建、模型未加载、未占用GPU，因此不是一次新的训练内故障。不能称02已在训练。已保留原外层启动日志；没有覆写旧审计签名或跳过保护。

06:51启动新的全124镜像CPU审计swe-full-python-envs-reliable-v2-20260919-01，使用原run_v2_env_audit.sh（cached-only、2 worker、600秒）。audit_full_python_envs.py与check_full_python_ready.py均新增pydicom_fixture源码签名项；续训入口支持显式指定AGL_FULL_ENV_AUDIT，待新审计全部通过才执行真实训练恢复。冻结原候选验收证据依然有效（评分源码未再次变化）。

07:01完整环境审计正常exit0，ready=true，124/124镜像通过；未修改旧审计记录。随后重新核验新签名、冻结补丁验收源码hash、A/B step20各24个非空角色文件及data.pt/完成标记、两组端口、八卡均14MiB、磁盘297.91GiB，通过后各启动一次实际续训入口。明确传入AGL_FULL_ENV_AUDIT=swe-full-python-envs-reliable-v2-20260919-01（完整路径在launch-after-audit.json）。02训练tag保持，原拦截日志不覆写；新的外层日志为ab-continuation-recovery-20260919-02/launch-A-after-audit.log、launch-B-after-audit.log，启动证据launch-after-audit.json。两组screen创建exit0，后续仍需核验实际加载与运行。代码修复fd0179e已推送实验分支。

07:08实际恢复核验：两组新02 run目录均已创建、run.exit不存在，screen存活；均通过FULL_PYTHON_READY（6248训练/470验证、124镜像、6718任务分支、784步）。trainer日志确认Setting global step to 20，且Actor与Critic的model/optimizer/rng/lr_scheduler均从对应原step20加载完成，两组已进入恢复后470题验证。此时completed=0/470，不提前报告修复率或新增更新；Ray日志去重，打印条数不等于rank数。两组分别占0–3与4–7卡，各卡约24GiB加载后显存。

新02 resolved-config.json与各自原20步逐项比较，差异仅total_training_steps 20→784、resume路径/模式、运行名/端口/审计输出目录，以及首批数据/步数核验和跨恢复checkpoint保留锁字段。模型、数据、PPO/GAE/奖励、学习率、KL、task/call/microbatch、预算均一致。01已验过首次恢复后21/22实际更新；02仍须等待本次验证后在线首批标记及更新，不混淆两次状态。自动巡检已改为新02任务，禁止重复启动旧01或02；本轮未新增清理。

## 2026-09-19 09:12（北京时间）：02双组通过旧故障点，继续原配置

- 本轮Windows OpenSSH BatchMode只读检查02两组screen、trainer日志、metrics.jsonl、原mini128 checkpoint根、pydicom任务原始评分、nvidia-smi及磁盘。A/B screen均存活、run.exit均不存在，日志持续前进；八卡利用率本次快照均100%，显存约48–49GiB，D1空闲296.43GiB。未停止、重启、删checkpoint或修改源码/超参数。
- 两组均已出现RELIABILITY_RESUME_NEXT_BATCH_OK step=21 rollout_weight_version=20，预期32个data_id校验通过，且step21及后续优化更新完成：本次02实际在线恢复验收已通过，不再仅依赖01的恢复证据。
- 09:11–09:12快照：A完整metrics到step24，正在step25 Critic更新（48/81 forwards）；B完整metrics到step23，正在step24 Actor更新（64/80 forwards）。两组都已完成原先故障的第23步。A step21–24训练成功6/32、9/32、9/32、8/32；B step21–23为10/32、7/32、3/32。不同批次/rollout不能据此宣布优劣。
- 恢复前同step20完整随机验证：A74/470=15.7447%，提交429、格式终止6、平均2441.36输出tokens；B81/470=17.2340%，提交434、格式终止3、平均2387.18tokens，均470题有奖励。这是在新增更新之前对step20权重的重测，不是新训练带来的收益；历史原20步A69/B73以及01重测A66/B63保持各自口径，不混算。尚无step40新验证。
- A step24 value MSE=.309945、EV=-.076018、Actor KL=.051868、clip fraction=.002803、Actor/Critic裁剪前grad norm=3.39/26.44；B step23对应.032676、.026852、.098973、.004028、3.50/2.61。所有已记录数值有限、Actor/Critic optimizer skipped均0。A value拟合仍弱，B近期更小的误差不能独立证明最终修复能力更好。新增各步格式终止均0；A最近提交31/32、B28/32，输出未见持续塌缩。
- 此前阻塞任务pydicom__pydicom.7d361b3d.pr_1920在02真实训练中：A rollout 8295850f15674f3bb8536a285a4ed694、B 63504d57f6174a1983bec271ea5ac923，均grading_status=completed、pytest_exit=1、reward=0、46项测试有状态，原官方79文件fixture provenance一致；实际测试1.818秒/2.027秒。证明修复已在生产训练生效，而非只通过单独验收；没有把未知评分记0。
- 单步已记录耗时A约17.0–23.7分钟、B17.0–28.2分钟，Actor和Critic更新各约5–7分钟，另有rollout与logprob/value计算；当前GPU满负载，并非screen空转。保持本轮A/B可比口径，暂不混入吞吐优化或学习参数调整；step40/80再评估效果和预算。
- 本次完整checkpoint仍为两组各step15/20（每点24个非空Actor/Critic pt、91.36GiB、data.pt和完成标记存在）；step25尚未完成保存。原根reliability-failure-step-23.json来自01旧失败，不能只凭该文件判定02失败。后续重点核验新25完整落盘后轮换15、共享锁和112GiB写前保护生效；不得把残留step5/10元数据误认完整点。
- 决策：原参数继续累计784步，不新增任务、不重采样补题、不改KL/gamma/lambda或学习率。下轮检查新完整checkpoint、运行前进及step40验证。

## 2026-09-19 11:15（北京时间）：新checkpoint落盘及跨恢复轮换通过

- 本轮通过Windows OpenSSH BatchMode仅只读检查02双组screen、trainer/metrics、原checkpoint根及GPU/磁盘。A完整记录step30并进入31采样（1/32完成），B完整记录29并进入30 Actor更新（48/82 forwards）；两screen存活、run.exit均不存在，日志持续前进。未重启、停止、手动清理或改参数。
- 关键验收完成：A保存25时轮换15、保存30时轮换20；B保存25时轮换15，均有RELIABILITY_CHECKPOINT_RETENTION日志。当前完整A25/30、B20/25，每点24个非空角色pt、约91.36GiB，data.pt及匹配step的reliability-state.json存在。旧目录保留metadata，不含完整权重。只核对落盘与标记，本轮未重新加载新点。
- 注意A20已被正常保留策略轮换，不可再调用硬编码step20的旧research/run_ab_continuation.sh做未来恢复！若之后真实故障，必须重新核对每组最新完整点、dataloader和预期下一批后准备新的恢复入口；当前运行正常，不需要恢复。B20也将在完整30落盘后轮换。
- A step28/29/30训练奖励4/32、8/32、4/32，提交29/31/29，格式终止0/0/1，平均输出2811/1988/3455 tokens。A30 value MSE=.131202、EV=-.014221、KL=.094672、clip fraction=.002581、Actor/Critic裁剪前grad norm=2.04/8.49。B27/28/29奖励5/32、3/32、8/32，提交29/30/27，格式终止均0，输出2530/1979/2471；B29 MSE=.109329、EV=.011953、KL=.136762、clip fraction=.003193、grad norm=2.96/12.69。所有已记录数值有限、optimizer skipped全为0，暂无持续行为塌缩；value EV仍接近0，不能称Critic充分拟合。B近期KL约.137–.158，继续观察，尚无根据单独调参破坏A/B口径。
- 尚无step40完整验证，最近验证仍是恢复前step20的随机重测A74/B81，不解释为新增训练收益。维持原参数至step40/80节点，不能用不同训练batch的奖励波动判定优劣。
- D1空闲294.46GiB。虽低于295GiB新启动门槛，但当前任务已在运行，明显高于112GiB单次保存保护线；不因此删除额外checkpoint或降低保护。GPU0–3处于rollout，利用率56–74%、显存约25GiB；4–7处于Actor更新，利用率92–100%、约40–41GiB。正常按不同阶段变化，不误报空转。下轮核验B30及两组后续35保存与40验证。

## 2026-09-19 13:18（北京时间）：两组35步保存正常，等待40步验证

- 本轮Windows OpenSSH BatchMode只读核验：A完整metrics至35，正在36 Critic更新（32/72 forwards）；B完整至36，进入37 rollout。两screen存活、无run.exit；A日志70秒未新增时GPU仍92–96%计算，不判定卡死；B刚进入rollout准备，瞬时0%利用率不等于异常。无停止、重启、清理、源码或参数修改。
- 两组完整checkpoint均为30/35，每点24个非空角色pt、91.36GiB、data.pt和匹配step完成标记。两组20权重均已轮换，25也已由正常保存策略轮换，残留目录非恢复点。35的跨恢复retention打印removed=[]表示上游已清理旧25，并非保留失败；实际完整点仍各两份。未来恢复须从经核验的最新完整点准备新入口，严禁旧硬编码step20启动脚本。
- A33/34/35奖励13/32、7/32、5/32，提交均30/32、格式终止均0，平均输出2545/2457/3409tokens。A35 MSE=.119418、EV=.054000、Actor KL=.126549、clip=.003086、Actor/Critic裁剪前grad norm=2.48/13.05。B34/35/36奖励7/32、10/32、6/32，提交32/28/28，格式0/0/1，平均输出1633/2343/2794；B36 MSE=.057686、EV=-.057397、KL=.187322、clip=.002901、grad=4.98/5.98。所有已记录数值有限、optimizer skipped全0，无持续输出塌缩。B KL近期升至约.17–.19，跟踪40步验证及行为指标，不凭单个KL值立即改变A/B配置。
- 最新完整验证仍仅step20重测A74/B81，没有step40结果。继续原配置、保持实验可比，等完整40验证再判断收益。D1空闲294.56GiB，当前写前112GiB保护及保留两份策略可继续，无额外清理需求。

## 2026-09-19 15:20（北京时间）：step40完整验证均68/470，B下降需重点跟踪

- 02两组完成step40更新、470题完整验证及checkpoint保存，均无run.exit、screen存活。A进入41 rollout（31/32执行完成），B进入41 Critic更新（48/83 forwards）；不是训练停止。完整恢复点两组均35/40，每点24个非空角色pt、91.36GiB、data.pt及匹配完成标记。D1空闲292.60GiB，写前112GiB保护保持，不额外删除。所有已记录数值有限、optimizer skipped全0。
- 验证step20→40：A74→68/470（15.7447%→14.4681%，-6题/-1.2766pp）；B81→68/470（17.2340%→14.4681%，-13题/-2.7660pp）。这里step20指02恢复前随机重测，不能替换原实验初始或混入之前的重复评测。原20步最终A69/B73、01重测A66/B63，说明同权重随机评测本身有波动；当前仍无可靠训练收益证据。
- 读取原始trace中的is_train、model_version、instance_id和唯一reward事件配对（每组每版本470个唯一任务，任务集合完全一致）：A两次成功53、新增15、丢失21、两次失败381，精确双侧McNemar p=.4050；B56、新增12、丢失25、两次失败377，名义p=.04703。B是值得关注的下降信号，不能只说没有变化；但单次随机评测、已有多次查看且未做多重比较校正，不把该p值当作持续退化或因果结论。step40 A/B配对：共同成功49、各独有19、共同失败383，最终总体相同，不能宣布任一head胜出。
- A验证提交429→422、格式终止6→7、长度截断11→14、平均输出2441.36→2758.69tokens；B提交434→448、格式3→3、截断5→5、输出2387.18→1863.22（约-22%）。B提交率提高但修复率下降，所以不能把输出缩短当效率收益；两组尚无此前持续格式/输出塌缩证据。原始终态中无requires_review，470题奖励均完整；candidate_failed是已按协议裁定的候选失败，不是缺失评分。
- A40训练6/32、提交30/32、格式0、输出2538.81；MSE=.135902、EV=.069401、Actor KL=.188034、clip=.003793、Actor/Critic裁剪前grad norm=5.04/15.30。B40训练7/32、提交31/32、格式0、输出1671.5；MSE=.103790、EV=.140593、KL=.274788、clip=.002815、grad=6.86/9.97。B对参考模型偏离增长较快，与验证下降共同列入下一次评估，不立即把KL值等同配置系数或数值崩溃。
- 两组metrics.jsonl各有两条完全相同step40记录（逐字段相等）。源码trainer.py在验证后logger.log及保存后logger.log两处记录同一metrics，未重复优化或重新评测；分析按step去重，不把两条作为独立证据或训练次数。当前不改活跃任务源码，仅记录这个日志重复问题。
- 决策：本轮未重启、停止、删点或调参；保持两组同口径，重点检查下一次step60完整验证，不机械等待784步或仅看数值有限便认定训练有效。若60继续出现配对修复率下降，结合提交/格式/输出、参考KL和完整恢复点决定有依据的单因素干预；不要在A/B中随意单边同时修改多个超参数。当前尚不足以宣布长期学习有效，也未达到必须立即中止的行为/数值崩溃证据。

## 2026-09-19 深入诊断补充

详见主目录 SWE_BASELINE_V2_DEEP_DIAGNOSIS_2026-09-19.md（research 副本 BASELINE_V2_DEEP_DIAGNOSIS_2026-09-19.md）。17:34 快照：A 完成45、B47；最近完整验证仍为 step40，两组68/470，没有修复率提升证据。

核查两组 step32 保存张量：终局奖励正确传播到此前所有有效动作，returns 与 episode reward、raw advantage 与 reward-value 最大误差均为0。发现 reliability 成功/失败分组只按终局 call 打标的指标缺陷，已本地修复并在远端独立 admin 目录通过22项 CPU 测试及真实快照回放；没有部署到活跃训练源码，不影响 PPO reward/GAE 或验证得分。旧分组日志不可作为 episode 成败比较依据。

actor/kl_loss 为跨 minibatch 累加量，不能跨 minibatch 配方直接比较；应使用真实 token/call 均值 sampled k3。当前参考偏离持续增加，后期A/B相近，不能沿用“B远大于A”的判断。Critic 在线 EV 偏低，零初始化没有持续拟合优势。下一候选是固定轨迹、相同起点、Actor冻结，对照 Critic minibatch128/32，先验证独立episode上的拟合，再决定在线单因素试验。本轮没有新GPU实验，当前A/B继续到60验证，不热改、不重启。
