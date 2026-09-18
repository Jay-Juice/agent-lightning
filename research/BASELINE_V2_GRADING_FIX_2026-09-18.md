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
