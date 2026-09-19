# SWE PPO 深入诊断：2026-09-19

## 结论与证据边界

当前没有证明baseline修复率有效提升。V2 A的正常优化配方相对旧v7主要变动为Actor和Critic调用minibatch同时32→128；B额外零初始化head。工程可靠性修复是必要条件，不应被当成学习效果进展。本次读取旧v7初跑与resume40日志、新A/B全部指标，并在CPU重新分析两组step32完整pre-update张量，没有改动或停止活跃训练。

17:34快照A完整45步、B47步；最新完整验证仍step40 A=B=68/470。02恢复前step20 A74/B81；原旧v7初跑0/20/40为70/73/72，后续resume40的60/80/100/120为69/50/60/60。旧初跑本身最后59步，126步属于resume40日志，不能遗漏续训或把两个run混为同一日志文件。样本协议与随机轨迹不同，不能用这些历史分数直接做因果比较。

## 1. 发现并修复了诊断分组错误，不是奖励传播错误

reliability_metrics.batch_diagnostics此前以每个call的token_level_scores之和是否正数判定success。二值终局奖励只放在最后一个call，所以成功episode的早期call被错误计入failure组。受影响的是reliability/success、failure及两组raw/whitened advantage统计，不影响真实reward、GAE、训练loss、验证修复率、value MSE/EV或总体KL。

本地改为按真实rollout_id汇总episode奖励，并把标签传播到同episode全部真实call；dummy行不参与、缺episode身份时不声称组统计可用。新增早期call正例、重复pad ID不污染失败轨迹、缺episode ID三类断言。Windows无torch，pytest跳过不算通过。修复仅WSL rsync dry-run/apply到独立admin/episode-metrics-audit-20260919-01目录，未覆盖活跃源码；Linux CPU 22 tests passed，并用两组实际step32张量验收通过。后续新任务部署时才能使用修正指标；当前运行日志依然是旧分组，不可继续按名字解释。

实际step32：A成功4/32，成功轨迹26个call（旧指标仅最后4个）；B成功6/32，成功轨迹38个call（旧指标仅最后6个）。按正确分组，成功call平均白化advantage分别+3.928/+3.049，失败组平均-.243/-.265，成功调用全部为正。B有约20.1%的失败call均值为正，A为0；PPO基线和白化允许这种情况，不能据此宣布符号接反。

## 2. 实际奖励传播逐元素验证通过

两组step32完整batch共各32条episode。排除dummy、按rollout_id汇总终局R后，gamma=lambda=1时每个真实action token的return应等于R，raw advantage应等于R-V。实际两式最大绝对误差均为0；成功轨迹较早call也得到正回报。故没有发现“只奖励最后提交、不把奖励传回前面”的接线错误。此证据限于保留的两个step32快照，不冒称所有批次均逐元素重放。

当前是逐token GAE，直接套gamma或lambda=.95会迅速衰减数千token之前的终局信号；不将它作为没有诊断依据的修复。

## 3. Critic在线预测弱，零head只改善冷启动

按21–40步逐batch对照“已知该批return均值的常数预测”MSE，A只有3/20批更低，B11/20批更低。这是使用当前批答案均值的事后诊断基线，不是可直接部署的oracle预测器。MSE弱可能含整体偏置，白化会消去全局偏置，不能仅凭MSE把训练无收益全归因于Critic。

31–40平均EV：A .04197，B .03882，说明两者整体上只解释很少回报变化；B没有持续保持离线诊断时的拟合优势。41后当前可用窗口A41–45 EV .04445，B41–47 EV -.04619，窗口长度不同，不作严格组间排名。

step32具体预测：A成功token value均值.5765、失败.5514，区分很弱；MSE .29864，而批均值常数MSE .05605，EV .0131。B成功.3441、失败.1925，MSE .08251、常数 .07893、EV .1347。两者偏差/拟合随batch变化，A21–30预测偏置范围-.462到+.503；不足以证明是学习率过高还是更新不足，需固定数据对照。

## 4. KL必须按同一分母比较，纠正此前解释

CAPO原dp_actor.py对actor/kl_loss与actor/pg_loss在各minibatch累加，不是整个batch的统一token均值。minibatch32→128后optimizer次数约四分之一，日志累加值也会随之变化；不能以旧KL数值几、当前零点几宣布偏离减少若干倍。grad norm是裁剪前范数，也不等于实际参数移动。clip fraction很小不证明没有学习，PPO old/new KL也不同于reference KL。

采用reliability/sampled_k3/old_vs_ref/mean（真实token均值）看本次同配置窗口：A21–30=.01770，31–40=.03073，41–45=.04964；B对应.02705、.04110、41–47=.05166。确有累积偏离，但不如原累加日志看上去那样能断言B远大于A；后期两组大致相近。该量是采样k3估计，不能把它当作精确全词表KL或梯度强度。

另发现A22 old_vs_rollout k3均值4.420被极端token主导：最大415816、p99=.01917，最大一项贡献约4.419；其余批通常约.001。同期原rollout_corr/k3=.00122。未保留22完整pre-update张量，不能精确定位到token或断言版本错配；该异常是需记录的尾部诊断风险，不将跨步均值.443解释为普遍训练/推理失配。

## 5. 当前采样与目标还有两个限制

每批仍只有32个任务、每题1条trajectory。放大调用minibatch不增加独立任务数，也不增加同题对照。step32成功任务只有A4/B6条；同条长轨迹内的很多token和call强相关，不能把几百call当成几百独立成功/失败样本。

损失为call等权，whitening为token等权；这本来就是现有CAPO适配配方，不是本轮新bug。step32成功轨迹平均约6.3–6.5call、失败约9.5–9.8call；按call等权与按episode等权不是同一目标。存在长度/任务难度权重偏差的候选解释，但一次快照不足以断言它导致退化，也不能默默改成1/T后仍声称完全相同baseline。

## 下一步：有边界的实质试验，而不是盲目扩大改动

第一优先：把Actor保守更新与Critic拟合预算拆开。当前两者都从minibatch32变128，Actor每批实际约2–4次optimizer，Critic也同样减少。先利用已有固定轨迹、同一个完整Critic状态，Actor冻结，比较Critic minibatch128 vs32，保持LR1e-5、数据pass数与其他条件相同；按完整episode划分fit/check，记录MSE相对常数、EV、偏置、耗时和实际更新次数。明确此试验改变Critic更新预算，不能同时改LR、初始化和GAE。此前离线P1用minibatch32证明能拟合，但不是当前在线minibatch128配方的充分验证。

若较小Critic minibatch使独立check有一致改善，再进入小预算在线候选：Actor128、Critic32，其他不变，使用共同已核验起点和匹配任务/采样预算，观察20步完整验证。若固定轨迹反而振荡，下一单因素改Critic LR1e-5→3e-6；不先认定“更多更新”一定正确。单做开头warmup无法自动解决40步以后持续拟合弱的问题，因此不将预热当万能修复。

第二候选：在相同Critic配方上仅reference KL .001→.005，检验“更强约束能否保留修复能力”。这是原计划D，不与Critic变化同时混入一个无法归因的run。比较正确k3、同题验证、提交/格式/输出，不用累加KL或loss量级推断梯度占比；过强约束也可能压制学习，必须验收。

暂不优先：再放大Actor调用minibatch、盲目改lambda=.95、同时换模型/算法/奖励、直接开启另一轮784步。独立任务batch增大或每题多采样是后续单独预算实验，不作为当前结论。

当前活跃A/B不热改、不打断，仍向step60检查点推进。本轮没有运行新的GPU训练、没有证明新候选提升、没有取得修复率突破；完成的是具体诊断、一个指标bug修复及其测试。下一次60必须按完整验证与重复/配对证据判断，而不是“数值有限就继续到底”。
