# V2 初始验证异常：固定补丁复评分

## 结论

`audit-v2-grading-recheck-20260918-01` 已完成，`run.exit=0`，耗时807.46秒。28条任务的参考补丁全部通过；28份原候选补丁均复现原退出码，未新增成功任务。退出码分布：23条4、2条3、2条124、1条137。

本次只做CPU/Docker固定补丁复评分，没有模型调用，没有GPU占用，没有修改原始日志或reward。并发2、测试预算600秒、容器4GiB/2CPU/pids256/networknone。每条先reference、后candidate。原始470条中69成功、373失败、28待复核；现在有对照证据支持把28条归为候选失败，完整分母下成功率为69/470=14.68085%。是否写入新的回放产物由主流程的修订分类器处理，本审计不改原始产物。

这是离线复核结论。原始在线run仍是初始验证后退出、PPO更新0步；不能将本审计成功描述为原始run已成功完成，亦不能视为新版在线训练验收通过。

## 证据和验收

- 原始run：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-v2-mini128-zero-4b-gpu47-20260918-01`。
- 新审计：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-v2-grading-recheck-20260918-01`。
- `manifest.json`固定28条trace/task/candidate patch hash；每条patch hash均与原grade匹配；trace rollout_id与validation身份均校验。
- 每条`case-XX-<rollout_id>`保留`task.json`、`candidate.patch`、`original-grade.json`、`episode.json`、`result.json`。
- 每leg保留完整`grade.json`、`test-output.txt`、`container-evidence.json`；后者含精确container id、容器inspect、限定since/until的Docker事件。
- 最终逐条核验：参考和候选的镜像ID、baseline HEAD、评分协议、测试数量、600秒预算、内存/CPU与原grade全部一致；reference标记、container id与事件一致，事件查询无错误；一致性错误数0。
- `summary.json` SHA256：`a0038a481f540f731bc4ed0ef51b1dc9242dc6de2845f0c14679ce7230de8360`。
- import时grader `full_python_agent.py` SHA256：`4ff2f9e76e9afd7240ebb71c61c544bb887fe0443bf55aa024b007f089cf9b3e`。4个相关模块保留于`source-snapshot`。运行期间主流程的后续源码修改不影响已import的本次grader。

## 超时与OOM

pdfminer的reference在6.74秒通过，候选601.91秒退出124；exceptiongroup的reference在1.94秒通过，候选601.71秒退出124。时长包含准备和清理，测试子进程仍按600秒限制。

alive-progress的reference在3.03秒通过，候选100.78秒退出137。精确容器`a0409702460fe5b9c99ae68c3d174b991a572d013111cd98525dbe09e0e94424`记录1条Docker `oom`事件，事件时间位于leg开始结束之间，inspect同时显示`OOMKilled=true`。此时init仍为`Running=true, Status=running, ExitCode=0`，被杀的是测试进程。本次能够证明固定补丁重放发生OOM，不声称找回了旧容器的OOM事件。

对应候选把有限的`zip(range(gap - block_size), infinite_ribbon)`改成`enumerate(infinite_ribbon)`，仅以`if i >= gap - block_size: yield fill`筛选且不终止，与无限生成导致OOM吻合。

## 28条紧凑结果

全部reference reward=1；全部candidate reward=0；`原/重放`为测试退出码。

| 序号 | instance_id | 原/重放 | reference秒 | candidate秒 |
|---:|---|---:|---:|---:|
|0|oauthlib__oauthlib.1fd52536.combine_file__6f9y9ztr|4/4|2.13|2.36|
|1|jd__tenacity.0d40e76f.func_pm_class_rm_funcs__ayw9xqf7|4/4|4.36|2.09|
|2|pdfminer__pdfminer.six.1a8bd2f7.combine_file__sd1ohtqw|4/4|6.53|2.76|
|3|pdfminer__pdfminer.six.1a8bd2f7.func_pm_remove_loop__qkhpk22t|4/4|6.55|2.45|
|4|pdfminer__pdfminer.six.1a8bd2f7.func_pm_class_rm_funcs__r593fw3y|4/4|6.73|2.54|
|5|oauthlib__oauthlib.1fd52536.lm_rewrite__tmoxygfi|4/4|2.23|2.38|
|6|agronholm__exceptiongroup.0b4f4937.func_basic__4njcbj1q|3/3|1.99|1.76|
|7|PyCQA__flake8.cf1542ce.lm_rewrite__9babnl6b|4/4|2.89|2.07|
|8|agronholm__exceptiongroup.0b4f4937.pr_95|4/4|2.07|1.80|
|9|Knio__dominate.9082227e.pr_187|4/4|1.98|1.86|
|10|bottlepy__bottle.a8dfef30.func_pm_ctrl_shuffle__2vcdggd7|4/4|2.27|2.42|
|11|bottlepy__bottle.a8dfef30.func_basic__ybr4xtnm|4/4|2.25|2.27|
|12|pdfminer__pdfminer.six.1a8bd2f7.func_basic__5l4ptru2|124/124|6.74|601.91|
|13|rsalmei__alive-progress.35853799.combine_file__rsy05m30|137/137|3.03|100.78|
|14|lepture__mistune.bf54ef67.combine_file__xqpha4me|4/4|2.01|1.85|
|15|lepture__mistune.bf54ef67.combine_file__rs1zuu4a|4/4|1.99|1.97|
|16|pandas-dev__pandas.95280573.combine_module__ijpr6qe5|4/4|10.48|6.24|
|17|agronholm__exceptiongroup.0b4f4937.combine_module__mkpnzqei|124/124|1.94|601.71|
|18|pandas-dev__pandas.95280573.func_pm_class_rm_funcs__wp46asyn|4/4|19.55|5.99|
|19|pandas-dev__pandas.95280573.combine_module__y7p75fvm|3/3|10.07|15.00|
|20|pandas-dev__pandas.95280573.func_pm_ctrl_invert_if__aw7aw3d2|4/4|9.91|5.91|
|21|pandas-dev__pandas.95280573.func_pm_ctrl_shuffle__73h0zw5z|4/4|18.92|6.05|
|22|pandas-dev__pandas.95280573.func_pm_remove_cond__b95e7d7b|4/4|15.75|5.76|
|23|pandas-dev__pandas.95280573.func_pm_class_rm_funcs__9k8jta6g|4/4|9.52|5.94|
|24|mahmoud__boltons.3bfcfdd0.func_pm_remove_cond__xcq6er4x|4/4|1.94|1.94|
|25|andialbrecht__sqlparse.e57923b3.func_basic__w7bwenem|4/4|2.17|1.90|
|26|pandas-dev__pandas.95280573.combine_file__m3gu6j4n|4/4|9.64|5.75|
|27|pandas-dev__pandas.95280573.func_pm_remove_cond__f4ref4bb|4/4|15.53|5.92|

## 源码与执行边界

只新增`research/regrade_v2_exceptions.py`与`research/run_regrade_v2_exceptions.sh`及本文档，未改生产源码、未提交。通过WSL rsync先dry-run后部署两个脚本，远端`py_compile`与`bash -n`通过后用唯一screen启动。使用已保存补丁，无采样调用，无旧目录覆盖，无训练进程干预。
