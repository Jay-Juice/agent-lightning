# Patsy 数据任务的测试修复

2026-09-13。正式全量训练尚未启动。本次只运行 CPU/Docker 诊断，没有启动模型或占用 GPU。

## 问题归属

15 条任务的测试改动已存在于发布的 SWE-smith 镜像任务分支 `Bug Patch` 提交中，先于本地 PPO 修改。这是数据任务的缺陷，不能据此称为 agent-lightning 的 PPO 实现错误，也不能断言是哪位作者操作造成。我们的全量 SWE 适配此前从 Bug Patch 恢复源码混合文件时，连带恢复了被改坏的测试；这部分适配现在修正。

固定镜像：`jyangballin/swesmith.x86_64.pydata_1776_patsy.a5d16484@sha256:60bbd514959a26d2782fb7ae92a7c5067c903282c13882cb0e5ec755ab382fbe`。

例如 `combine_file__4zw01ak1` 的 `cc5425f9f2e568fe201e47a458271c95248e3183` 改写 `test_CategoricalSniffer` 等测试；`func_basic__c7sryiet` 的 `b88dff1c1a4d522ed4aa8c86700fb439e5b4d470` 把 `ValueError` 预期改成 `TypeError`。

## 实际修改

- 从 Bug Patch 恢复带缺陷源码，仅把发生变异的顶层测试函数/测试类替换为注入缺陷之前的版本。逐段替换保留业务源码；独立 AST 检查确保业务代码没有被参考实现覆盖。
- 测试清单变化、重复定义、无法处理的嵌套测试或 doctest 变异会拒绝处理。模型导出的测试内容仍必须与准备后的基线一致，未放宽测试修改权限。
- 参考对照针对准备后的基线生成补丁，并限制在原始 Bug Patch 修改的路径内，避免直接套用旧补丁产生上下文冲突。
- 全量启动预检区分可恢复的混合任务与只改坏测试的任务。后者仍阻止正式训练启动，不会因为测试已恢复而通过启动验收。
- 不改 PPO loss、GAE、学习率、模型配置、奖励判定或测试节点集合。属于自定义数据/评分适配，结果应注明此规范化处理，不能宣称逐字沿用原始 SWE-smith 任务。

## 真实验证

证据根目录（A800）：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/swe-embedded-test-repairs-20260913-02`。
包含源码/数据 SHA-256、分支审计、每题测试日志、参考导出补丁和评分结果。共 27 次真实评分：6 条各运行空补丁、参考修复、导出补丁重放；9 条运行规范化后的空补丁。

下表 ID 的共同前缀为 `pydata__patsy.a5d16484.`。

| 混合任务 | 空补丁 F2P | 修复后 F2P | 修复后 P2P | 空/参考/导出重放奖励 |
| --- | --- | --- | --- | --- |
| combine_file__4zw01ak1 | 1/3 | 3/3 | 18/18 | 0 / 1 / 1 |
| combine_file__742yas93 | 1/14 | 14/14 | 23/23 | 0 / 1 / 1 |
| combine_file__9t660qje | 1/33 | 33/33 | 21/21 | 0 / 1 / 1 |
| combine_file__hn93r21p | 2/5 | 5/5 | 8/8 | 0 / 1 / 1 |
| combine_file__i1wgvq3x | 0/32 | 32/32 | 40/40 | 0 / 1 / 1 |
| combine_file__s0u0xoks | 0/29 | 29/29 | 29/29 | 0 / 1 / 1 |

**6/6 可保留的混合任务全部通过三组验证。** 部分原标记 F2P 在恢复测试后已经通过；测试节点集合完整保留，空补丁仍无法满足全部 F2P。

另外 9 条去掉测试 AST 后没有业务代码变动，恢复测试后全部在空补丁下得到 1。因此它们不适合作为当前“修复业务源码、禁止改测试”的训练任务；不能通过添加新缺陷伪装成原任务修复。

```text
func_basic__3ygynmye
func_basic__autc5whk
func_basic__c7sryiet
func_basic__di24tglu
func_pm_remove_assign__wejihg7c
lm_rewrite__35zbdp36
lm_rewrite__8zfv7wmq
lm_rewrite__pmhlov0r
lm_rewrite__sxqo1wp0
```

这 9 条尚未从数据中剔除。当前 `python-full-v5` 仍为 6260 train / 470 val，仅应用此前用户批准的 55 条联网依赖剔除。建议单独记录并剔除上述 9 条，则为 6251 train / 470 val，保留 6 条已修复的混合任务；四个完整 epoch 仍为 784 个 PPO step。不得将本次 6 条参考补丁成功率当作模型成功率。

34 项评分/导出/编辑回归测试通过，Ruff 通过。第一轮诊断目录 `swe-embedded-test-repairs-20260913-01` 因新增诊断脚本把导出的路径列表误读为拒绝列表而中断混合任务的最后重放，记录原样保留；第二轮已修正诊断脚本并完整通过，第一轮失败不作为训练框架失败统计。

全部 **124 个镜像、6730 条当前训练/验证任务**也完成了分支复查：分支结构失败 0、无法支持的内嵌测试冲突 0、受保护路径冲突 0；发现且恢复的测试变异仍为这 15 条，其中 9 条没有业务缺陷。证据为 `logs/swe-normalized-branch-audit-20260913-01`。这是静态分支/导出兼容性检查，不能等同于逐题测试执行或模型评测。

## 复现

在已激活的 A800 环境、远端仓库目录中执行（输出目录必须不存在）：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
python examples/multiturn_ppo/audit_embedded_test_repairs.py \
  --data /media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/python-full-v5 \
  --output /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/<new-audit-directory>
```

Windows 为源码权威，四个 Python 文件通过 WSL `rsync -anvzR` 检查后用 `rsync -avzR` 部署，无 `--delete`。服务器只产生诊断日志和临时容器，不直接修改远端源文件。完整 124 镜像正反评分验收仍需在最终数据及源码固定后重新生成，不能沿用旧 123/124 结果的签名。
