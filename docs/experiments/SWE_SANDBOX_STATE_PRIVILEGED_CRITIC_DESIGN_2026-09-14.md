# SWE Sandbox-State Privileged Critic 设计方案

> 日期：2026-09-14
> 目标框架：Agent Lightning + 当前 SWE multi-turn PPO / CAPO backend
> 目标任务：SWE-smith 训练，后续 SWE-bench Actor-only 评测
> 研究主题：将 ALFWorld 中的 Asymmetric Actor-Critic / Privileged Critic 迁移到软件工程 Agent
> 本文用途：交接给后续实施 Agent，作为第一版实现的技术设计与边界约束
> 状态：**设计稿，尚未表示本方案已在当前 SWE 训练代码中实现**

---

## 0. 一句话结论

本方案只有一条主线：

\[
\boxed{
\text{Actor}: \pi_\theta(a_t\mid H_t),\qquad
\text{Critic}: V_\phi(H_t,Z_t)
}
\]

其中 Actor 继续只使用正常 SWE terminal/tool interaction history \(H_t\)，而 Critic 在训练阶段额外读取当前 action / model call **之前**的真实 sandbox state：

\[
\boxed{
Z_t =
\operatorname{BudgetSerialize}
\left(
D(
\operatorname{Snapshot}(S_0),
\operatorname{Snapshot}(S_t)
)
\right)
}
\]

第一版的 `Snapshot()` 只包含两大类：

1. `/testbed` 工作区的当前文件系统状态；
2. 当前 rollout sandbox 内的 process / socket runtime state。

这里的 cumulative delta、4K token budget、event transport、Critic-only input 都属于同一套方案的不同组成部分，不是多个互斥方案。

后续可以研究 learned state encoder、full tool output、full current state 等增强，但**第一版一律不做**。

---

# 1. 研究问题

当前 SWE PPO 中，Actor 和普通 Critic 都主要依赖可见交互历史：

\[
H_t =
\{
\text{issue / task},
\text{previous model outputs},
\text{executed shell commands},
\text{visible tool observations}
\}
\]

普通 Critic：

\[
V_\phi(H_t)
\]

必须同时完成两件事：

1. 从长交互历史中推断当前真实代码 / runtime 状态；
2. 在推断出的状态上预测最终任务成功概率 / return。

本项目希望将机器人强化学习中的 Asymmetric Actor-Critic 思路迁移到 SWE Agent：

\[
\text{Actor}: \pi_\theta(a_t\mid H_t)
\]

\[
\text{Critic}: V_\phi(H_t,Z_t)
\]

其中 \(Z_t\) 是训练阶段通过 sandbox control plane 获得的当前真实环境状态。

研究问题不是“把答案告诉 Critic 会不会更强”，而是：

> 当 action history 不能完全确定软件环境当前真实状态时，训练期额外提供当前 sandbox state，是否能够降低 Critic 的状态重建负担，提高 value estimation、credit assignment 和 PPO sample efficiency？

---

# 2. 与 ALFWorld 的对应关系

ALFWorld 中：

```text
Actor:
    textual observation / interaction history

Critic:
    textual history
    + simulator current symbolic facts
```

SWE 中对应为：

```text
Actor:
    issue + shell/tool interaction history

Critic:
    same Actor history
    + current sandbox state
```

对应关系：

```text
ALFWorld simulator symbolic state
            ↓
SWE rollout sandbox state
```

核心叙事保持一致：

> The actor knows what it attempted; the privileged critic observes what the environment actually became.

在 SWE 中，Actor 可能知道自己执行了：

```bash
python setup.py
pytest ...
python server.py &
make ...
```

但并不必然知道：

- 哪些文件被工具间接创建或修改；
- 新建文件是否真正写入成功；
- formatter / generator 实际改了哪些文件；
- background process 是否仍然存活；
- server 是否真正监听某个端口；
- 某个 command 被截断后完整产生了哪些环境副作用。

因此 action history 并不唯一确定当前真实 sandbox state。

---

# 3. 第一版方法的严格定义

## 3.1 Actor 信息

Actor 保持当前正式 baseline 的输入和 rollout 行为：

\[
\pi_\theta(a_t\mid H_t)
\]

不得因为加入 Privileged Critic 而修改：

- task / issue prompt；
- system prompt；
- tool protocol；
- Actor history；
- rollout generation；
- reward；
- old log-prob；
- reference log-prob；
- Actor PPO update；
- validation / deployment inference。

## 3.2 Critic 信息

Critic：

\[
V_\phi(H_t,Z_t)
\]

其中：

\[
Z_t = \text{pre-model-call sandbox state}
\]

必须满足：

### Temporal admissibility

\[
Z_t
\]

必须在当前 model call / action \(a_t\) 生成之前已经存在。

禁止把：

\[
S_{t+1}
\]

错配给当前 action。

## 3.3 Privilege 来源

Privilege 必须来自 Actor 正常 observation channel 之外的 control-plane state。

第一版定义：

\[
\boxed{
S_t =
\text{mutable state inside this rollout's sandbox boundary}
}
\]

而不是：

```text
我们认为对任务有用的几个 feature
```

因此方法目标是：

> zero semantic feature engineering, not zero environment integration.

不同 repo / task 不写不同规则。

---

# 4. 第一版 Sandbox State 内容

第一版只实现两个 namespace。

## 4.1 Filesystem state

作用范围：

```text
/testbed
```

或者当前自定义 SWE agent 中对应的 task working tree。

每个文件的 canonical metadata：

```text
relative_path
type
mode
size
mtime / stat metadata
content hash
```

对于文本文件，在文件发生变化时额外读取：

```text
current textual content
```

对于 binary / 非文本文件，只保留：

```text
path
size
hash
type
```

禁止扫描任务容器以外的共享服务器目录。

## 4.2 Runtime state

只读取当前 rollout sandbox / container / cgroup / namespace 内部的运行状态。

包括：

```text
process:
    pid / namespace-local id
    ppid
    command
    state
    cwd

socket:
    protocol
    local address
    listening state
    owner process if resolvable
```

第一版不追求复杂 service semantics。

例如不要写：

```python
if nginx:
    ...
if postgres:
    ...
```

只做 generic process / socket enumeration。

---

# 5. 明确禁止进入 PI 的内容

以下内容即使 control plane 能访问，也禁止作为第一版 Privileged Critic 输入：

```text
hidden .git history
pre-bug / fixed commit
gold patch
reference solution
FAIL_TO_PASS / PASS_TO_PASS 答案
hidden tests 内容
grader / evaluator internal state
最终 resolved
最终 reward
future trajectory
future tool observation
当前 action 执行后的 state
```

当前公开 SWE-smith agent 会把 `.git` 从 `/testbed` 搬到隐藏路径，因为其中包含 injected bug 修复和删除的测试信息。

因此：

```text
Actor-owned sandbox state
        !=
Evaluator / harness secret state
```

本方案只允许前者。

---

# 6. 为什么第一版使用 Cumulative Sandbox Delta

完整 repo 往往非常大，而且 episode 内绝大多数文件保持不变。

因此不直接把：

\[
S_t
\]

全文放入 Critic，而是计算：

\[
\boxed{
\Delta S_t=D(S_0,S_t)
}
\]

其中：

- \(S_0\)：episode / rollout 初始化后的 sandbox snapshot；
- \(S_t\)：当前 model call 前 snapshot；
- \(D\)：generic structural diff。

注意：

\[
D(S_0,S_t)
\]

不是：

\[
D(S_{t-1},S_t)
\]

也不是 event log。

如果一个文件改了以后又恢复到初始状态，则它应从 cumulative delta 中消失。

---

# 7. Delta 示例

```text
[SANDBOX STATE DELTA]

filesystem:
  modified:
    src/parser.py
      size: 4812 -> 5073

  added:
    repro.py
      size: 824

  removed:
    tmp/old.cache

process:
  added:
    pid: 182
    command: python server.py
    state: sleeping
    cwd: /testbed

socket:
  added:
    tcp 127.0.0.1:8000 LISTEN
    owner_pid: 182
```

如果 token budget 允许，再展开文本文件的实际内容差异：

```text
[FILE DELTA: src/parser.py]

@@ ...
-old code
+new code
```

---

# 8. 不建议直接使用 `git diff`

`git diff` 可以作为 debug / sanity check，但不建议作为方法定义。

原因：

1. 方法会显得 SWE-specific；
2. untracked 新文件容易漏掉；
3. tool 产生的其他 writable state 不一定由 git 表示；
4. Runtime state 完全不在 git 中；
5. 当前 SWE agent 的 `.git` 还属于隐藏 harness 区域，不应依赖它作为 Actor-side environment API。

因此第一版应实现 generic filesystem snapshotter。

---

# 9. Filesystem Snapshot 的工程实现建议

不应在每个 turn 对整个 repo 的每个文件重新做完整 SHA 和全文读取。

建议流程：

```text
initial snapshot
    ↓
stat scan
    ↓
size / mtime / inode 无变化
    → 复用旧 hash

可能变化
    → 重新 hash

hash 变化
    → 标记 dirty
    → 文本文件按需读取内容
```

可选优化：

```text
inotify / watchdog
```

维护 dirty path set。

但首版如果 repo 规模允许，可以先用简单、确定性的 stat + hash 方案保证正确性，再优化速度。

---

# 10. Pre-action 时间对齐

这是实现中最重要的约束。

当前逻辑应变成：

```text
messages = H_t

capture sandbox state S_t
        ↓
emit privileged_state event
        ↓
query Actor
        ↓
generate action a_t
        ↓
execute action
        ↓
receive observation
        ↓
append observation
        ↓
next turn
```

伪代码：

```python
for turn_index in range(max_turns):

    snapshot_t = snapshotter.snapshot()

    logical_call_id = new_id()

    emit_privileged_state(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        turn_index=turn_index,
        logical_call_id=logical_call_id,
        snapshot=snapshot_t,
    )

    response = query_actor(
        messages,
        logical_call_id=logical_call_id,
    )

    action = parse(response)

    output = run(action)

    messages.append(render_observation(output))
```

禁止：

```python
response = query_actor(...)
run(action)
snapshot = snapshotter.snapshot()
# 然后把这个 snapshot 配给当前 response
```

这是未来信息泄漏。

---

# 11. `logical_call_id` 是必须新增的字段

不能只依赖：

```text
prompt hash
prompt_token_ids
turn_index
```

因为：

- request 可能重试；
- Gateway 可能发生 pause / weight sync；
- 当前 Agent Lightning Gateway 会对重复 prompt request 做去重；
- 同样 prompt 不一定表示同一逻辑调用。

因此每次逻辑 Actor call 必须生成：

```text
logical_call_id
```

绑定：

```text
rollout_id
attempt_id
turn_index
logical_call_id
snapshot_id
model_request
response
```

缺失唯一配对关系时：

```text
drop / reject training row
```

不能猜测匹配。

---

# 12. Lightning Event 数据通路

Agent Lightning event API 支持任意：

```text
event_type
data
```

因此第一版建议新增：

```text
event_type = privileged_state
```

数据结构建议：

```json
{
  "logical_call_id": "...",
  "turn_index": 7,
  "snapshot_sequence": 7,
  "captured_at": 123456.7,
  "state_schema_version": 1,
  "state_hash": "...",
  "state_ref": "...",
  "summary": {
    "changed_files": 14,
    "new_processes": 1,
    "listening_sockets": 1
  }
}
```

如果 raw snapshot 较大：

不要把全文长期放 Gateway memory store。

优先：

```text
event:
    state_id
    hash
    metadata
    raw state file reference
```

raw snapshot 压缩后单独写入 D1：

```text
runtime/<run>/privileged_state/<rollout>/<call>.json.zst
```

---

# 13. RolloutManager 修改

当前 `Triplet` 已有：

```python
metadata: dict[str, Any]
```

后续应把配对后的 PI metadata 写入每个 Triplet：

```python
Triplet(
    prompt=...,
    response=...,
    reward=...,
    metadata={
        "server": ...,
        "logical_call_id": logical_call_id,
        "privileged_state_id": state_id,
        "privileged_state_hash": state_hash,
        "snapshot_sequence": snapshot_sequence,
        "privileged_state_ref": state_ref,
    },
)
```

要求：

- 一个有效 Triplet 对应唯一 pre-action state；
- 缺失 / 多义 / 时序错误直接拒绝；
- 不把 PI 拼进 Actor model request messages。

---

# 14. RolloutAdapter 修改

当前 transition 模式已经天然按：

```text
rollout_id
turn_index
```

拆成每次模型调用一行。

这正是第一版 PI 最适合的数据表示。

应增加：

```text
logical_call_id_list
privileged_state_id_list
privileged_state_ref_list
privileged_state_hash_list
```

随后为每行分别构建：

```text
actor view
critic view
```

Actor view：

```text
H_t
+
same response IDs
```

Critic view：

```text
H_t
+
budgeted Z_t
+
same response IDs
```

必须复用：

```text
same response_ids
same response_mask
```

---

# 15. Critic-only Tensor

参考已经在 ALFWorld / CAPO 中验证过的做法，构造独立：

```text
critic_input_ids
critic_attention_mask
critic_position_ids
```

Actor 原字段继续是：

```text
input_ids
attention_mask
position_ids
```

不得原地覆盖公共 batch。

---

# 16. Trainer 路由

只有：

```text
compute_values
update_critic
```

使用 Critic view。

以下全部保持 Actor view：

```text
rollout
old log-prob
reference log-prob
reward
Actor PPO update
validation
```

数据流：

```text
actor_input
    ├── rollout
    ├── old logprob
    ├── ref logprob
    └── actor update

critic_input
    ├── compute_values
    └── update_critic
```

---

# 17. Token GAE 保持不变

当前正式 SWE PPO baseline 已使用：

```text
token GAE
gamma = 1
lambda = 1
```

第一版 PI 实验必须保持这些设置不变。

对于一个 model call：

```text
response token 1
response token 2
...
response token K
```

全部使用同一份：

\[
Z_t
\]

即：

\[
V(H_t,Z_t,a_{t,<k})
\]

下一次 model call 才切换：

\[
Z_{t+1}
\]

PI prompt tokens 本身不属于 RL 时间步，也不进入 GAE 递推。

---

# 18. Context 是第一版必须处理的问题

当前正式 baseline 的关键配置为：

```text
Qwen3-4B-Instruct-2507
context = 65536
response max = 4096
turns = 32
observation char cap = 32000
```

因此不能简单：

```text
Actor prompt
+
完整 PI
```

否则可能挤掉 Actor history。

这会把实验变成：

\[
V(\text{truncated }H_t,Z_t)
\]

对比：

\[
V(H_t)
\]

失去单变量意义。

---

# 19. Context 核心原则

\[
\boxed{
\textbf{永远优先保留完整 Actor }H_t
}
\]

如果空间不足：

```text
截 PI
```

而不是：

```text
截 Actor history
```

---

# 20. 动态 PI Budget

定义：

\[
P_t = |\text{Actor prompt tokens}|
\]

\[
R_t = |\text{actual response tokens}|
\]

Critic sequence 上限：

\[
L_C
\]

则当前样本可用 PI budget：

\[
B_t
=
L_C-P_t-R_t-M
\]

其中 \(M\) 是 safety margin。

实际：

\[
B_t^{PI}
=
\min(
B_{\max},
\max(0,B_t)
)
\]

第一版建议：

```text
B_max = 4096 tokens
```

但这个值必须通过 PI-size audit 验证。

---

# 21. Raw Snapshot 先保存，序列化可以后做

时间正确性要求的是：

```text
raw state S_t
```

必须在 action 前采集。

但是：

```text
S_t -> textual PI
```

不要求必须在 action 前立刻完成。

因此推荐：

```text
pre-action:
    capture raw structured snapshot

rollout completed:
    已知 actual actor prompt tokens
    已知 actual response tokens
        ↓
    calculate exact PI budget
        ↓
    budgeted serialization
```

这样不会使用 future state，只是晚一些决定如何编码已经保存的 pre-action state。

---

# 22. PI Serializer 的固定优先级

不要做 task-specific ranking。

采用 generic hierarchy：

```text
Level 1: snapshot / delta summary
Level 2: complete changed-path manifest
Level 3: process + socket state
Level 4: text file content / unified diff
Level 5: omitted-state metadata
```

例如：

```text
[STATE SUMMARY]
changed_files: 14
added: 2
removed: 0
modified: 12
new_processes: 1
listening_sockets: 1

[FILE MANIFEST]
M src/parser.py
M src/utils.py
A repro.py

[RUNTIME]
+ python server.py pid=182
+ tcp 127.0.0.1:8000 LISTEN

[FILE DELTAS]
...
```

如果超预算：

```text
[OMITTED]
7 file deltas
6341 estimated tokens omitted
```

这样 Critic 至少知道还有未展开的状态变化。

---

# 23. 4K Budget 不是最终结论

第一阶段实现 snapshotter 后，应先做只读 PI-size audit。

建议收集至少数百到数千个真实 model calls：

```text
actor_prompt_tokens
response_tokens
context_slack

changed_file_count
filesystem_delta_bytes
filesystem_delta_tokens

process_delta_tokens
socket_delta_tokens

PI 1K coverage
PI 2K coverage
PI 4K coverage
PI 8K coverage
```

目标是回答：

```text
1K 可以完整表示多少样本？
2K 呢？
4K 呢？
8K 呢？
```

根据真实分布决定正式 \(B_{\max}\)。

---

# 24. PI Audit 建议指标

每个 call 记录：

```text
privilege/raw_record_count
privilege/raw_bytes
privilege/raw_tokens_estimate

privilege/budget_tokens
privilege/serialized_tokens
privilege/truncated
privilege/omitted_record_count

privilege/files_changed
privilege/files_added
privilege/files_removed
privilege/processes_changed
privilege/sockets_changed

privilege/serialization_time
privilege/snapshot_time
```

建议增加：

\[
\text{state coverage}
=
\frac{
\text{records preserved}
}{
\text{raw changed records}
}
\]

以及：

\[
\text{information density}
=
\frac{
\text{records preserved}
}{
\text{PI tokens}
}
\]

---

# 25. 第一版不包含 Full Tool Output

当前 SWE agent 对超长 command output 会进行 observation truncation。

这意味着：

```text
Actor:
    only sees truncated output

control plane:
    may possess full stdout/stderr
```

这确实是一种非常强的 privilege。

但是第一版不要把它与 sandbox state 混在一起。

原因：如果效果提升，无法判断来自：

```text
true environment state
```

还是：

```text
只是补全了被截断的 tool output
```

所以：

```text
Full Tool Feedback PI
```

只作为后续独立 ablation。

---

# 26. 第一版也不包含 Learned State Encoder

如果未来发现 raw sandbox delta 经常需要：

```text
20K
50K
100K tokens
```

那么文本序列化会成为瓶颈。

后续可以研究：

\[
e_i = E(x_i)
\]

将每个 state record 编码后，再用 fixed latent queries：

\[
z_{1:K}
=
\operatorname{Resampler}(\{e_i\})
\]

最终：

\[
V(H_t,z_{1:K},a_{t,<k})
\]

例如把任意 sandbox state 压成固定 32 / 64 / 128 个 privileged latent tokens。

但是这会同时改变：

```text
Critic architecture
representation
PI
```

第一版禁止引入，避免混淆主变量。

---

# 27. 第一版正式方法名称建议

暂定：

## Sandbox-State Privileged Critic

或：

## Asymmetric Actor-Critic with Sandbox State for Software Engineering Agents

一句话方法描述：

> We treat the execution sandbox as the simulator of a software-engineering agent. The deployable policy observes only normal tool interactions, while the training-time critic additionally observes the pre-action state of the sandbox through an out-of-band control interface.

---

# 28. 代码改动层级

实际文件名需以当前自定义 SWE PPO 工作树为准。

预计修改层：

```text
SWE agent / full_python_agent.py
        ↓
SandboxSnapshotter
        ↓
privileged_state Event
        ↓
Gateway
        ↓
AglRolloutManager
        ↓
Triplet.metadata
        ↓
RolloutAdapter
        ↓
critic-only input builder
        ↓
CAPO / PPO trainer
        ↓
compute_values + update_critic
```

---

# 29. 推荐新增模块

建议新增类似：

```text
agentlightning/privileged_state/
    __init__.py
    snapshot.py
    filesystem.py
    runtime.py
    canonicalize.py
    diff.py
    serialize.py
```

以及：

```text
agentlightning/verl/privileged_critic.py
```

具体文件位置可根据当前分支结构调整。

环境相关 snapshot adapter 可放：

```text
examples/multiturn_ppo/
```

或者当前 SWE agent 所在目录。

核心要求是：

```text
generic state engine
```

不要把 repo/task semantic rule 写进 trainer。

---

# 30. 推荐的数据字段

每次 model call 至少保存：

```text
rollout_id
attempt_id
turn_index
logical_call_id

snapshot_sequence
snapshot_timestamp
state_schema_version
state_hash
privileged_state_ref

actor_prompt_hash
actor_prompt_tokens
response_tokens

pi_budget_tokens
pi_serialized_tokens
pi_truncated

terminated
truncated
termination_reason
```

---

# 31. Actor Leakage 自动断言

正式训练前必须有 hard assertion：

```text
Actor prompt 不含 PI marker
Actor input_ids 与 baseline builder 一致
Actor old/ref logprob 使用 Actor inputs
Actor PPO update 使用 Actor inputs

Critic prompt 包含正确 PI
compute_values 使用 Critic view
update_critic 使用同一 Critic view

Critic response_ids == Actor response_ids
Critic response_mask == Actor response_mask

state_timestamp < model_request timestamp
state turn / call id 唯一匹配
```

失败应：

```text
raise / reject sample
```

不要仅 warning。

---

# 32. 单元测试

至少覆盖以下测试。

## 32.1 Snapshot

```text
文件未变化 -> diff 中不存在
文件修改 -> changed
文件新增 -> added
文件删除 -> removed
修改后恢复初始 -> cumulative diff 消失
binary -> 不读全文，只 metadata/hash
```

## 32.2 Runtime

```text
后台进程启动 -> added
后台进程退出 -> 与初始一致时消失
监听端口出现 -> added
rollout 外进程 -> 不可见
```

## 32.3 时间对齐

构造两步轨迹：

```text
S0
 -> call 0 uses Z0
 -> execute a0
 -> S1
 -> call 1 uses Z1
```

断言：

```text
call0 不能看到 S1
```

## 32.4 Request Retry

相同 prompt 重试：

```text
logical_call_id A
logical_call_id B
```

必须能够唯一绑定 state 和 response。

## 32.5 Context

```text
Actor prompt 很长
PI 很长
```

断言：

```text
Actor H_t 不被 PI 截断
PI 被 budget 截断
response 不错位
```

## 32.6 Actor / Critic View

断言：

```text
actor tensor unchanged
critic tensor different
same response IDs
same response mask
batch 原对象未被原地修改
```

---

# 33. Smoke Test

正式训练前做小型真实环境 smoke。

必须验证：

```text
1. raw snapshot 能生成
2. privileged_state event 能到 Gateway
3. RolloutManager 能唯一配对
4. RolloutAdapter 能生成 Critic view
5. compute_values 正常
6. update_critic 正常
7. Actor logprob 不变
8. token GAE 正常
9. Actor/critic 两组件可保存
10. 无明显 context overflow / OOM
```

零成功 smoke 只能验证通路，不能验证方法效果。

---

# 34. PI-size Audit 阶段

建议先不改训练：

```text
privileged_state_enabled = capture_only
```

只采集：

```text
snapshot
delta
serialization stats
context headroom
```

不送 Critic。

目标：

- 确定 4K 是否足够；
- 确定 snapshot 开销；
- 确定文件 diff 的长尾；
- 确定 process/socket 是否经常非空；
- 检查是否出现 context 无余量的 turn。

---

# 35. 正式实验的最小对照

第一轮主实验只做：

## Baseline

\[
V(H_t)
\]

## Sandbox-State Privileged Critic

\[
V(H_t,Z_t)
\]

两组保持：

```text
Actor model
Critic model
reward
PPO
token GAE
gamma / lambda
LR
batch
sampling
prompt
context policy
validation
training data
```

全部一致。

研究主变量只有：

```text
Critic 是否收到 sandbox PI
```

---

# 36. 后续机制对照

主实验稳定后，再考虑：

## History-derived state summary

\[
V(H_t,\hat Z(H_t))
\]

仅从 Actor 已见 history 构造 state summary。

用于区分：

```text
真实信息增量
vs
只是更方便的 state representation
```

## Shuffled State

将其他 rollout 的 \(Z\) 给当前 Critic。

验证收益是否来自正确 state，而不是额外文本长度。

## Full Tool Feedback

加入 Actor 被截断的 stdout/stderr。

单独实验，不与主方法混合。

## Full Current State

与 cumulative delta 对比，分析 compression 是否损失信息。

## Learned State Encoder

只有 text PI context 成为明显瓶颈时再做。

---

# 37. 第一版明确不做的事情

实施 Agent 不要顺手加入：

```text
gold patch
hidden tests
goal progress
grader progress
test pass ratio
final outcome
future state

Actor-visible PI
reward shaping
新的 process reward

critic warmup
新的 LR
新的 PPO clip
新的 gamma/lambda
新的 Actor prompt
新的 reward
新的 architecture
learned state encoder
```

如果发现当前 baseline bug，则应先独立说明并修复 baseline，而不是把修复混进 PI 实验。

---

# 38. 当前 Baseline 需要保持的关键设置

根据 2026-09-14 的正式 SWE PPO handoff，当前主 baseline 为：

```text
Model:
    Qwen3-4B-Instruct-2507

PPO backend:
    CAPO

Data:
    6248 Python train
    470 Python val

Train:
    4 epochs
    784 PPO steps

Task batch:
    32

PPO minibatch:
    32

Actor / critic LR:
    1e-6 / 1e-5

gamma:
    1

lambda:
    1

PPO clip:
    0.2

KL:
    0.001

GAE:
    token GAE

Turns:
    32

Response max:
    4096

Context:
    65536

Observation char cap:
    32000

Validation:
    initial
    every 20 steps
    final
```

PI 实验原则上应继承该 baseline，而不是重新设计训练 recipe。

实际实现时必须再次检查当前 run 的 `resolved-config.json`，不能只依赖本文。

---

# 39. 对 Context 长度的正式决策流程

实施 Agent 不要直接把 Critic context 改成更大。

先按以下顺序：

### Step A

保持：

```text
critic context = current baseline context
```

PI dynamic budget 先设：

```text
max 4096
```

Actor H 不截。

### Step B

通过 capture-only audit 统计：

```text
4K state coverage
context headroom
PI truncation rate
```

### Step C

如果大量样本 PI 因为 context headroom 不足被截：

再单独测试：

```text
critic context 72K / 80K
```

但这属于系统容量实验。

必须测：

```text
critic forward memory
critic update memory
throughput
OOM
```

不能直接用于主实验而不说明额外成本。

---

# 40. 成功验收标准

第一版实现完成必须满足：

```text
[ ] Actor 输入 byte/token 级不含 PI
[ ] reward 不变
[ ] PPO objective 不变
[ ] GAE 不变
[ ] PI 为 action 前 state
[ ] hidden evaluator info 不进入 PI
[ ] 每个 Triplet 唯一匹配 state
[ ] Actor H 不因 PI 被截
[ ] Critic-only tensor 独立
[ ] compute_values / update_critic 都使用 PI view
[ ] filesystem cumulative delta 正确
[ ] runtime state 局限在 rollout sandbox
[ ] 4K budget 的截断行为可审计
[ ] snapshot / serialize 性能有日志
[ ] 小规模真实训练可以完成 Actor/Critic update
[ ] checkpoint actor + critic 完整
```

---

# 41. 论文主叙事

不要写成：

> We manually design privileged features for SWE tasks.

建议写成：

> We treat the software execution sandbox as the simulator state of a software-engineering agent.

Actor：

\[
\pi_\theta(a_t\mid H_t)
\]

Critic：

\[
V_\phi(H_t,Z_t)
\]

其中：

\[
Z_t
=
\text{current out-of-band sandbox state}
\]

为了控制 context：

\[
\tilde Z_t
=
D(Z_0,Z_t)
\]

并做 budgeted serialization。

更完整的一句话：

> Long-horizon software agents interact with a persistent execution environment whose realized state is only partially reflected in terminal transcripts. We therefore train an asymmetric value function that additionally observes the pre-action state of the task sandbox, while keeping the policy, reward, and deployment interface unchanged.

---

# 42. 与相关工作的边界

当前 idea 不应 claim：

```text
首次提出 privileged value function
```

更合理的定位是：

```text
Privileged Value Function
+
real long-horizon SWE agent
+
environment-side sandbox state
+
strict PPO value baseline
```

与 hindsight / retrospective 方法不同：

```text
future trajectory
gold answer
final verifier
final reward
```

不进入 Critic PI。

方法的特色是：

```text
current pre-action environment state
```

---

# 43. 实施 Agent 推荐执行顺序

## Phase 0：只读核对当前源码

确认当前自定义工作树：

```text
full_python_agent.py
Gateway event path
AglRolloutManager
RolloutAdapter
CAPO trainer adapter
current resolved config
```

不要直接根据公开 main 的文件行号修改。

## Phase 1：实现 Snapshotter

仅：

```text
filesystem
process
socket
```

并有 deterministic schema / hash。

## Phase 2：Capture-only

在真实 rollout 中 action 前采 snapshot。

落盘。

不送 Critic。

跑 PI-size audit。

## Phase 3：Event + Alignment

实现：

```text
logical_call_id
privileged_state event
unique Triplet matching
```

## Phase 4：Critic View

构造：

```text
critic_input_ids
critic_attention_mask
critic_position_ids
```

Actor view 保持不变。

## Phase 5：Trainer 接入

只改：

```text
compute_values
update_critic
```

## Phase 6：Smoke

验证：

```text
time alignment
actor leakage
response identity
token GAE
memory
checkpoint
```

## Phase 7：正式 Baseline vs PI

同配置。

只改变：

```text
privileged_state.enabled
```

---

# 44. 参考项目文档

实施前建议阅读以下已有交接：

```text
PRIVILEGED_CRITIC_PROJECT_HANDOFF_2026-09-02.md
TOKEN_GAE_PRIVILEGED_CRITIC_GUIDE.md
CAPO_PRIVILEGED_CRITIC_RESEARCH_2026-09-06.md
CAPO_P1_HANDOFF_2026-09-09.md
AGENT_LIGHTNING_PRIVILEGED_CRITIC_RESEARCH_2026-09-09.md
PPO_BASELINE_READINESS_2026-09-10.md
SWE_PPO_BASELINE_HANDOFF_2026-09-14.md
```

其中旧运行状态只作为历史记录，实施时以当前源码和当前 resolved config 为准。

---

# 45. 最终交接摘要

第一版不要复杂化。

主方法只有：

\[
\boxed{
\text{pre-action sandbox snapshot}
\rightarrow
\text{cumulative sandbox delta}
\rightarrow
\text{budgeted textual PI}
\rightarrow
\text{Critic only}
}
\]

Sandbox state 第一版只有：

```text
/testbed filesystem state
+
rollout-local process/socket state
```

Critic context 第一版采用：

```text
动态 budget
最大先按 4096 tokens
优先保留 Actor H_t
PI 不够空间就截 PI
```

Transport 使用：

```text
privileged_state event
+
logical_call_id
```

训练侧使用：

```text
same response IDs
same response mask
separate critic input
compute_values + update_critic only
```

其余：

```text
reward
Actor
old/ref logprob
token GAE
PPO objective
validation
```

全部保持当前 baseline 不变。

如果第一版有效，再做：

```text
history-derived summary
shuffled state
full tool feedback
full current state
learned state encoder
```

这些都不属于第一版实施范围。
