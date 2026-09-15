# SWE PI-v2 implementation and launch

Date: 2026-09-15 (Asia/Shanghai)

PI-v2 was implemented in the isolated branch `experiment/swe-sandbox-pi-v2-20260915`, preserving the P1-v1 and baseline worktrees. The implementation changes only the privileged-state encoding, no-PI context fallback, and read-only value/coverage diagnostics. PPO loss, GAE, reward, actor inputs, model, sampling, learning rates, batch sizes, context and response budgets remain unchanged.

Commits:

- `a38edc4` semantic-hunk PI-v2 and frozen prepared-baseline blob map
- `ca9001c` graceful hunk chunking and coverage accounting
- `1f90d10` real Docker PI-v2 audit
- `d02c7ea` export serializer and final deployment source

Checks completed before launch:

- Server tests: 54 passed, 2 warnings.
- Real Docker smoke: one controlled file change serialized as one semantic hunk, with 1/1 changed lines included under the 4096-token budget.
- Full Python environment audit: 124/124 images passed.
- GitHub branch pushed: `experiment/swe-sandbox-pi-v2-20260915`.

Runtime:

- tag: `capo-swe-pythonfull-v7-p2-4b-gpu03-20260915-01`
- GPUs: 0,1,2,3
- port: 18701
- data: python-full-v7, 6248 train / 470 validation
- training: 4 epochs, 784 steps, batch 32, microbatch 2
- PI: `semantic_hunks_v2`, max 4096 tokens, safety margin 32

The previous P1-v1 screen was explicitly stopped before this launch. The V2 run passed task-branch checks and reached the initial 470-task validation; at the first status check it had completed 17/470 validation rollouts with no execution failures.
