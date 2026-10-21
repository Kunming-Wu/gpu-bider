---
name: gpu-bider
description: Maintain a queue of experiments waiting for GPU time, watch the card, and start the next one the moment it fits. Use when the user wants to queue training or inference runs, asks to add, list, reorder or cancel a pending experiment, wants to know when their job can start or why it has not, wants something started automatically once another job finishes, or asks how long a running job has left.
when_to_use: The user mentions a queue of experiments, waiting for a GPU, "run this after that finishes", "tell me when the GPU is free", "why hasn't my job started", pending or running experiments, sharing a GPU with colleagues, or an ETA for a job that is currently running.
argument-hint: "[add|list|watch|status|probe|explain|memory|intent|stats|estimate] [details]"
allowed-tools: Read, Write, Edit, Glob, Grep, Bash, TaskOutput, TaskStop
disallowed-tools: NotebookEdit
---

# gpu-bider

One command drives everything: `bider`. It keeps a queue in `~/.bider/queue.json`,
what it has learned about the machine in `~/.bider/memory.json`, and reads the
card with `nvidia-smi`.

    bider --help

**Never run `bider watch` in the foreground.** It blocks until a task can start,
which can be hours. Always start it with the Bash tool's `run_in_background`
option and keep the task id.

## The two rules this tool exists to enforce

1. **It never touches another process.** No signals, no preemption, no
   killing. It waits for the card and uses it while it is free.
2. **Between bider users, waiting is queued, not raced.** A claim is written
   before a job starts, so two bider instances cannot both decide a free card is
   theirs and OOM each other. This is what makes it safe to deploy on a shared
   server; do not disable it without saying so.

Say this out loud when the user is deciding whether to install it. The value
is not "gets the card faster", it is "takes only what nobody else wants".

## The queue

Each task carries `vram_gb`, `gpus`, and `util_pct` in its `cost`.

**Run order: pinned first, then explicit `rank`, then cheapest-first**
(ascending `gpus x vram x util`), with `priority` only breaking ties. Cheapest
first is deliberate - a thirty-second smoke test should not sit behind a
six-hour training run - and memory profiles bias it further (below). When the
user says a specific experiment must go first, set that explicitly; do not
invent an ordering.

```bash
bider list                                  # in the order it will run
bider add "python train.py --lr 1e-4" --name "lr sweep" \
    --vram 3.5 --util 60 --est-min 45    # --gpus N, --workdir DIR, --env K=V
bider pin "lr sweep"                        # before everything else
bider update "lr sweep" --set vram_gb=4
bider remove "lr sweep"                     # prefixes are fine
bider policy --set reserve_vram_gb=2.5
bider runs                                  # what ran, and how long it took
```

Estimate `--vram` and `--util` from what the user says, or from the code
(batch size, model size, precision), and state the assumption. Pessimistic is
safe - it only delays the start. Optimistic can wedge a card that is also
driving somebody's display. `bider runs` shows what earlier runs actually cost.

## Watching, and reporting back

Start it in the background (Bash tool, `run_in_background: true`):

```bash
bider watch --interval 20 --timeout-min 60
```

| exit | verdict | meaning |
|---|---|---|
| 0 | `TRIGGERED` | a task fits right now - go start it |
| 2 | `TIMEOUT` | nothing changed within `--timeout-min` |
| 3 | `NOTHING_PENDING` | queue is empty |

The payload sits between `BIDER_TRIGGER` and a line of `=`. When a background
watch finishes, the harness wakes you with its output. Then:

1. Read it: `task_id`, the arithmetic under `headroom` (including any
   `claimed_gb`), `changes_since_start` (usually `proc_exited` with the pid
   and name of whatever finished), and `memory` blocks if any.
2. Start it: `bider run <task_id>`. `run` re-checks the card and refuses if it no
   longer fits; add `--detach` to return immediately.
3. Tell the user in one or two lines what started and why it was judged safe.
4. **Re-arm** the background watch for the next task, and repeat until the
   queue is empty or the user stops asking.

Useful flags: `--task <id>` for one specific task; `--require-change` to fire
only if something actually changed since the watch began (that is how you wait
for *somebody finished* rather than for idle capacity); `--stay-alive` to keep
watching an empty queue. If a background watch runs more than about ten
minutes without triggering, re-arm it - sessions recycle long-running shells.
`--on-trigger run` starts the task with no model in the loop, for when the
user wants the machine to keep working while Claude Code is closed.

## Explaining a wait

```bash
bider explain                  # for the head of the queue
bider explain --task lr-sweep
```

Answers "why isn't my job running" in prose, with the numbers: the reserved
window, the protected colleague, the shortfall, who holds the card, who
registered first, and a projected finish time when one can be derived. Use
this instead of reconstructing an answer yourself - and use it whenever the
user asks why something is waiting.

## Memory - what the machine knows about the others

`memory.json` holds facts that outrank the heuristics: a protected process or
an in-force window is a hard block, not a hint.

```bash
bider memory                                   # everything remembered
bider memory --protect wei --match loro_train \
    --because "deadline this week" --until 2026-10-22
bider memory --window wei --from 14:00 --to 18:00 --days mon-fri
bider memory --profile train --profile-vram heavy --profile-util light
bider memory --suggest                         # learn shapes from evidence
bider memory --forget window-wei
bider memory --export -                       # dump it; --import-file reads it back
```

* **protected** - never disturb this person's processes. It applies *only
  while one of them is actually on the card*; when they are not running you
  are free to use the machine, and the entry survives so a later session still
  knows about them. `--until` lets it lapse on its own. `--task` scopes it to
  one of our own runs, `--private` keeps it out of the shared directory.
* **window** - recurring time reserved for someone, possibly crossing midnight
  (`--from 22:00 --to 02:00`). The watcher will not start a task inside one,
  and will not begin a run in the last few minutes before one opens. Note the
  honest limit: without preemption it can only decline to *occupy* - it cannot
  make a job already running stop.
* **profile** - how a process family behaves. `--vram heavy --util light`
  describes something that holds memory but barely computes; when such a
  process is on the card the queue is reordered toward tasks shaped the
  opposite way, so the two coexist instead of contending. Advice biases
  ordering; `pin` and `rank` always win.

**When the user mentions anything like this, record it.** "Don't touch Wei's
job", "X needs the card 2-6pm", "that eval script hogs memory but not compute"
are one command each, and the machine then behaves accordingly without being
told again. Say what you recorded. Prefer `--profile-util`/`--profile-vram`
claims the user actually made over guesses; `--suggest` only promotes a family
after several polls agree and marks it `"source": "observed"`.

## Sharing the card with colleagues

```bash
bider config --set shared_dir=/path/everyone/can/see
bider config --set owner=yourname
bider config                                   # shows whether it is readable
```

Every `bider` pointed at the same directory works together - claims, published
memory, and the intent board all live there. Without it, bider is single-user and
safe on a personal machine but blind to everyone else.

`bider intent` shows who is waiting for what, and `bider intent publish` adds us.
Publishing is optional and read-only to others: it never blocks anybody by
itself, but when somebody registered first for room we both want, the watcher
stands down for them rather than grabbing the last free gigabyte. `bider ledger`
is the audit trail of starts, yields and refusals, with reasons.

## ETA for a running job

```bash
bider estimate --pid 12345 --log path/to/job.log --json
bider estimate --run lr-sweep-20261019-140825 --json
```

Matches `42%|`, `21/50`, `Epoch 3/10`, `step 250/1000`, bare `%`, and projects
from the recent rate, discarding samples from before a restart. `eta_human` is
the quotable field, `basis` says what it measured. A null `eta_s` with a `note`
means no progress marker was found - ask where progress is printed rather than
guessing. One sample is not enough: use `--for-seconds 60 --interval 5`, or
start `watchproc` in the background.

The watcher also uses this internally, so a job about to finish is polled
faster; `foreign_eta_s` in the decision is that projection, and it is a guess
about somebody else's job. Do not present it as a promise.

## This machine

An RTX 3060 Laptop, 6GB, which **also drives the display**. The default policy
holds back `reserve_vram_gb=2.0` and `reserve_util_pct=25`: a task starts only
if, after its own appetite, that much remains. A 5GB task will therefore never
start here, even on an idle card.

Do not raise the reserves unless the user asks. Do not run a stress test or a
GPU-saturating job to check that the watcher works - verify against
`BIDER_FAKE_GPU` instead, which takes inline JSON or a path to a snapshot and
replaces the real card for every read-only command:

```bash
bider selftest                # scheduling, claims, memory, ETA maths
python test_e2e.py         # watcher edge-triggering, two-user claims, CLI
```

## Files

```
~/.bider/queue.json              the queue
~/.bider/memory.json             what we know about other users
~/.bider/config.json             owner, shared_dir, gpu_index, notify_cmd
~/.bider/state/ledger.jsonl      every start, yield and refusal, with reasons
~/.bider/state/watcher.json      the last watch, for `bider watch --status`
~/.bider/runs/<task>-<stamp>/    stdout.log + meta.json per started run
$shared_dir/claims/           reservations, and the intent board
$shared_dir/memory/           what each user has published
```

Both JSON files are safe to edit by hand when the CLI is awkward.
