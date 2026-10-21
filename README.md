<div align="center">

<img src="docs/assets/logo.svg" alt="gpu-bider" width="104" height="104">

# gpu-bider

**A polite GPU job queue.** It waits for the card, uses it the moment it is
free, and never touches anybody else's process.

[![CI](https://github.com/Kunming-Wu/gpu-bider/actions/workflows/ci.yml/badge.svg)](https://github.com/Kunming-Wu/gpu-bider/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.11%20%7C%203.12-2a78d6)](https://github.com/Kunming-Wu/gpu-bider)
[![Dependencies](https://img.shields.io/badge/dependencies-none-0ca30c)](pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-52514e)](#)
[![License](https://img.shields.io/badge/license-MIT-2a78d6)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-2%20suites%2C%20no%20GPU%20required-0ca30c)](RELEASING.md)

[Website](docs/index.html) · [Limitations](#limitations-stated-plainly) · [How it decides](#how-it-decides) · [Sharing a card](#sharing-the-card)

</div>

---

## The problem

You share a GPU box with three other people. You have six experiments to run.
The card is busy for the next two hours, so you open a terminal, watch
`nvidia-smi`, and get on with something else — and by the time you come back
the card has been idle for forty minutes and somebody else has taken it.

The naive fix is a script that polls and grabs. That works until two people
install it, and then it makes things *worse*: both see the same free card, both
decide it fits, both start, and one of them dies with a CUDA OOM. A decision
is not visible in `nvidia-smi` until seconds later, and nothing in the loop
covers that gap.

gpu-bider is what I wanted instead.

## What it does

```
your job ends ──┐
                ├──▶  card is free  ──▶  the next queued experiment starts
colleague's ────┘                          (and it records why that was safe)
```

It keeps a queue of experiments, watches the card, and starts the next one the
moment it fits. Two rules hold everywhere in the code:

> **1. It never touches another process.** No signals, no preemption, no
> killing. If somebody else is running, it is simply not your turn yet.
>
> **2. Between gpu-bider users, waiting is queued, not raced.** A *claim* is
> written before anything starts, so two people cannot both decide the same
> free card is theirs.

Everything else follows from those two: it is safe to deploy, safe to leave
running, and safe for a colleague to install next to yours.

## Install

```bash
pipx install git+https://github.com/Kunming-Wu/gpu-bider
```

<details>
<summary>Without pipx, or without installing at all</summary>

```bash
git clone https://github.com/Kunming-Wu/gpu-bider && cd gpu-bider
python install.py --check     # diagnostics: python, nvidia-smi, selftest
python install.py             # seeds ~/.bider and writes ./bider + ./bider.cmd
```

`install.py` never writes outside `~/.bider` and the project directory.
</details>

Optional, to let Claude Code drive it:

```bash
python install.py --skill     # copy the skill into ./.claude/skills
```

There is no daemon, no root, no account, and no network call anywhere in the
codebase. The only third-party dependency is `psutil`, and only if you want ETA
estimates for *other people's* jobs:

```bash
pipx install "gpu-bider[eta]"
```

## Use

```bash
bider add "python train.py --lr 1e-4" --name "lr sweep" --vram 3.5 --util 60
bider list                    # the order it will run in
bider watch --stay-alive      # wait for the card
bider run "lr sweep"          # ... or start it by hand right now
bider status                  # gpu + queue + watcher + memory, one screen
bider explain                 # why hasn't my job started?
bider stats                   # what this card has actually been doing
```

Through Claude Code, just say what you want:

> queue a run of `python train.py --lr 3e-4`, it needs about 3GB, and tell me
> when it can start

Claude Code works out the footprint, adds it, and puts a watch in the
background. When the card frees up it starts the job and tells you. Ask
*"put the lr sweep first"*, *"cancel the profile run"*, or *"that training job
has been going 40 minutes, how much longer?"* and it does that instead.

### Exit codes from `watch`

| code | verdict | meaning |
|:---:|---|---|
| `0` | `TRIGGERED` | a task fits right now — go start it |
| `2` | `TIMEOUT` | nothing changed within `--timeout-min` |
| `3` | `NOTHING_PENDING` | the queue is empty |
| `4` | — | error (no `nvidia-smi`, etc.) |

On a trigger it prints a JSON payload between `BIDER_TRIGGER` and a line of
`=`, carrying the task, the headroom arithmetic (including what others have
already claimed), and what changed since the watch began — usually
`proc_exited`, with the pid and name of whatever finished.

<details>
<summary>Example payload</summary>

```json
{
  "verdict": "TRIGGERED",
  "task_id": "lr-sweep",
  "decision": {
    "gpu": { "mem_free_gb": 5.85, "util_pct": 0.0 },
    "headroom": {
      "free_gb": 5.85, "need_gb": 3.5, "vram_after_gb": 2.35,
      "vram_reserve_gb": 2.0, "claimed_gb": 0.0
    },
    "reasons": ["fits: 2.35GB vram and 40.0% util left for the desktop"]
  },
  "changes_since_start": [
    { "kind": "proc_exited",
      "procs": [{ "pid": 4242, "name": "someone_else.exe" }] }
  ],
  "next_step": "Deploy task 'lr-sweep'. The intended command is:\n    bider run lr-sweep"
}
```
</details>

## How it decides

Checked in this order. The first one that says no wins.

```
   ┌──────────────────────────────────────────────────────────────────┐
   │  1  memory      a protected colleague, or a reserved window      │
   │  2  claims      what other bider instances already reserved      │
   │  3  capacity    free − need ≥ reserve, for vram and for util     │
   │  4  first fit   head does not fit → try the next task            │
   │  5  intent      somebody registered first for the same room      │
   └──────────────────────────────────────────────────────────────────┘
```

The reserve defaults to **2 GB and 25%** held back from every task, because on
most of these boxes the GPU is also driving the display. That is why a 5 GB
task will not start on a 6 GB card even when the card looks idle — by design.

`bider probe` is the one-shot version and prints the arithmetic; `bider explain`
turns the same facts into prose. Both are worth reading before believing any
scheduling decision.

### Run order

1. `pinned` tasks first
2. then explicit `rank`
3. then **ascending footprint** (`gpus × vram × util`) — cheapest first, biased
   by memory profiles
4. `priority` only breaks ties

Cheapest-first is deliberate: a thirty-second smoke test should not sit behind
a six-hour training run. Override it with `bider pin <id>` or
`bider move <id> --top`.

## Sharing the card

Point every `bider` at one directory the whole group can see:

```bash
bider config --set shared_dir=/shared/gpucard
bider config --set owner=yourname
```

That directory holds three things:

| | |
|---|---|
| `claims/` | reservations, one file per job, with a TTL. Two colleagues starting at the same instant cannot both win; the loser re-evaluates on the next tick. A claim whose process died is dropped after a short grace period, so a crash cannot hold the card hostage. |
| `memory/` | what each person has published about their own needs — so you can see their deadline and stay out of its way. |
| intent board | `bider intent` shows who is waiting for what. |

Without `shared_dir`, bider is single-user: still useful on a personal machine,
but blind to everyone else.

### The intent board

`bider intent publish` says *"I am waiting for 5 GB."* It is **read-only to
others and never blocks anybody by itself.** What it buys is information: when
two people both want the last 5 GB and one registered first, the other's
watcher sees that and stands down instead of racing for it. People who never
publish still appear, as claims, once they start.

## Memory — what the machine knows about the others

The most useful part, and the reason this is not just a polling loop.

```bash
bider memory --protect wei --match loro_train \
      --because "deadline this week" --until 2026-10-22

bider memory --window wei --from 14:00 --to 18:00 --days mon-fri

bider memory --profile train --profile-vram heavy --profile-util light

bider memory --suggest        # learn shapes from what it has observed
```

<table>
<tr><th align="left">kind</th><th align="left">meaning</th></tr>
<tr><td><b>protected</b></td><td>Never disturb this person's processes. Applies only while one of them is actually <i>on the card</i> — when they are not running you are free to use the machine, and the entry survives so a later session still knows about them. <code>--until</code> lets it lapse; <code>--task</code> scopes it to one of your own runs; <code>--private</code> keeps it out of the shared directory.</td></tr>
<tr><td><b>window</b></td><td>Recurring time reserved for someone, possibly crossing midnight (<code>--from 22:00 --to 02:00</code>). The watcher will not start a task inside one, and will not begin a run in the last few minutes before one opens — no point starting a two-hour job at 13:55 before a 14:00 reservation.</td></tr>
<tr><td><b>profile</b></td><td>How a process family behaves. <code>--vram heavy --util light</code> describes something that holds memory but barely computes, so the queue is reordered toward tasks shaped the opposite way. Both jobs then coexist instead of contending. Advice only biases ordering — <code>pin</code> and <code>rank</code> win.</td></tr>
</table>

Protections and windows are **hard blocks**; profiles are **hints**. Every
decision is reported in `probe`, `status`, `explain` and the watcher payload,
with the reason, so nothing is mysterious. `bider observe --suggest` promotes a
family to a profile only after several polls agree, and marks it
`"source": "observed"` so you can tell evidence from a statement.

> **The honest limit.** Without preemption, a window can only stop bider from
> *occupying* the card. It cannot make a job that is already running stop. If
> you need the card back for a reservation that just began, bider will tell you
> and decline to add more — stopping your own job is your call.

## ETA

```bash
bider estimate --pid 1234 --log job.log --json
bider estimate --run lr-sweep-20261019-140825 --json
```

Reads the log tail, matches `42%|`, `21/50`, `Epoch 3/10`, `step 250/1000` and
bare percentages, then projects a finish from the rate the number is moving at
*now*, discarding samples from before a restart.

**No progress marker means no ETA.** It returns `null` and says what it looked
for, rather than guessing.

Drop a task's log into `~/.bider/logs/` (or symlink it) and the watcher picks
it up automatically — including, where permissions allow, for other people's
jobs. That is what lets it poll faster as a colleague's run nears its end.

## Limitations, stated plainly

- **On Windows/WDDM, `nvidia-smi` reports `[N/A]` for per-process GPU memory.**
  Bider therefore never attributes memory to a process. It diffs whole-GPU
  totals and diffs the set of compute PIDs, both of which are reliable. On
  Linux, NVML does report per-process memory and estimates get better.
- **Inside a container, pid attribution can be wrong** — `nvidia-smi` may
  report host pids. Protected-process matching then fails *open*: it will not
  block.
- **A shared directory over NFS has unreliable locking.** Claims are therefore
  TTL-based as well as locked: a lost race costs a duplicate write and a
  re-evaluation, never a stuck card.
- **Pids get reused.** A claim guards against this with a TTL and a liveness
  check; a very unlucky reuse could still mislead a protected-process match.
- **Utilization is not a reservation.** Two jobs each claiming 50% do not
  cleanly share a card — they contend. The util figure is a politeness margin,
  not a partition. Real isolation needs MIG or MPS, which is a different tool.
- **Nobody is obliged to use this.** Someone who has never heard of bider can
  start a job whenever they like. Bider is built to lose gracefully when that
  happens.

## Tests

```bash
bider selftest       # scheduling, claims, memory, windows, ETA maths
python test_e2e.py   # watcher edge-triggering, two-user claims, CLI, sharing
```

**Neither touches the GPU.** Set `BIDER_FAKE_GPU` to a JSON snapshot and every
read-only command runs against that instead:

```bash
BIDER_FAKE_GPU='{"gpus":[{"index":0,"name":"fake","mem_total_mb":6144,
  "mem_free_mb":512,"util_pct":99}],
  "procs":[{"pid":4242,"name":"other.exe"}]}' \
  bider probe
```

That is how the watcher is tested without loading the card that is currently
drawing somebody's screen. CI runs both suites on Linux, macOS and Windows
across Python 3.8, 3.11 and 3.12.

## Files

```
bider.py                 the whole program — one file, stdlib only
install.py               one-shot setup and diagnostics
docs/index.html          the project page
skill/gpu-bider/         the Claude Code skill, authored here
~/.bider/queue.json          the queue
~/.bider/memory.json         what we know about other users
~/.bider/config.json         owner, shared_dir, gpu_index, notify_cmd
~/.bider/state/ledger.jsonl  every start, yield and refusal, with reasons
~/.bider/logs/               drop task logs here for ETA estimates
~/.bider/runs/<task>-<stamp>/  stdout.log + meta.json per started run
$shared_dir/claims/          reservations, and the intent board
$shared_dir/memory/          what each user has published
```

`queue.json` and `memory.json` are plain JSON and safe to edit by hand when the
CLI is awkward.

## Deliberately out of scope

Recorded so they are decisions, not oversights.

| | |
|---|---|
| **Preemption** | Never. gpu-bider does not stop other people's work. |
| **A server** | Sharing is a directory. No daemon, no port, nothing to be granted. |
| **MIG / MPS** | Real partitioning is a different tool. This manages time, not hardware slices. |
| **Fair-share quotas** | A group that needs quotas and rotation wants Slurm. The niche here is the small shared box that has no admin. |

---

<div align="center">

MIT licensed. See [`RELEASING.md`](RELEASING.md) for the release checklist.

</div>
