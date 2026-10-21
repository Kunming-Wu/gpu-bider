#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bider.py - a polite GPU job queue.

Single-file, stdlib-only. Keeps a queue of experiments waiting for GPU time,
watches the card, and starts the next one the moment it fits.

The principles
--------------
1. It never touches another process.  No signals, no preemption, no killing.
   It waits for the card and uses it while it is free.
2. Between bider users on one machine, waiting is queued and visible, not a
   race.  Claims in a shared directory mean two people cannot both decide a
   free card is theirs and OOM each other.
3. It remembers who else needs the card - a colleague with a deadline, a
   recurring block of reserved time - and gets out of the way.  A protected
   process or an in-force window is a hard block, not a hint.

It has no dependencies beyond the standard library and nvidia-smi, needs no
root, and no server.  Setting `shared_dir` to a directory two machines can
both see is the only step needed to make it multi-host.

Exit codes for `probe` / `watch`:
  0  TRIGGERED      - a task can be started right now
  2  TIMEOUT        - waited `--timeout-min`, nothing changed
  3  NOTHING_PENDING- nothing left to run
  4  ERROR
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

# --------------------------------------------------------------------------
# paths and configuration
# --------------------------------------------------------------------------
# Everything lives under BIDER_HOME (default ~/.bider).  Keeping state next to the
# script is fine for a checkout and wrong for an installed tool, and the
# difference only shows up on somebody else's machine.

VERSION = "2.0.0"


def _default_home():
    return os.environ.get("BIDER_HOME") or os.path.join(
        os.path.expanduser("~"), ".bider")


HOME = os.path.abspath(_default_home())
QUEUE_PATH = os.path.join(HOME, "queue.json")
MEMORY_PATH = os.path.join(HOME, "memory.json")
CONFIG_PATH = os.path.join(HOME, "config.json")
STATE_DIR = os.path.join(HOME, "state")
RUNS_DIR = os.path.join(HOME, "runs")
LOGS_DIR = os.path.join(HOME, "logs")
OBS_PATH = os.path.join(STATE_DIR, "observations.json")
LEDGER_PATH = os.path.join(STATE_DIR, "ledger.jsonl")
WATCHER_STATE = os.path.join(STATE_DIR, "watcher.json")
WATCHER_LOG = os.path.join(STATE_DIR, "watcher.log")
WATCHER_PID = os.path.join(STATE_DIR, "watcher.pid")
LOCK_PATH = os.path.join(STATE_DIR, "queue.lock")


def set_home(path):
    """Point every path at a different home.  Used by --home and by tests."""
    global HOME, QUEUE_PATH, MEMORY_PATH, CONFIG_PATH, STATE_DIR, RUNS_DIR
    global LOGS_DIR, OBS_PATH, LEDGER_PATH, WATCHER_STATE, WATCHER_LOG
    global WATCHER_PID, LOCK_PATH
    HOME = os.path.abspath(path)
    QUEUE_PATH = os.path.join(HOME, "queue.json")
    MEMORY_PATH = os.path.join(HOME, "memory.json")
    CONFIG_PATH = os.path.join(HOME, "config.json")
    STATE_DIR = os.path.join(HOME, "state")
    RUNS_DIR = os.path.join(HOME, "runs")
    LOGS_DIR = os.path.join(HOME, "logs")
    OBS_PATH = os.path.join(STATE_DIR, "observations.json")
    LEDGER_PATH = os.path.join(STATE_DIR, "ledger.jsonl")
    WATCHER_STATE = os.path.join(STATE_DIR, "watcher.json")
    WATCHER_LOG = os.path.join(STATE_DIR, "watcher.log")
    WATCHER_PID = os.path.join(STATE_DIR, "watcher.pid")
    LOCK_PATH = os.path.join(STATE_DIR, "queue.lock")


DEFAULT_POLICY = {
    "reserve_vram_gb": 2.0,    # memory kept free for the desktop / other work
    "reserve_util_pct": 25,    # utilization kept free, as a floor not a budget
    "poll_seconds": 20,        # idle poll interval
    "poll_floor_seconds": 3,   # fastest we ever poll
    "poll_ceiling_seconds": 120,  # slowest we ever poll
    "settle_ticks": 2,         # consecutive good polls before triggering
    "allow_concurrent": False, # may our own tasks overlap each other?
    "first_fit": True,         # skip a task that does not fit and try the next?
    "require_claim": True,     # reserve through the shared directory first
    "claim_ttl_s": 120,        # a claim with no live process is dropped after this
    "polite": True,            # back off when another person is on the card
}

DEFAULT_CONFIG = {
    "owner": "",              # who you are, for the claim files
    "shared_dir": "",         # "" = claims are local to this machine only
    "gpu_index": None,        # which card to watch, when there is more than one
    "notify_cmd": "",         # run on every ledger event, e.g. a curl to Slack
    "timezone": "",           # label only; times are always local time
}

MEMORY_KINDS = ("protected", "window", "profile", "intent")

WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _fix_stdout():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def now_epoch():
    return time.time()


def log(msg, path=None, echo=True):
    line = "[%s] %s" % (now_iso(), msg)
    if echo:
        print(line, flush=True)
    if path:
        ensure_dir(os.path.dirname(path))
        with open(path, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(line + "\n")


def ensure_dir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)


def atomic_write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def append_jsonl(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def load_config():
    c = read_json(CONFIG_PATH, None) or {}
    for k, v in DEFAULT_CONFIG.items():
        c.setdefault(k, v)
    if not c["owner"]:
        c["owner"] = _default_owner()
    return c


def save_config(c):
    atomic_write_json(CONFIG_PATH, c)


def _default_owner():
    for key in ("BIDER_OWNER", "USERNAME", "USER", "LOGNAME"):
        v = os.environ.get(key)
        if v:
            return re.sub(r"[^A-Za-z0-9_.-]", "_", v)[:32]
    try:
        import getpass
        return re.sub(r"[^A-Za-z0-9_.-]", "_", getpass.getuser())[:32]
    except Exception:
        return "unknown"


def host_name():
    return re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname() or "host")[:32]


class FileLock(object):
    """Tiny cross-process lock.

    Good enough for one machine.  Across NFS this is advisory and unreliable,
    which is why claims also carry a TTL: a lock that goes wrong costs a
    duplicate write, never a stuck card.
    """

    def __init__(self, path, timeout=10.0, stale=30.0):
        self.path = path
        self.timeout = timeout
        self.stale = stale
        self.fd = None

    def __enter__(self):
        ensure_dir(os.path.dirname(self.path))
        deadline = time.time() + self.timeout
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except OSError:
                try:
                    if time.time() - os.path.getmtime(self.path) > self.stale:
                        os.unlink(self.path)
                        continue
                except OSError:
                    continue
                if time.time() > deadline:
                    raise RuntimeError("could not acquire lock %s" % self.path)
                time.sleep(0.1)

    def __exit__(self, *exc):
        try:
            if self.fd is not None:
                os.close(self.fd)
        except OSError:
            pass
        try:
            os.unlink(self.path)
        except OSError:
            pass
        self.fd = None
        return False


# --------------------------------------------------------------------------
# platform
# --------------------------------------------------------------------------

IS_WINDOWS = os.name == "nt"


def pid_alive(pid):
    """Is this pid running?  Used for claims and for our own tasks."""
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if IS_WINDOWS:
        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", "PID eq %d" % pid, "/NH"],
                stderr=subprocess.DEVNULL, timeout=10)
            return str(pid) in out.decode("utf-8", "replace")
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError as e:
        import errno
        return getattr(e, "errno", None) == errno.EPERM
    except Exception:
        return False


def kill_tree(pid):
    """Stop a process we started.  Never used on anybody else's."""
    if not pid:
        return
    try:
        if IS_WINDOWS:
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(int(pid))],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            import signal
            os.kill(int(pid), signal.SIGTERM)
    except Exception:
        pass


def detached_flags():
    """creationflags that outlive the parent, where the platform has them."""
    if IS_WINDOWS:
        return 0x00000008 | 0x00000200  # DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP
    return 0


def spawn_detached(cmd, cwd, stdout_fh):
    return subprocess.Popen(
        cmd, cwd=cwd, stdout=stdout_fh, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, close_fds=True,
        creationflags=detached_flags(), start_new_session=(not IS_WINDOWS))


def run_notify(config, event):
    """Fire the configured notification hook.  Best effort, never fatal."""
    cmd = (config or {}).get("notify_cmd")
    if not cmd:
        return
    try:
        subprocess.Popen(cmd, shell=True,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env=dict(os.environ,
                                  BIDER_EVENT=event.get("event", ""),
                                  BIDER_TASK=event.get("task_id", "") or ""))
    except Exception:
        pass



# --------------------------------------------------------------------------
# GPU sampling
# --------------------------------------------------------------------------

def find_nvidia_smi():
    p = shutil.which("nvidia-smi")
    if p:
        return p
    for cand in (
        os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"),
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ):
        if os.path.isfile(cand):
            return cand
    return None


def _num(s):
    s = (s or "").strip()
    if s in ("", "N/A", "[N/A]", "Not Supported", "[Not Supported]"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _smi(path, query, tries=2):
    """One nvidia-smi query.  A single transient failure is retried, because
    a hiccup that reads as "the card is empty" would start a job onto a busy
    card - the one mistake this tool must never make."""
    last = None
    for _ in range(max(1, tries)):
        try:
            out = subprocess.check_output(
                [path, "--query-%s" % query, "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=15,
            )
        except Exception as e:
            last = e
            time.sleep(0.2)
            continue
        rows = []
        for line in out.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if line:
                rows.append([c.strip() for c in line.split(",")])
        return rows
    raise RuntimeError("nvidia-smi failed: %s" % last)


def _fake_snapshot():
    """Test hook: BIDER_FAKE_GPU points at a JSON file (or holds inline JSON)."""
    spec = os.environ.get("BIDER_FAKE_GPU", "").strip()
    if not spec:
        return None
    if os.path.isfile(spec):
        data = read_json(spec, None)
        if data is None:
            raise RuntimeError("BIDER_FAKE_GPU file unreadable: %s" % spec)
        return data
    return json.loads(spec)


def snapshot():
    """Return {ts, gpus:[...], procs:[...], source, error}."""
    fake = _fake_snapshot()
    if fake is not None:
        fake = dict(fake)
        fake.setdefault("source", "fake")
        fake.setdefault("ts", time.time())
        fake.setdefault("error", None)
        return fake

    path = find_nvidia_smi()
    if not path:
        return {"ts": time.time(), "gpus": [], "procs": [], "source": "none",
                "error": "nvidia-smi not found"}

    try:
        gpus = []
        for r in _smi(path, "gpu=index,uuid,name,driver_version,memory.total,"
                            "memory.used,memory.free,utilization.gpu,temperature.gpu"):
            r = (r + [""] * 9)[:9]
            gpus.append({
                "index": int(_num(r[0]) or 0),
                "uuid": r[1],
                "name": r[2],
                "driver": r[3],
                "mem_total_mb": _num(r[4]),
                "mem_used_mb": _num(r[5]),
                "mem_free_mb": _num(r[6]),
                "util_pct": _num(r[7]),
                "temp_c": _num(r[8]),
            })
            g = gpus[-1]
            if g["mem_free_mb"] is None and g["mem_total_mb"] is not None \
                    and g["mem_used_mb"] is not None:
                g["mem_free_mb"] = g["mem_total_mb"] - g["mem_used_mb"]
        procs = []
        for r in _smi(path, "compute-apps=gpu_uuid,pid,process_name,used_memory"):
            r = (r + [""] * 4)[:4]
            procs.append({
                "gpu_uuid": r[0], "pid": int(_num(r[1]) or -1),
                "name": r[2], "mem_mb": _num(r[3]),
            })
        snap = {"ts": time.time(), "gpus": gpus, "procs": procs,
                "source": path, "error": None}
        only = load_config().get("gpu_index")
        if only is not None:
            snap = _restrict_to_gpu(snap, only)
        return snap
    except Exception as e:
        return {"ts": time.time(), "gpus": [], "procs": [], "source": path,
                "error": str(e)}


def _restrict_to_gpu(snap, index):
    """Narrow a multi-card machine to one card.

    Everything downstream assumes one pool, so this is the one place that has
    to know the difference.  Claims also record the card, so two people on
    different cards of the same host do not block each other.
    """
    try:
        index = int(index)
    except (TypeError, ValueError):
        return snap
    keep = [g for g in snap["gpus"] if g["index"] == index]
    if not keep:
        return snap
    uuid = None
    for g in keep:
        uuid = g.get("uuid") or uuid
    out = dict(snap)
    out["gpus"] = keep
    out["procs"] = [p for p in snap["procs"] if p.get("gpu_uuid") is None
                    or uuid is None or p.get("gpu_uuid") == uuid]
    out["gpu_index"] = index
    return out


def gpu_aggregate(snap):
    """Collapse a snapshot into the numbers the policy cares about."""
    gpus = snap.get("gpus") or []
    if not gpus:
        return None
    total = sum(g["mem_total_mb"] or 0 for g in gpus)
    used = sum(g["mem_used_mb"] or 0 for g in gpus)
    free = sum(g["mem_free_mb"] or 0 for g in gpus)
    utils = [g["util_pct"] for g in gpus if g["util_pct"] is not None]
    return {
        "n_gpus": len(gpus),
        "mem_total_gb": total / 1024.0,
        "mem_used_gb": used / 1024.0,
        "mem_free_gb": free / 1024.0,
        "util_pct": (sum(utils) / len(utils)) if utils else 0.0,
        "max_util_pct": max(utils) if utils else 0.0,
    }


def foreign_procs(snap, config=None):
    """Compute processes on the card that are not ours.

    'Ours' is decided by pid, not by name: our own children are the ones we
    already know about, and guessing from a name would classify a colleague
    running the same script as us.
    """
    ours = set(int(t.get("pid") or 0) for t in running_tasks(load_queue()))
    return [p for p in (snap.get("procs") or [])
            if p.get("pid") not in ours and p.get("pid", -1) > 0]


# --------------------------------------------------------------------------
# claims: the reservation layer that keeps bider users from colliding
# --------------------------------------------------------------------------
# Without this, two bider users see the same free card, both decide it fits, and
# both start - the bug is that a decision is not visible in nvidia-smi until
# seconds later.  So a decision is written down first, in a directory every
# bider on the machine (or every bider pointed at the same shared_dir) reads.
#
# A claim is one file per (owner, host, pid).  Distinct files mean two
# processes never write the same path, so no lock is needed to record one;
# only the read-modify-write of a whole *decision* needs the lock, and a lost
# race there just means we re-evaluate on the next tick.

CLAIM_GRACE_S = 20.0     # a claim may briefly outlive the pid it was taken for


def _claim_glob(d):
    try:
        return [os.path.join(d, n) for n in os.listdir(d) if n.endswith(".json")]
    except OSError:
        return []


def claim_expired(c, now=None):
    now = now or now_epoch()
    if c.get("expires_at") and c["expires_at"] < now:
        return True
    # a claim made on this host for a pid that is gone was either abandoned
    # or already finished; a starting process has a few seconds of grace
    if c.get("host") == host_name() and c.get("pid"):
        if not pid_alive(c["pid"]) and (now - c.get("at_epoch", 0)) > CLAIM_GRACE_S:
            return True
    return False


def load_claims(config=None, release_mine=None):
    """All live claims.  Expired ones are dropped (and the files removed)."""
    config = config or load_config()
    d = shared_claims_dir(config)
    now = now_epoch()
    out = []
    for path in _claim_glob(d):
        c = read_json(path)
        if not isinstance(c, dict) or "owner" not in c:
            continue
        if claim_expired(c, now):
            if release_mine is None or c.get("owner") == config.get("owner"):
                try:
                    os.unlink(path)
                except OSError:
                    pass
            continue
        c["_path"] = path
        out.append(c)
    return out


def claims_totals(claims, exclude_owner=None):
    """What the live claims add up to, as if it were already on the card."""
    vram = util = 0.0
    gpus = 0
    n = 0
    for c in claims:
        if exclude_owner and c.get("owner") == exclude_owner:
            continue
        vram += float(c.get("vram_gb") or 0)
        util += float(c.get("util_pct") or 0)
        gpus += int(c.get("gpus") or 1)
        n += 1
    return {"vram_gb": round(vram, 2), "util_pct": round(util, 1),
            "gpus": gpus, "n": n}


def take_claim(task, config=None, ttl=None, pid=None, note=""):
    """Reserve the card for a task we are about to start.  Best effort.

    The claim is what the *next* decider reads, so it must be written before
    the process starts and is deliberately cheap to make.
    """
    config = config or load_config()
    pol = load_queue()["policy"]
    ttl = float(ttl or pol.get("claim_ttl_s", 120))
    d = shared_claims_dir(config)
    ensure_dir(d)
    c = task.get("cost") or {}
    claim = {
        "owner": config.get("owner"),
        "host": host_name(),
        "pid": pid,
        "task_id": task.get("id"),
        "vram_gb": float(c.get("vram_gb", 0) or 0),
        "util_pct": float(c.get("util_pct", 0) or 0),
        "gpus": int(c.get("gpus", 1) or 1),
        "gpu_index": config.get("gpu_index"),
        "note": note,
        "at_epoch": now_epoch(),
        "expires_at": now_epoch() + ttl,
        "created_at": now_iso(),
    }
    name = "%s@%s.%s.json" % (claim["owner"], claim["host"], claim["pid"] or "pre")
    path = os.path.join(d, name)
    atomic_write_json(path, claim)
    claim["_path"] = path
    return claim


def release_claim(claim):
    if not claim:
        return
    path = claim.get("_path")
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def release_claims_for(task_id, config=None):
    """Drop any of our claims that name this task.  Called when it ends."""
    config = config or load_config()
    for c in load_claims(config):
        if c.get("task_id") == task_id and c.get("owner") == config.get("owner"):
            release_claim(c)


# ---- intent: a non-binding 'I want this soon' board -----------------------
# Read-only by default: publishing intent never changes anybody's scheduling
# on its own.  What it buys is information - if the card is about to be taken
# by someone who declared first, you can choose to wait instead of grabbing
# the last free gigabyte and forcing them to wait.

def intent_path(config=None):
    config = config or load_config()
    d = shared_claims_dir(config)
    return os.path.join(d, "intent-%s@%s.json" % (config.get("owner"), host_name()))


def load_intents(config=None):
    config = config or load_config()
    d = shared_claims_dir(config)
    now = now_epoch()
    out = []
    for path in _claim_glob(d):
        base = os.path.basename(path)
        if not base.startswith("intent-"):
            continue
        i = read_json(path)
        if not isinstance(i, dict):
            continue
        if i.get("expires_at") and i["expires_at"] < now:
            continue
        i["_path"] = path
        out.append(i)
    return out


def publish_intent(config=None, ttl=3600):
    """Publish what our queue is waiting for.  Never blocks, never fails hard."""
    config = config or load_config()
    d = shared_claims_dir(config)
    ensure_dir(d)
    q = load_queue()
    mem = load_memory()
    tasks = []
    for t in pending_tasks(q, mem, None):
        c = t.get("cost") or {}
        tasks.append({"task_id": t["id"], "vram_gb": float(c.get("vram_gb", 0) or 0),
                      "util_pct": float(c.get("util_pct", 0) or 0),
                      "since": t.get("created_at")})
    board = {"board": "intent", "owner": config.get("owner"), "host": host_name(),
             "updated_at": now_iso(), "at_epoch": now_epoch(),
             "expires_at": now_epoch() + ttl, "tasks": tasks}
    try:
        atomic_write_json(intent_path(config), board)
    except OSError:
        return None
    return board


def competes_with_us(cand, our_task, our_config, intents):
    """Should we stand down for somebody who declared first?

    Only two things make us yield: they registered their intent before us,
    and their task would plausibly take the room we were about to take.  A
    stale or tiny intent never holds the card hostage.
    """
    if not our_task:
        return None
    ours_v = float((our_task.get("cost") or {}).get("vram_gb", 0) or 0)
    ours_since = our_task.get("created_at") or ""
    for i in intents:
        if i.get("owner") == our_config.get("owner"):
            continue
        for t in i.get("tasks") or []:
            their_v = float(t.get("vram_gb") or 0)
            if their_v + ours_v <= cand["mem_free_gb"] - 0.01:
                continue          # we can both fit; no reason to yield
            if (t.get("since") or "") and ours_since and t["since"] < ours_since:
                return {"owner": i.get("owner"), "host": i.get("host"),
                        "task_id": t.get("task_id"), "vram_gb": their_v,
                        "since": t.get("since")}
    return None


# --------------------------------------------------------------------------
# queue
# --------------------------------------------------------------------------

def load_queue():
    q = read_json(QUEUE_PATH, None)
    if q is None:
        q = {"version": 1, "policy": dict(DEFAULT_POLICY), "tasks": []}
    q.setdefault("policy", {})
    for k, v in DEFAULT_POLICY.items():
        q["policy"].setdefault(k, v)
    q.setdefault("tasks", [])
    return q


def save_queue(q):
    q["updated_at"] = now_iso()
    atomic_write_json(QUEUE_PATH, q)


def footprint(t):
    """Lower == cheaper == 'lighter requirement'."""
    c = t.get("cost") or {}
    return (c.get("gpus", 1) or 1) * 1000.0 \
        + float(c.get("vram_gb", 0) or 0) * 10.0 \
        + float(c.get("util_pct", 0) or 0)


def _order_key(t, rank_bias):
    pinned = 0 if t.get("pinned") else 1
    if t.get("rank") is not None:
        order = float(t["rank"])
    else:
        order = 1000.0 + footprint(t) + rank_bias
    return (pinned, order, -float(t.get("priority", 0) or 0),
            t.get("created_at", ""))


def sort_tasks(tasks, mem=None, snap=None):
    """Order the queue.

    Pinned first, then explicit rank, then cheapest-first.  With a memory in
    hand, `profile` entries bias the cheap-first arm: a task the memory says
    pairs badly with what is currently on the card sinks below the others.
    Pinned and explicitly ranked tasks are never moved - the user asked.
    """
    bias = mem_rank_bias(mem, snap, list(tasks)) if mem else {}
    return sorted(tasks, key=lambda t: _order_key(t, bias.get(t.get("id"), 0.0)))


def pending_tasks(q, mem=None, snap=None):
    return sort_tasks([t for t in q["tasks"] if t.get("status") == "pending"],
                      mem, snap)


def running_tasks(q):
    return [t for t in q["tasks"] if t.get("status") == "running"]


def find_task(q, tid):
    for t in q["tasks"]:
        if t["id"] == tid:
            return t
    # allow unique prefix match for convenience
    hits = [t for t in q["tasks"] if t["id"].startswith(tid)]
    if len(hits) == 1:
        return hits[0]
    return None


def new_id(q, name):
    base = re.sub(r"[^a-z0-9]+", "-", (name or "task").lower()).strip("-")[:24] or "task"
    existing = set(t["id"] for t in q["tasks"])
    if base not in existing:
        return base
    i = 2
    while "%s-%d" % (base, i) in existing:
        i += 1
    return "%s-%d" % (base, i)


# --------------------------------------------------------------------------
# memory: what this machine has learned about its other users
# --------------------------------------------------------------------------
# This is a shared server as often as it is a laptop.  Two things are worth
# remembering across sessions, and neither is derivable from a single
# nvidia-smi sample:
#
#   protected  - a process (by pid or name) that must not be disturbed, or a
#                task that must not cause one to be disturbed
#   window     - a recurring block of time that must be left free for someone
#   profile    - an observation that a process family behaves a certain way
#                (e.g. holds a lot of VRAM but little compute), which makes it
#                a good partner for a task shaped the opposite way
#
# All three are written as files under memory/, one fact per file, in the
# plainest JSON that still round-trips.  `bider.py remember --export` dumps them;
# --import reads them back, so a memory folder can be copied between machines.

DEFAULT_MEMORY = {"version": 1, "entries": [], "advice": []}


def load_memory():
    """Our own memory, plus whatever other people have published.

    Other people's entries live in the shared directory and are read-only to
    us: we can see that Wei has a deadline, but only Wei edits Wei's file.
    """
    m = read_json(MEMORY_PATH, None)
    if m is None:
        m = {"version": 1, "entries": [], "advice": []}
    m.setdefault("entries", [])
    m.setdefault("advice", [])
    m["entries"] = list(m["entries"])

    mine = set(e.get("id") for e in m["entries"])
    for path in shared_memory_files():
        other = read_json(path)
        if not isinstance(other, dict):
            continue
        for e in other.get("entries", []):
            if not isinstance(e, dict) or e.get("kind") not in MEMORY_KINDS:
                continue
            if not e.get("id") or e["id"] in mine:
                continue
            e = dict(e)
            # `source` keeps saying how the entry was learned; `shared` marks
            # that we may read it but must never edit or republish it
            e["shared"] = True
            e["shared_from"] = os.path.basename(path)
            e["shared_owner"] = other.get("owner")
            m["entries"].append(e)
            mine.add(e["id"])
    return m


def save_memory(m):
    m["updated_at"] = now_iso()
    atomic_write_json(MEMORY_PATH, m)


def _slug(text, prefix):
    base = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:28]
    return "%s-%s" % (prefix, base) if base else prefix


def add_memory_entry(mem, kind, **kw):
    if kind not in MEMORY_KINDS:
        raise ValueError("unknown memory kind: %s (known: %s)"
                         % (kind, ", ".join(MEMORY_KINDS)))
    e = {"kind": kind, "created_at": now_iso()}
    e.update(kw)
    e.setdefault("source", "user")
    if not e.get("id"):
        e["id"] = _slug(e.get("person") or e.get("name") or kind, kind)
    existing = [x for x in mem["entries"] if x.get("id") == e["id"]]
    if existing:
        existing[0].update(e)
        e = existing[0]
    else:
        mem["entries"].append(e)
    return e


def find_memory(mem, key):
    hits = [e for e in mem["entries"]
            if e.get("id") == key or key in (e.get("person"), e.get("name"))
            or str(e.get("id", "")).startswith(key)]
    return hits[0] if len(hits) == 1 else None


def shared_dir(config=None):
    """Where claim files and published memory live.  Empty means local-only."""
    c = config or load_config()
    d = c.get("shared_dir") or ""
    return os.path.abspath(os.path.expanduser(d)) if d else ""


def shared_claims_dir(config=None):
    d = shared_dir(config)
    return os.path.join(d, "claims") if d else os.path.join(HOME, "claims")


def shared_publish_dir(config=None):
    d = shared_dir(config)
    return os.path.join(d, "memory") if d else ""


def shared_memory_files(config=None):
    d = shared_publish_dir(config)
    if not d or not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".json"):
            out.append(os.path.join(d, name))
    return out


def publish_memory(mem, config=None):
    """Make our own memory visible to everyone else on the shared card.

    Only entries we own are published, and not the ones the user marked
    private.  Sharing is the point of a shared card: a colleague cannot let
    you know about their deadline if you cannot see it.
    """
    d = shared_publish_dir(config)
    if not d:
        return None
    config = config or load_config()
    entries = [e for e in mem["entries"]
               if not e.get("shared") and not e.get("private")]
    ensure_dir(d)
    path = os.path.join(d, "%s.json" % (config.get("owner") or host_name()))
    if not entries:
        # nothing of ours to say: remove the file rather than leave stale
        # entries that other people would keep obeying
        try:
            os.unlink(path)
        except OSError:
            pass
        return None
    atomic_write_json(path, {"version": 1, "owner": config.get("owner"),
                             "host": host_name(), "updated_at": now_iso(),
                             "entries": entries})
    return path


# ---- time windows ---------------------------------------------------------

def parse_clock(s):
    """'22:00' -> minutes past midnight."""
    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", s or "")
    if not m:
        raise ValueError("expected HH:MM, got %r" % s)
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 24 or mi > 59:
        raise ValueError("out of range: %s" % s)
    return (h % 24) * 60 + mi


def parse_days(s):
    """'mon-fri' / 'sat,sun' / '' (every day) -> set of 0..6, Monday=0."""
    if not s or s.strip().lower() in ("*", "all", "every"):
        return set(range(7))
    out = set()
    for part in str(s).replace(";", ",").split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "-" in part:
            a, b = [x.strip()[:3] for x in part.split("-", 1)]
            if a not in WEEKDAYS or b not in WEEKDAYS:
                raise ValueError("unknown weekday in %r" % s)
            i, j = WEEKDAYS.index(a), WEEKDAYS.index(b)
            if i <= j:
                out.update(range(i, j + 1))
            else:                                   # wraps the week, e.g. fri-mon
                out.update(list(range(i, 7)) + list(range(0, j + 1)))
        else:
            part = part[:3]
            if part not in WEEKDAYS:
                raise ValueError("unknown weekday in %r" % s)
            out.add(WEEKDAYS.index(part))
    return out


def window_active(w, when=None):
    """Is this window in force at `when` (a datetime, default now)?

    Handles windows that cross midnight (22:00-02:00).
    """
    when = when or datetime.now()
    days = parse_days(w.get("days"))
    start = parse_clock(w.get("from"))
    end = parse_clock(w.get("to"))
    now_min = when.hour * 60 + when.minute
    today = when.weekday()
    if start <= end:
        return today in days and start <= now_min < end
    # wraps midnight: tonight belongs to `today`, tomorrow morning to the
    # previous day's schedule
    if now_min >= start:
        return today in days
    if now_min < end:
        return ((today - 1) % 7) in days
    return False


def window_description(w):
    days = parse_days(w.get("days"))
    if len(days) == 7:
        d = "every day"
    elif days == set([0, 1, 2, 3, 4]):
        d = "weekdays"
    elif days == set([5, 6]):
        d = "weekends"
    else:
        d = ",".join(WEEKDAYS[i] for i in sorted(days))
    return "%s %s-%s" % (d, w.get("from"), w.get("to"))


def _hhmm_to_epoch(now, hhmm, forward=True):
    """The next (or previous) occurrence of hh:mm relative to `now`."""
    h, m = divmod(parse_clock(hhmm), 60)
    cand = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if forward and cand <= now:
        cand += timedelta(days=1)
    if not forward and cand > now:
        cand -= timedelta(days=1)
    return cand


def check_windows(mem, when=None, lookahead_s=900):
    """Returns (blocking_entry, seconds_until_a_window_opens).

    `lookahead_s` is how far ahead we care about: a window opening inside it
    stops us starting a long run that would have to be interrupted.
    """
    when = when or datetime.now()
    blocking, soonest = None, None
    for e in mem["entries"]:
        if e.get("kind") != "window" or e.get("disabled"):
            continue
        if window_active(e, when):
            return e, 0
        try:
            start = _hhmm_to_epoch(when, e.get("from"))
        except ValueError:
            continue
        delta = (start - when).total_seconds()
        if 0 < delta <= lookahead_s and start.weekday() in parse_days(e.get("days")):
            if soonest is None or delta < soonest[0]:
                soonest = (delta, e)
    return None, (soonest[0] if soonest else None)


# ---- matching a process to what we remember ------------------------------

def _norm(s):
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def proc_matches(pat, proc):
    """Does a memory entry - or a bare pattern - describe this process?

    Entries carry the pattern under `match`, but a bare {"name": ...} is also
    accepted so tests and callers can pass either.
    """
    pat = pat.get("match") if isinstance(pat.get("match"), dict) else pat
    if not pat:
        return False
    if proc.get("pid") is not None and pat.get("pid") is not None:
        try:
            if int(pat["pid"]) == int(proc["pid"]):
                return True
        except (TypeError, ValueError):
            pass
    pname = proc.get("name") or ""
    if not pname:
        return False
    for field in ("name", "path", "family", "process"):
        want = pat.get(field)
        if not want:
            continue
        if _norm(want) and _norm(want) in _norm(pname):
            return True
        if want in pname:
            return True
    return False


def entry_expired(e, now=None):
    """A memory entry with an `until` in the past has lapsed."""
    until = e.get("until")
    if not until:
        return False
    now = now or datetime.now()
    try:
        return datetime.strptime(until, "%Y-%m-%d").date() < now.date()
    except ValueError:
        return False


def active_entries(mem, snap, now=None):
    """Which memory entries are in force right now.

    A `protected` entry is in force only while one of the processes it names
    is actually on the card - that is the whole meaning of the entry.  If the
    person is not running, we are free to use the machine; we just keep the
    entry so a future session still knows about them.
    """
    now = now or datetime.now()
    procs = snap.get("procs") or []
    out = []
    for e in mem["entries"]:
        if e.get("disabled") or entry_expired(e, now):
            continue
        k = e.get("kind")
        if k == "protected":
            hits = [p for p in procs if proc_matches(e, p)]
            if hits:
                out.append((e, hits))
        elif k == "window":
            if window_active(e, now):
                out.append((e, []))
    return out


def dormant_entries(mem, now=None):
    """Protections that are still remembered but not currently triggered."""
    now = now or datetime.now()
    return [e for e in mem["entries"]
            if e.get("kind") == "protected" and not e.get("disabled")
            and not entry_expired(e, now)]


def mem_rank_bias(mem, snap=None, tasks=None):
    """Per-task ordering bias.

    Two sources, both advisory: stored `advice` entries (written by hand or by
    a previous session) and advice derived live from the profiles that match
    whatever is on the card right now.  The live one wins when they disagree,
    because it is about *this* moment.
    """
    bias = {}
    for a in mem.get("advice", []):
        if a.get("task"):
            bias[a["task"]] = float(a.get("bias", 0.0))
    if snap is not None:
        for a in profile_advice(mem, load_observations(), snap, tasks)[0]:
            bias[a["task"]] = float(a["bias"])
    return bias


def profile_advice(mem, obs, snap, tasks=None):
    """Turn 'this family behaves like X' into ordering advice.

    If the card is currently held by a process whose family we remember as
    VRAM-heavy but compute-light, then the useful thing to run next is a task
    shaped the other way: compute-heavy, VRAM-light.  Advice is a bias, never
    a veto - `bider.py pin` still wins.
    """
    holders = []
    procs = snap.get("procs") or []
    for p in procs:
        for e in mem["entries"]:
            if e.get("kind") == "profile" and not e.get("disabled") \
                    and proc_matches(e, p):
                holders.append((p, e))
    advice, notes = [], []
    if not holders:
        return advice, notes
    if tasks is None:
        tasks = pending_tasks_raw()
    for p, e in holders:
        shape = e.get("shape") or {}
        held_vram = shape.get("vram") or "heavy"
        held_util = shape.get("util") or "light"
        want_util = "heavy" if held_util == "light" else "light"
        want_vram = "light" if held_vram == "heavy" else "heavy"
        notes.append("%s (%s) holds %s VRAM / %s compute, so prefer tasks that "
                     "are %s compute / %s VRAM"
                     % (p.get("name", "?").split("\\")[-1], e.get("family") or e.get("id"),
                        held_vram, held_util, want_util, want_vram))
        for t in tasks:
            c = t.get("cost") or {}
            vu = float(c.get("util_pct", 0) or 0)
            vv = float(c.get("vram_gb", 0) or 0)
            good = (want_util == "heavy" and vu >= 40) or \
                   (want_util == "light" and vu <= 25)
            bad = (want_vram == "light" and vv >= 3.0) or \
                  (want_vram == "heavy" and vv <= 0.5)
            bias = -150.0 if (good and not bad) else (150.0 if bad else 0.0)
            rationale = ("pairs well with the %s process currently on the card"
                         % (e.get("family") or "remembered"))
            if not good and not bad:
                continue
            advice.append({"task": t["id"], "bias": bias,
                           "why": rationale if bias < 0 else
                                  "same shape as the %s process holding the card"
                                  % (e.get("family") or "remembered")})
    # dedupe: keep the strongest bias per task
    best = {}
    for a in advice:
        cur = best.get(a["task"])
        if cur is None or abs(a["bias"]) > abs(cur["bias"]):
            best[a["task"]] = a
    return list(best.values()), notes


def pending_tasks_raw():
    q = load_queue()
    return sort_tasks([t for t in q["tasks"] if t.get("status") == "pending"])


# ---- observations: things worth remembering that we noticed ourselves ----

def load_observations():
    return read_json(OBS_PATH, {"version": 1, "observations": []}) \
        or {"version": 1, "observations": []}


def observe(snap=None, window=1200):
    """Correlate our run bookkeeping with process start/stop, so that
    'experiment X holds ~N GB' and 'family Y behaves like Z' become facts
    instead of guesses.  Cheap: reads files we already write."""
    snap = snap or snapshot()
    obs = load_observations()
    seen = obs["observations"]
    known = set(o.get("key") for o in seen)
    added = []

    # 1. how much VRAM our own finished runs actually held
    for m in _recent_run_metas(window):
        key = "oursize:%s" % m.get("run_id")
        if key in known or m.get("status") == "starting":
            continue
        g0 = (m.get("gpu_at_start") or {}).get("mem_used_gb")
        g1 = (m.get("gpu_at_end") or {}).get("mem_used_gb")
        if g0 is None or g1 is None:
            continue
        held = round(g0 - g1, 2)
        want = (find_cost_of(m.get("task_id")) or {}).get("vram_gb")
        if held <= 0:
            continue
        seen.append({"key": key, "kind": "oursize", "task_id": m.get("task_id"),
                     "held_gb": held, "declared_gb": want,
                     "at": m.get("ended_at"), "source": "observed",
                     "note": "measured from whole-GPU totals before/after our run"})
        added.append(seen[-1])

    # 2. the shape of someone else's process family: keeps a running median
    #    rather than one row per poll, so a family's profile sharpens with
    #    time instead of flooding the file
    procs = snap.get("procs") or []
    agg = gpu_aggregate(snap) or {}
    by_key = dict((o.get("key"), o) for o in seen)
    for p in procs:
        fam = _family(p.get("name"))
        if not fam:
            continue
        key = "shape:%s" % fam
        cur = by_key.get(key)
        util = agg.get("util_pct")
        used = agg.get("mem_used_gb")
        if util is None:
            continue
        if cur is None:
            cur = {"key": key, "kind": "shape", "family": fam,
                   "name": p.get("name"), "first_at": now_iso(),
                   "source": "observed", "samples": 1,
                   "util_sum": util, "used_sum": used or 0.0}
            seen.append(cur)
            by_key[key] = cur
            added.append(cur)
        else:
            cur["samples"] = cur.get("samples", 0) + 1
            cur["util_sum"] = cur.get("util_sum", 0.0) + util
            cur["used_sum"] = cur.get("used_sum", 0.0) + (used or 0.0)
        cur["at"] = now_iso()
        cur["gpu_util_pct"] = round(cur["util_sum"] / cur["samples"], 1)
        cur["gpu_used_gb"] = round(cur["used_sum"] / cur["samples"], 2)
        cur["n_procs"] = len(procs)
        cur["note"] = ("%d sample(s) so far; a family needs several before its "
                       "shape is trustworthy" % cur["samples"])

    obs["observations"] = seen[-500:]
    atomic_write_json(OBS_PATH, obs)
    return added, obs


def _family(name):
    """A stable-ish label for a family of processes, e.g. 'train' from
    'train_worker_3.exe'.  Stripping trailing digits matters: a server usually
    names its workers with an index, and those are all the same family."""
    base = os.path.basename((name or "").replace("\\", "/"))
    base = re.sub(r"\.(exe|py|bin|sh|out|dll)$", "", base, flags=re.I)
    base = re.split(r"[-_. ]", base)[0]
    base = re.sub(r"\d+$", "", base)
    return base.lower() if len(base) > 2 else ""


def _recent_run_metas(seconds):
    out = []
    if not os.path.isdir(RUNS_DIR):
        return out
    now = time.time()
    for d in sorted(os.listdir(RUNS_DIR), reverse=True)[:50]:
        p = os.path.join(RUNS_DIR, d, "meta.json")
        try:
            if now - os.path.getmtime(p) > seconds * 6:
                continue
        except OSError:
            continue
        m = read_json(p)
        if m:
            out.append(m)
    return out


def find_cost_of(task_id):
    for t in load_queue()["tasks"]:
        if t["id"] == task_id:
            return t.get("cost")
    return None


def shape_of_family(obs, family, min_samples=3):
    """Aggregate shape samples into VRAM-heavy/light and util-heavy/light.

    A sample only counts once its family has been seen `min_samples` times, so
    a family that just started is not profiled off one poll.
    """
    pts = [o for o in obs["observations"]
           if o.get("kind") == "shape" and o.get("family") == family
           and o.get("gpu_util_pct") is not None
           and o.get("samples", 1) is not None]
    total = sum(o.get("samples", 1) for o in pts)
    if not pts or total < min_samples:
        return None
    util = sorted(p["gpu_util_pct"] for p in pts)
    med = util[len(util) // 2]
    mem = [p.get("gpu_used_gb") or 0 for p in pts]
    med_mem = sorted(mem)[len(mem) // 2]
    return {
        "family": family, "samples": total,
        "util_median_pct": round(med, 1),
        "used_gb_median": round(med_mem, 2),
        "vram": "heavy" if med_mem > 2.0 else "light",
        "util": "light" if med < 30 else ("heavy" if med > 70 else "medium"),
    }


def memory_report(mem, obs, snap=None):
    """Everything the memory currently implies, for `bider.py memory`."""
    snap = snap or snapshot()
    now = datetime.now()
    blocked, soon = check_windows(mem, now)
    live = active_entries(mem, snap, now)
    rows = []
    for e, hits in live:
        if e.get("kind") == "protected":
            rows.append({"id": e.get("id"), "person": e.get("person"),
                         "kind": "protected", "in_force": True,
                         "because": e.get("because"),
                         "procs": [{"pid": h["pid"], "name": h["name"]} for h in hits],
                         "until": e.get("until"), "created_at": e.get("created_at")})
        else:
            rows.append({"id": e.get("id"), "person": e.get("person"),
                         "kind": "window", "in_force": True,
                         "because": e.get("because"),
                         "when": window_description(e)})
    advice, notes = profile_advice(mem, obs, snap)
    upcoming = []
    for e in mem["entries"]:
        if e.get("kind") != "window" or e.get("disabled"):
            continue
        if not blocked or blocked.get("id") != e.get("id"):
            try:
                start = _hhmm_to_epoch(now, e.get("from"))
                upcoming.append({"id": e.get("id"), "when": window_description(e),
                                 "starts_in_min": round((start - now).total_seconds() / 60, 1)})
            except ValueError:
                pass
    upcoming.sort(key=lambda r: r["starts_in_min"])
    return {
        "ts": now_iso(),
        "blocked_by_window": {"id": blocked.get("id"), "person": blocked.get("person"),
                              "when": window_description(blocked),
                              "because": blocked.get("because")} if blocked else None,
        "window_starts_in_s": soon,
        "in_force": rows,
        "advice_now": advice,
        "advice_notes": notes,
        "upcoming_windows": upcoming[:5],
        "entries": mem["entries"],
        "observed_shapes": _observed_shapes(obs),
    }


def _observed_shapes(obs, limit=6):
    """Distinct families with enough evidence to describe a shape."""
    fams = []
    for o in obs["observations"]:
        if o.get("kind") == "shape" and o.get("family") and o["family"] not in fams:
            fams.append(o["family"])
    out = []
    for fam in fams:
        shape = shape_of_family(obs, fam)
        if shape:
            out.append(shape)
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------

def evaluate(q, tid=None, snap=None, mem=None, obs=None, config=None,
             claims=None, intents=None, our_pid=None):
    """Can we start a task right now? Returns a JSON-friendly decision dict.

    `mem` is consulted first and can override anything: a protected process or
    an in-force time window is a hard block, not a heuristic.  Claims come
    next, because they encode decisions other bider instances have already made
    but which nvidia-smi cannot show yet.
    """
    snap = snap or snapshot()
    agg = gpu_aggregate(snap)
    mem = load_memory() if mem is None else mem
    obs = load_observations() if obs is None else obs
    config = config or load_config()
    pol = q["policy"]

    out = {
        "ts": now_iso(),
        "gpu": agg,
        "gpu_source": snap.get("source"),
        "gpu_error": snap.get("error"),
        "policy": pol,
        "task_id": None,
        "runnable": False,
        "reasons": [],
        "headroom": None,
        "memory": {"blocks": [], "advice": [], "notes": []},
        "claims": {"mine": None, "others": []},
        "intent": {"yield_to": None, "notes": []},
        "skipped": [],
    }

    # ---- memory gates, before anything else -------------------------------
    window, soon = check_windows(mem)
    if window:
        out["memory"]["blocks"].append({
            "id": window.get("id"), "kind": "window",
            "person": window.get("person"), "when": window_description(window),
            "because": window.get("because"),
            "reason": "reserved time: %s is holding %s"
                      % (window.get("person") or "someone", window_description(window))})
    if soon:
        out["memory"]["notes"].append(
            "a reserved window opens in %.0f min" % (soon / 60.0))
    for e, hits in active_entries(mem, snap):
        if e.get("kind") != "protected":
            continue
        out["memory"]["blocks"].append({
            "id": e.get("id"), "kind": "protected", "person": e.get("person"),
            "because": e.get("because"), "until": e.get("until"),
            "procs": [{"pid": h["pid"], "name": h["name"]} for h in hits],
            "reason": "%s is running on this machine and must not be disturbed"
                      % (e.get("person") or e.get("id"))})
    # ---- advisory memory, computed up front so a refusal still carries it --
    pending_all = [t for t in q["tasks"] if t.get("status") == "pending"]
    advice, notes = profile_advice(mem, obs, snap, pending_all)
    out["memory"]["advice"] = advice
    out["memory"]["notes"].extend(notes)

    if out["memory"]["blocks"]:
        out["reasons"].extend(b["reason"] for b in out["memory"]["blocks"])
        return out

    if agg is None:
        out["reasons"].append("no GPU data (%s)" % (snap.get("error") or "unknown"))
        return out

    # ---- candidate selection, with first-fit ------------------------------
    if tid:
        task = find_task(q, tid)
        if task is None:
            out["reasons"].append("task not found: %s" % tid)
            return out
        if task.get("status") != "pending":
            out["reasons"].append("task %s status=%s (not pending)"
                                  % (task["id"], task.get("status")))
            return out
        cands = [task]
    else:
        cands = pending_tasks(q, mem, snap)
        if not cands:
            out["reasons"].append("no pending tasks")
            return out
        if not pol.get("first_fit", True):
            cands = cands[:1]

    out["task_id"] = cands[0]["id"]
    out["task"] = cands[0]

    busy = running_tasks(q)
    if busy and not pol.get("allow_concurrent"):
        out["reasons"].append("already running: %s" % ", ".join(t["id"] for t in busy))
        return out

    # ---- what is already spoken for ---------------------------------------
    # A live claim means somebody decided this space was theirs seconds ago;
    # showing it in nvidia-smi takes a few seconds longer.  Counting it here
    # is what stops two bider users from OOMing each other.
    if claims is None:
        claims = load_claims(config, release_mine=our_pid)
    # filter here too, so a caller that hands us a raw list still gets the
    # same answer as one that read the directory
    claims = [c for c in claims if not claim_expired(c)]
    mine_owner = config.get("owner")
    others = [c for c in claims if c.get("owner") != mine_owner]
    mine = [c for c in claims if c.get("owner") == mine_owner
            and c.get("pid") != our_pid]
    held = claims_totals(others + mine)
    out["claims"]["others"] = [
        {"owner": c.get("owner"), "host": c.get("host"), "pid": c.get("pid"),
         "task_id": c.get("task_id"), "vram_gb": c.get("vram_gb"),
         "util_pct": c.get("util_pct"),
         "age_s": round(max(0.0, now_epoch() - (c.get("at_epoch") or now_epoch())), 1)}
        for c in others]
    if held["n"]:
        out["memory"]["notes"].append(
            "%d live claim(s) already reserve %.2fGB / %.0f%%"
            % (held["n"], held["vram_gb"], held["util_pct"]))

    reserve_vram = float(pol["reserve_vram_gb"])
    reserve_util = float(pol["reserve_util_pct"])
    # Claims reduce what is really free.  They are subtracted from the free
    # pool rather than from the reserve, because a claim is another job about
    # to appear on the card, not headroom we owe the desktop.
    free_gb = agg["mem_free_gb"] - held["vram_gb"]
    util = agg["util_pct"] + held["util_pct"]

    def fits(task):
        c = task.get("cost") or {}
        need_vram = float(c.get("vram_gb", 0) or 0)
        need_util = float(c.get("util_pct", 0) or 0)
        need_gpus = int(c.get("gpus", 1) or 1)
        if need_gpus > agg["n_gpus"]:
            return False, "needs %d gpu(s), only %d present" % (need_gpus, agg["n_gpus"]), {}
        vram_after = free_gb - need_vram
        util_after = (100.0 - util) - need_util
        head = {"free_gb": round(free_gb, 2), "need_gb": need_vram,
                "vram_after_gb": round(vram_after, 2), "vram_reserve_gb": reserve_vram,
        "gpu_util_pct": round(util, 1), "idle_util_pct": round(100.0 - util, 1),
        "need_util_pct": need_util, "util_after_pct": round(util_after, 1),
        "util_reserve_pct": reserve_util,
        "claimed_gb": held["vram_gb"], "claimed_util_pct": held["util_pct"]}
        if vram_after < reserve_vram:
            why = ("vram short: free %.2fGB - need %.2fGB = %.2fGB < reserve %.2fGB"
                   % (free_gb, need_vram, vram_after, reserve_vram))
            if held["vram_gb"]:
                why += (" (%.2fGB is already reserved by another bider instance)"
                        % held["vram_gb"])
            return False, why, head
        if util_after < reserve_util:
            return False, ("util short: idle %.1f%% - need %.1f%% = %.1f%% < reserve %.1f%%"
                           % (100.0 - util, need_util, util_after, reserve_util)), head
        return True, ("fits: %.2fGB vram and %.1f%% util left for the desktop"
                      % (vram_after, util_after)), head

    # Try candidates in order.  The light heads the queue; if the light one
    # does not fit because somebody's job is holding the card, a *different
    # shape* might, which is the whole point of first-fit.
    chosen, chosen_why, chosen_head = None, None, None
    for i, t in enumerate(cands):
        ok, why, head = fits(t)
        if ok:
            chosen, chosen_why, chosen_head = t, why, head
            out["task_id"], out["task"] = t["id"], t
            if i:
                out["reasons"].append(
                    "started with the %d lower-ranked task(s) skipped" % i)
            break
        out["skipped"].append({"task_id": t["id"], "reason": why})
        if i == 0:
            chosen_why, chosen_head = why, head

    out["headroom"] = chosen_head
    if chosen is None:
        out["reasons"].extend(s["reason"] for s in out["skipped"][:1])
        return out

    # ---- memory advice restated for the task finally chosen ----------------
    out["reasons"].append(chosen_why)
    out["runnable"] = True
    return out


def change_events(prev, cur):
    """Human/JSON description of what changed between two snapshots."""
    ev = []
    if not prev:
        return ev
    pg = gpu_aggregate(prev) or {}
    cg = gpu_aggregate(cur) or {}
    d_free = cg.get("mem_free_gb", 0) - pg.get("mem_free_gb", 0)
    if d_free > 0.05:
        ev.append({"kind": "vram_freed", "delta_gb": round(d_free, 2)})
    d_util = cg.get("util_pct", 0) - pg.get("util_pct", 0)
    if d_util < -5:
        ev.append({"kind": "util_dropped", "delta_pct": round(d_util, 1)})
    pp = set(p["pid"] for p in prev.get("procs") or [])
    cp = set(p["pid"] for p in cur.get("procs") or [])
    gone, new = pp - cp, cp - pp
    if gone:
        names = {p["pid"]: p["name"] for p in prev.get("procs") or []}
        ev.append({"kind": "proc_exited",
                   "procs": [{"pid": i, "name": names.get(i, "?")} for i in sorted(gone)]})
    if new:
        names = {p["pid"]: p["name"] for p in cur.get("procs") or []}
        ev.append({"kind": "proc_started",
                   "procs": [{"pid": i, "name": names.get(i, "?")} for i in sorted(new)]})
    return ev


# --------------------------------------------------------------------------
# the ledger: a durable, append-only record of what bider did and why
# --------------------------------------------------------------------------
# A shared tool has to be auditable.  Every decision that could surprise
# somebody - a start, a yield, a refusal - is appended here with its reason,
# so "why did my job wait two hours" has an answer that is not a guess.

LEDGER_MAX_BYTES = 4 * 1024 * 1024


def record(event, **fields):
    """Append one event to the ledger.  Best effort; never raises."""
    try:
        entry = {"ts": now_iso(), "at_epoch": now_epoch(), "event": event,
                 "host": host_name()}
        entry.update(fields)
        append_jsonl(LEDGER_PATH, entry)
        if os.path.getsize(LEDGER_PATH) > LEDGER_MAX_BYTES:
            _rotate_ledger()
        run_notify(load_config(), entry)
        return entry
    except Exception:
        return None


def _rotate_ledger():
    try:
        os.replace(LEDGER_PATH, LEDGER_PATH + ".1")
    except OSError:
        pass


def read_ledger(limit=200, events=None):
    """Newest last.  Reads the rotated file too, so history is not lost."""
    out = []
    for path in (LEDGER_PATH + ".1", LEDGER_PATH):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if events and e.get("event") not in events:
                        continue
                    out.append(e)
        except OSError:
            continue
    return out[-limit:]


# --------------------------------------------------------------------------
# rendering helpers
# --------------------------------------------------------------------------

def fmt_gpu(snap):
    agg = gpu_aggregate(snap)
    lines = []
    if snap.get("error"):
        lines.append("GPU error: %s" % snap["error"])
    for g in snap.get("gpus") or []:
        lines.append("gpu%d %s  %s/%s MiB free=%s  util=%s%%  %sC" % (
            g["index"], g["name"],
            _fmt(g["mem_used_mb"]), _fmt(g["mem_total_mb"]),
            _fmt(g["mem_free_mb"]), _fmt(g["util_pct"]), _fmt(g["temp_c"])))
    if agg:
        lines.append("total free %.2fGB / %.2fGB, util %.1f%% (idle %.1f%%)" % (
            agg["mem_free_gb"], agg["mem_total_gb"], agg["util_pct"], 100 - agg["util_pct"]))
    procs = snap.get("procs") or []
    if procs:
        lines.append("compute processes:")
        for p in procs:
            lines.append("  pid=%-7s %-40s mem=%s" % (
                p["pid"], (p["name"] or "")[-40:], _fmt(p["mem_mb"])))
    else:
        lines.append("compute processes: (none reported)")
    return "\n".join(lines)


def _fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return ("%.1f" % v) if isinstance(v, float) else str(v)


def task_line(t, idx=None):
    c = t.get("cost") or {}
    flags = []
    if t.get("pinned"):
        flags.append("PIN")
    if t.get("rank") is not None:
        flags.append("rank=%s" % t["rank"])
    pre = ("%2d. " % idx) if idx is not None else ""
    return "%s%-22s %-9s vram=%-4sGB gpu=%-2s util=%-3s%% prio=%-3s %s%s" % (
        pre, t["id"], "[%s]" % t.get("status", "?"),
        c.get("vram_gb", 0), c.get("gpus", 1), c.get("util_pct", 0),
        t.get("priority", 0), " ".join(flags),
        ("  # " + t["name"]) if t.get("name") else "")


def print_queue(q, as_json=False, mem="auto", snap=None):
    if as_json:
        print(json.dumps(q, indent=2, ensure_ascii=False))
        return
    if mem == "auto":
        mem = load_memory()
    pol = q["policy"]
    print("queue: %s" % QUEUE_PATH)
    print("policy: reserve %.1fGB vram / %.0f%% util, poll %ss, settle %s ticks, "
          "concurrent=%s, first_fit=%s"
          % (pol["reserve_vram_gb"], pol["reserve_util_pct"], pol["poll_seconds"],
             pol["settle_ticks"], pol["allow_concurrent"], pol.get("first_fit", True)))
    print("")
    groups = [
        ("RUNNING", [t for t in q["tasks"] if t.get("status") == "running"]),
        ("PENDING (in run order)", pending_tasks(q, mem, snap)),
        ("OTHER", [t for t in q["tasks"] if t.get("status") not in ("pending", "running")]),
    ]
    for title, items in groups:
        if not items:
            continue
        print("== %s ==" % title)
        for i, t in enumerate(items):
            print(task_line(t, i + 1 if title.startswith("PENDING") else None))
            if t.get("command"):
                print("      $ %s" % t["command"])
        print("")


# --------------------------------------------------------------------------
# running tasks
# --------------------------------------------------------------------------

def _env_for_task(t):
    """The environment a task runs in.

    `task.sandbox` maps to CUDA_VISIBLE_DEVICES, so a multi-card machine can
    give each task its own card without any other change.
    """
    env = dict(os.environ)
    for k, v in (t.get("env") or {}).items():
        env[str(k)] = str(v)
    if t.get("sandbox") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(t["sandbox"])
    return env


def _run_dir(t):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = os.path.join(RUNS_DIR, "%s-%s" % (t["id"], stamp))
    ensure_dir(d)
    return d


def _launch(task, snap, detach, timeout_min, dry_run=False, log_path=None):
    """Start a task.  Returns (code, meta).  Shared by `run` and the watcher."""
    run_dir = _run_dir(task)
    out_path = log_path or os.path.join(run_dir, "stdout.log")
    meta_path = os.path.join(run_dir, "meta.json")
    env = _env_for_task(task)
    cwd = task.get("workdir") or os.getcwd()
    cmdline = task["command"]
    config = load_config()

    meta = {
        "run_id": os.path.basename(run_dir),
        "task_id": task["id"],
        "command": cmdline,
        "workdir": cwd,
        "owner": config.get("owner"),
        "host": host_name(),
        "started_at": now_iso(),
        "gpu_at_start": gpu_aggregate(snap),
        "pid": None,
        "status": "starting",
    }

    if dry_run:
        meta["status"] = "dry-run"
        atomic_write_json(meta_path, meta)
        return 0, meta

    if not os.path.isdir(cwd):
        cwd = os.getcwd()

    with open(out_path, "ab") as fh:
        fh.write(("=== %s ===\n$ %s\n(cwd=%s)\n\n"
                  % (now_iso(), cmdline, cwd)).encode("utf-8"))
        fh.flush()
        proc = subprocess.Popen(
            cmdline, shell=True, cwd=cwd, env=env,
            stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=detached_flags() if detach else 0,
        )
    meta["pid"] = proc.pid
    meta["status"] = "running"
    meta["log"] = out_path
    atomic_write_json(meta_path, meta)

    # Claim before anyone else can decide this space is free.  Written after
    # the process exists so the claim carries a real pid, but the window
    # between the two is milliseconds and the next decider re-checks anyway.
    claim = None
    if load_queue()["policy"].get("require_claim", True):
        claim = take_claim(task, config, pid=proc.pid)
        meta["claim"] = os.path.basename(claim.get("_path") or "")

    with FileLock(LOCK_PATH):
        q2 = load_queue()
        t2 = find_task(q2, task["id"])
        if t2:
            t2["status"] = "running"
            t2["run_id"] = meta["run_id"]
            t2["pid"] = proc.pid
            t2["started_at"] = meta["started_at"]
            save_queue(q2)

    record("started", task_id=task["id"], run_id=meta["run_id"], pid=proc.pid,
           owner=config.get("owner"), host=host_name(),
           command=cmdline, claimed=(claim is not None))

    print("started %s -> pid=%s run=%s\nlog: %s" % (
        task["id"], proc.pid, meta["run_id"], out_path), flush=True)

    if detach:
        return 0, meta

    t0 = time.time()
    rc = None
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if timeout_min and (time.time() - t0) > timeout_min * 60:
                print("timeout after %s min; stopping our own pid %s"
                      % (timeout_min, proc.pid))
                kill_tree(proc.pid)
                rc = proc.wait()
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("interrupted; leaving pid %s running" % proc.pid)
        rc = None

    meta["ended_at"] = now_iso()
    meta["exit_code"] = rc
    meta["duration_s"] = round(time.time() - t0, 1)
    meta["status"] = "finished" if rc == 0 else ("aborted" if rc is not None else "detached")
    meta["gpu_at_end"] = gpu_aggregate(snapshot())
    atomic_write_json(meta_path, meta)

    if claim:
        release_claim(claim)
    else:
        release_claims_for(task["id"], config)

    with FileLock(LOCK_PATH):
        q3 = load_queue()
        t3 = find_task(q3, task["id"])
        if t3:
            t3["status"] = "done" if rc == 0 else ("failed" if rc is not None else "running")
            t3["exit_code"] = rc
            t3["ended_at"] = meta["ended_at"]
            save_queue(q3)

    record("finished", task_id=task["id"], run_id=meta["run_id"],
           exit_code=rc, duration_s=meta["duration_s"],
           owner=config.get("owner"), host=host_name())

    print("task %s ended rc=%s after %.1fs" % (task["id"], rc, time.time() - t0))
    try:
        observed, _ = observe()
        for o in observed:
            if o.get("task_id") == task["id"]:
                print("observed: %s held %.2fGB (declared %.2fGB) -- %s"
                      % (task["id"], o["held_gb"], o.get("declared_gb") or 0, o["note"]))
    except Exception:
        pass
    print_tail(out_path, 40)
    return (0 if rc == 0 else 1), meta


def cmd_run(args):
    q = load_queue()
    task = find_task(q, args.task)
    if task is None:
        print("error: no such task: %s" % args.task, file=sys.stderr)
        return 4
    if task.get("status") != "pending" and not args.force:
        print("error: task %s status=%s; use --force to run anyway"
              % (task["id"], task.get("status")), file=sys.stderr)
        return 4

    snap = snapshot()
    dec = evaluate(q, task["id"], snap)
    if not dec["runnable"] and not args.no_check:
        print("refusing to start %s:" % task["id"], file=sys.stderr)
        for r in dec["reasons"]:
            print("  - %s" % r, file=sys.stderr)
        print("\ncurrent GPU:\n%s" % fmt_gpu(snap), file=sys.stderr)
        record("refused", task_id=task["id"], reasons=dec["reasons"],
               owner=load_config().get("owner"))
        return 2

    rc, meta = _launch(task, snap, args.detach, args.timeout_min,
                       dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "run_dir": os.path.join(RUNS_DIR, meta["run_id"]),
                          "command": meta["command"], "cwd": meta["workdir"],
                          "env_extra": task.get("env") or {},
                          "decision": dec}, indent=2))
    return rc



def print_tail(path, n):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return
    if not lines:
        return
    print("--- last %d lines of %s ---" % (min(n, len(lines)), os.path.basename(path)))
    for ln in lines[-n:]:
        sys.stdout.write(ln if ln.endswith("\n") else ln + "\n")
    print("--- end ---")


# --------------------------------------------------------------------------
# watch
# --------------------------------------------------------------------------

def watcher_state(**kw):
    st = read_json(WATCHER_STATE, {}) or {}
    st.update(kw)
    st["updated_at"] = now_iso()
    atomic_write_json(WATCHER_STATE, st)
    return st


def cmd_watch(args):
    if args.stop:
        st = read_json(WATCHER_STATE, {}) or {}
        pids = [st.get("pid")] + list(st.get("child_pids") or [])
        killed = []
        for p in [p for p in pids if p] + _read_pidfile():
            try:
                kill_tree(p)
                killed.append(int(p))
            except Exception:
                pass
        try:
            os.unlink(WATCHER_PID)
        except OSError:
            pass
        watcher_state(running=False, stopped_at=now_iso())
        print("stopped watcher pids: %s" % (killed or "(none found)"))
        return 0

    if args.status:
        st = read_json(WATCHER_STATE, {})
        if not st:
            print("no watcher state recorded")
            return 0
        alive = pid_alive(st.get("pid"))
        st["pid_alive"] = alive
        st["pid_file"] = _read_pidfile()
        print(json.dumps(st, indent=2, ensure_ascii=False))
        return 0

    if args.intent is not None:
        return cmd_intent(args)

    if args.daemon:
        return _watch_daemon(args)
    return _watch_foreground(args)


def _read_pidfile():
    try:
        with open(WATCHER_PID, "r", encoding="utf-8") as fh:
            return [int(x) for x in fh.read().split() if x.strip().isdigit()]
    except Exception:
        return []


def _bider_command():
    """How to invoke ourselves again: installed entry point if there is one."""
    exe = shutil.which("bider")
    if exe:
        return [exe]
    return [sys.executable, os.path.abspath(__file__)]


def _watch_daemon(args):
    ensure_dir(STATE_DIR)
    cmd = _bider_command() + ["watch",
                           "--interval", str(args.interval),
                           "--timeout-min", str(args.timeout_min),
                           "--on-trigger", args.on_trigger,
                           "--foreground"]
    if args.settle_ticks is not None:
        cmd += ["--settle-ticks", str(args.settle_ticks)]
    if args.task:
        cmd += ["--task", args.task]
    if args.require_change:
        cmd += ["--require-change"]
    if args.stay_alive:
        cmd += ["--stay-alive"]
    if args.no_observe:
        cmd += ["--no-observe"]
    if args.no_intent:
        cmd += ["--no-intent"]
    if args.home_dir:
        cmd += ["--home", args.home_dir]
    logfh = open(WATCHER_LOG, "ab")
    p = spawn_detached(cmd, HOME if os.path.isdir(HOME) else os.getcwd(), logfh)
    with open(WATCHER_PID, "w", encoding="utf-8") as fh:
        fh.write(str(p.pid))
    watcher_state(running=True, pid=p.pid, launcher_pid=os.getpid(), mode="daemon",
                  on_trigger=args.on_trigger, started_at=now_iso(),
                  log=WATCHER_LOG)
    print("watcher daemon started: pid=%s\nlog: %s\nstop with: bider.py watch --stop"
          % (p.pid, WATCHER_LOG))
    return 0


def _adaptive_interval(base, dec, settle_needed, interval, config):
    """Poll faster exactly when it can matter.

    The value of a low-latency poll is concentrated in the seconds after
    somebody else's job ends, so poll slowly while the card is busy and
    quickly once it frees - or once we think it is about to.
    """
    pol = load_queue()["policy"]
    floor = float(pol.get("poll_floor_seconds", 3))
    ceiling = float(pol.get("poll_ceiling_seconds", 120))
    if dec["runnable"]:
        # it can start now, so there is nothing to gain by waiting long
        return max(floor, min(base, floor * 2))
    agg = dec["gpu"]
    if not agg:
        return base
    # somebody is on the card: how close are they to finishing?
    eta = dec.get("foreign_eta_s")
    if eta is not None:
        return max(floor, min(ceiling, eta / 4.0))
    if agg["util_pct"] < 5 and agg["mem_free_gb"] > 1.0:
        return max(floor, base / 2.0)     # quiet but not enough room yet
    return min(ceiling, base)


def _watch_foreground(args):
    q = load_queue()
    pol = q["policy"]
    config = load_config()
    base_interval = args.interval or pol["poll_seconds"]
    settle_needed = max(1, args.settle_ticks if args.settle_ticks is not None
                        else int(pol.get("settle_ticks", 2)))
    timeout_s = args.timeout_min * 60 if args.timeout_min else None
    t0 = time.time()
    # the ledger records every tick under the watcher's pid, which is what
    # `stats` groups by; a daemon has a different pid from the shell that
    # launched it, so this comes from the state file when there is one
    our_pid = (read_json(WATCHER_STATE, {}) or {}).get("launcher_pid") or os.getpid()

    prev = None
    baseline = None
    good = 0
    ticks = 0
    all_events = []
    changed = False
    interval = base_interval

    if not args.json:
        log("watching: interval=%ss timeout=%smin settle=%d on_trigger=%s task=%s "
            "shared=%s"
            % (base_interval, args.timeout_min, settle_needed, args.on_trigger,
               args.task or "(queue head)", shared_dir(config) or "(local only)"),
            path=WATCHER_LOG)
    watcher_state(running=True, pid=our_pid, mode="foreground",
                  on_trigger=args.on_trigger, started_at=now_iso(),
                  interval=base_interval, task=args.task)

    while True:
        ticks += 1
        snap = snapshot()
        q = load_queue()
        events = change_events(prev, snap)
        if events:
            all_events.extend(events)
            changed = True
        if prev is None and baseline is not None:
            baseline = snap

        claims = load_claims(config, release_mine=our_pid)
        intents = load_intents(config)
        dec = evaluate(q, args.task, snap, config=config, claims=claims,
                       intents=intents, our_pid=our_pid)
        agg = dec["gpu"]

        # If the card is busy, work out when it will be free: that turns the
        # ETA from a report into a scheduling input.
        if not args.no_eta and agg and not dec["runnable"]:
            dec["foreign_eta_s"] = _foreign_eta(snap, config)

        if agg:
            msg = ("tick %d: free %.2fGB util %.1f%% | head=%s runnable=%s | %s"
                   % (ticks, agg["mem_free_gb"], agg["util_pct"],
                      dec["task_id"], dec["runnable"],
                      "; ".join(dec["reasons"][:2])))
        else:
            msg = "tick %d: %s" % (ticks, "; ".join(dec["reasons"][:2]))
        if not args.json:
            log(msg, path=WATCHER_LOG)

        # opportunistically record observations; cheap, and it is how the
        # memory earns its keep without a separate cron
        if not args.no_observe and ticks % 5 == 1:
            try:
                observe(snap)
            except Exception:
                pass

        # keep the intent board current, so the next person can see we are
        # waiting and for how much
        if not args.no_intent and ticks % 15 == 1:
            try:
                publish_intent(config)
            except Exception:
                pass

        # a reserved window about to open is worth interrupting for: there is
        # no point starting a two-hour run fifteen minutes before someone
        # needs the card
        soon = dec["memory"].get("window_starts_in_s")
        if dec["runnable"] and soon is not None and soon < (settle_needed * interval + 60):
            dec["memory"]["blocks"].append({
                "kind": "window_imminent",
                "reason": "a reserved window opens in %.0f min; not starting now"
                          % (soon / 60.0)})
            dec["reasons"].append(dec["memory"]["blocks"][-1]["reason"])
            dec["runnable"] = False

        # somebody declared first and their task needs the room we want
        if dec["runnable"] and intents and not args.no_intent:
            y = competes_with_us(agg, dec.get("task"), config, intents)
            if y:
                dec["intent"]["yield_to"] = y
                dec["runnable"] = False
                dec["reasons"].append(
                    "%s registered for %.2fGB before us; standing down"
                    % (y.get("owner"), y.get("vram_gb") or 0))
                record("yielded", task_id=dec["task_id"], to_owner=y.get("owner"),
                       to_task=y.get("task_id"), vram_gb=y.get("vram_gb"),
                       owner=config.get("owner"))
                try:
                    publish_intent(config)
                except Exception:
                    pass

        nobody_pending = (dec["task_id"] is None and "no pending tasks" in dec["reasons"])
        if nobody_pending:
            # do not exit just because the queue is empty if the user asked us
            # to keep watching for the card to free up; only an explicit flag
            # makes emptiness terminal
            if not args.stay_alive:
                if args.treat_empty_as_timeout:
                    return _finish(args, 2, "TIMEOUT", None, dec, all_events, baseline)
                return _finish(args, 3, "NOTHING_PENDING", None, dec, all_events, baseline)

        if dec["runnable"]:
            if args.require_change and not changed:
                good = 0
            else:
                good += 1
                if good >= settle_needed:
                    return _finish(args, 0, "TRIGGERED", dec["task_id"], dec,
                                   all_events, baseline)

        prev = snap
        if timeout_s is not None and (time.time() - t0) > timeout_s:
            return _finish(args, 2, "TIMEOUT", dec["task_id"], dec, all_events, baseline)
        interval = _adaptive_interval(base_interval, dec, settle_needed, interval,
                                      config)
        if not args.no_sleep:
            time.sleep(interval)


def _foreign_eta(snap, config=None):
    """When will the other person's job be done?  None when we cannot say.

    Best effort and explicitly uncertain: we can read a log we have no right
    to read only if the OS lets us, and a pid we recognise.  A wrong guess
    here costs a poll interval, never a job.
    """
    agg = gpu_aggregate(snap)
    if not agg or agg["util_pct"] < 5:
        return None
    best = None
    for p in foreign_procs(snap, config):
        log = _guess_log_for(p)
        if not log:
            continue
        prog = _scan_progress(_tail_text(log))
        if not prog:
            continue
        # one sample is not enough for a rate; store and come back later
        key = "eta-%s" % p["pid"]
        hist = read_json(os.path.join(STATE_DIR, "%s.json" % key), {"samples": []})
        hist["samples"].append({"t": time.time(), "frac": prog[2],
                                "cur": prog[0], "tot": prog[1]})
        hist["samples"] = hist["samples"][-200:]
        atomic_write_json(os.path.join(STATE_DIR, "%s.json" % key), hist)
        eta, _, _ = _eta_from_samples(hist["samples"], 900)
        if eta is not None and (best is None or eta < best):
            best = eta
    return best


def _guess_log_for(proc):
    """A readable log for somebody else's process, if we can find one.

    Deliberately narrow: we only look at files the process has open and that
    we are permitted to read.  Permission denied is the normal answer, and
    the normal outcome is that we report no ETA.
    """
    pid = proc.get("pid")
    if not pid:
        return None
    cands = []
    try:
        import psutil
        p = psutil.Process(int(pid))
        for f in p.open_files():
            cands.append(f.path)
    except Exception:
        pass
    if not cands:
        d = os.path.join(HOME, "logs")
        if os.path.isdir(d):
            cands = [os.path.join(d, n) for n in os.listdir(d)
                     if n.endswith((".log", ".out")) and str(pid) in n]
    for c in cands:
        if c.endswith((".log", ".out", ".txt")) and os.access(c, os.R_OK):
            return c
    return None


def _finish(args, code, verdict, task_id, dec, events, baseline):
    watcher_state(running=False, pid=os.getpid(), stopped_at=now_iso(),
                  verdict=verdict, task_id=task_id)
    config = load_config()
    payload = {
        "verdict": verdict,
        "exit_code": code,
        "task_id": task_id,
        "decision": dec,
        "changes_since_start": events,
        "baseline_gpu": gpu_aggregate(baseline) if baseline else None,
        "next_step": None,
    }
    if code == 0:
        payload["next_step"] = (
            "Deploy task '%s'. The intended command is:\n"
            "    bider run %s\n"
            "The watcher already recorded a claim for it; `run` re-checks the "
            "card and refuses if it no longer fits. Then re-arm the watcher "
            "with a new background watch for the next task."
            % (task_id, task_id))
        record("triggered", task_id=task_id, verdict=verdict,
               owner=config.get("owner"), reasons=dec.get("reasons"))
        if args.on_trigger == "run":
            payload["auto_run"] = _auto_run(task_id, dec)
        elif args.on_trigger == "claude":
            payload["auto_run"] = _spawn_claude(task_id, dec)
    elif code == 2:
        record("timeout", task_id=task_id, owner=config.get("owner"),
               reasons=dec.get("reasons"))
    elif code == 3:
        record("queue_empty", owner=config.get("owner"))

    print("\n" + "=" * 68)
    print("BIDER_TRIGGER " + json.dumps(payload, indent=2, ensure_ascii=False))
    print("=" * 68 + "\n")
    if events:
        print("changes since watcher started:")
        for e in events:
            print("  - %s" % json.dumps(e, ensure_ascii=False))
    if dec.get("headroom"):
        h = dec["headroom"]
        print("headroom: free %.2fGB (claimed %.2fGB) - need %.2fGB = %.2fGB "
              "(reserve %.2fGB)"
              % (h["free_gb"], h.get("claimed_gb", 0), h["need_gb"],
                 h["vram_after_gb"], h["vram_reserve_gb"]))
    return code


def _auto_run(task_id, dec):
    """--on-trigger run: start it here instead of waking a model up."""
    q = load_queue()
    task = find_task(q, task_id)
    if task is None:
        return {"error": "task vanished"}
    rc, meta = _launch(task, snapshot(), detach=True, timeout_min=None)
    return {"exit_code": rc, "run_id": meta.get("run_id"), "pid": meta.get("pid")}


def _spawn_claude(task_id, dec):
    """Unattended path: hand the trigger to a fresh headless Claude Code."""
    exe = shutil.which("claude") or shutil.which("claude.cmd")
    if not exe:
        return {"error": "claude executable not found on PATH"}
    prompt = (
        "/gpu-bider The watcher triggered: task '%s' is now runnable on this machine. "
        "Follow the skill instructions: start it with `bider run %s`, "
        "report the result briefly, then re-arm the watcher for the next task."
        % (task_id, task_id))
    cmd = [exe, "-p", prompt]
    extra = os.environ.get("BIDER_CLAUDE_ARGS", "").strip()
    if extra:
        cmd += extra.split()
    try:
        p = subprocess.Popen(cmd, cwd=HOME, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        out, _ = p.communicate(timeout=int(os.environ.get("BIDER_CLAUDE_TIMEOUT", "900")))
        return {"claude_rc": p.returncode,
                "output": out.decode("utf-8", "replace")[-4000:]}
    except Exception as e:
        return {"error": "failed to run claude: %s" % e}


# --------------------------------------------------------------------------
# estimate: when will a running program finish?
# --------------------------------------------------------------------------

PROGRESS_PATTERNS = [
    # tqdm: 45%|####  12/26 [00:31<00:36, ...]
    (r"(?P<pct>\d{1,3})%\|", None, 100.0),
    # tqdm counters: 1234/5000
    (r"(?P<cur>\d+)\s*/\s*(?P<tot>\d+)", "cur/tot", None),
    # keras/torch: Epoch 3/10
    (r"[Ee]poch\s+(?P<cur>\d+)\s*/\s*(?P<tot>\d+)", "cur/tot", None),
    # generic "step 400/2000" / "iter"
    (r"(?:step|iter|batch|it)\D{0,3}(?P<cur>\d+)\s*[/of]\s*(?P<tot>\d+)", "cur/tot", None),
    # bare percentage
    (r"(?P<pct>\d{1,3}(?:\.\d+)?)\s*%", None, 100.0),
]


def _scan_progress(text, patterns=None, total=None):
    """Return (current, total) from the last progress-looking line, or None."""
    pats = patterns or PROGRESS_PATTERNS
    best = None
    for line in text.splitlines():
        for spec in pats:
            if isinstance(spec, str):
                spec = (spec, None, None)
            rx, kind, scale = spec
            m = re.search(rx, line)
            if not m:
                continue
            gd = m.groupdict()
            if "pct" in gd and gd.get("pct") is not None:
                cur = float(gd["pct"])
                tot = float(total or scale or 100.0)
            elif gd.get("cur") is not None and gd.get("tot") is not None:
                try:
                    cur, tot = float(gd["cur"]), float(gd["tot"])
                except ValueError:
                    continue
            else:
                continue
            if tot <= 0 or cur < 0 or cur > tot * 1.5:
                continue
            frac = min(1.0, cur / tot)
            # keep the most recent (last) plausible reading
            best = (cur, tot, frac)
    return best


def _tail_text(path, max_bytes=262144):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _eta_from_samples(samples, window_s):
    """Project a finish time from progress samples.

    Only the most recent `window_s` seconds count, so a job that speeds up or
    slows down is projected on how it is going *now*.  Samples before the most
    recent backward jump are dropped, because a counter going backwards means
    the job restarted a phase (or restarted entirely).
    """
    if not samples:
        return None, None, 0
    now = samples[-1]["t"]
    win = [s for s in samples
           if (now - s["t"]) <= window_s and s.get("frac") is not None]
    cut = 0
    for i in range(1, len(win)):
        if win[i]["frac"] < win[i - 1]["frac"] - 1e-9:
            cut = i
    win = win[cut:]
    if len(win) < 2:
        return None, None, len(win)
    a, b = win[0], win[-1]
    dt = b["t"] - a["t"]
    df = b["frac"] - a["frac"]
    if dt <= 1.0 or df <= 1e-6:
        return None, None, len(win)
    rate = df / dt
    if rate <= 0:
        return None, None, len(win)
    eta = (1.0 - b["frac"]) / rate
    basis = ("progress %s -> %s over %.0fs (%.2f%%/min, %d samples in the last %ds)"
             % (a["cur"], b["cur"], dt, rate * 60 * 100, len(win), int(window_s)))
    return eta, basis, len(win)


def _cpu_time(pid):
    """CPU seconds consumed by a pid tree, or None."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        p = psutil.Process(int(pid))
        tot = p.cpu_times().user + p.cpu_times().system
        for ch in p.children(recursive=True):
            try:
                ct = ch.cpu_times()
                tot += ct.user + ct.system
            except Exception:
                pass
        return tot
    except Exception:
        return None


def cmd_estimate(args):
    pid = args.pid
    logpath = args.log
    if args.run:
        meta = None
        for d in sorted(os.listdir(RUNS_DIR)) if os.path.isdir(RUNS_DIR) else []:
            if d == args.run or d.startswith(args.run):
                meta = read_json(os.path.join(RUNS_DIR, d, "meta.json"))
                if meta:
                    pid = meta.get("pid")
                    logpath = os.path.join(RUNS_DIR, d, "stdout.log")
                    break
        if not meta:
            print("error: run %s not found" % args.run, file=sys.stderr)
            return 4

    patterns = PROGRESS_PATTERNS
    if args.pattern:
        patterns = [(args.pattern, None, None)] + PROGRESS_PATTERNS

    key = "%s-%s" % (pid if pid else "nopid",
                     hashlib.sha1((os.path.abspath(logpath) if logpath else "-")
                                  .encode("utf-8", "replace")).hexdigest()[:10])
    hist_path = os.path.join(STATE_DIR, "estimate-%s.json" % key)
    hist = read_json(hist_path, {"samples": []}) or {"samples": []}

    window = args.window_seconds if args.window_seconds else max(60.0, args.interval * 6)

    deadline = time.time() + (args.for_seconds or 0)
    printed = False
    while True:
        alive = pid_alive(pid) if pid else True
        text = _tail_text(logpath) if logpath else ""
        prog = _scan_progress(text, patterns, args.total) if text else None
        sample = {"t": time.time(), "frac": prog[2] if prog else None,
                  "cur": prog[0] if prog else None, "tot": prog[1] if prog else None,
                  "alive": alive}
        if args.pid:
            cpu = _cpu_time(pid)
            if cpu is not None:
                sample["cpu_s"] = round(cpu, 1)
        hist["samples"].append(sample)
        # keep only samples that can still matter, plus a little slack
        cutoff = time.time() - max(window * 4, 900.0)
        hist["samples"] = [s for s in hist["samples"] if s["t"] >= cutoff][-2000:]

        eta_s = None
        basis = None
        n_win = 0
        eta_s, basis, n_win = _eta_from_samples(hist["samples"], window)
        if eta_s is None and args.eta_cpu_hours:
            cpu_pts = [s for s in hist["samples"] if s.get("cpu_s") is not None]
            if len(cpu_pts) >= 2:
                dt = cpu_pts[-1]["t"] - cpu_pts[0]["t"]
                dc = cpu_pts[-1]["cpu_s"] - cpu_pts[0]["cpu_s"]
                if dt > 1 and dc > 0:
                    basis = ("no progress projection available; cpu duty %.0f%% "
                             "(CPU time is not a finish-time proxy)" % (dc / dt * 100))

        out = {
            "pid": pid,
            "log": logpath,
            "alive": alive,
            "progress": {"current": prog[0], "total": prog[1],
                         "percent": round(prog[2] * 100, 1)} if prog else None,
            "elapsed_s": round(hist["samples"][-1]["t"] - hist["samples"][0]["t"], 1)
            if hist["samples"] else 0,
            "eta_s": round(eta_s, 1) if eta_s is not None else None,
            "eta_human": _human(eta_s) if eta_s is not None else None,
            "finish_at": (datetime.fromtimestamp(time.time() + eta_s).astimezone()
                          .isoformat(timespec="seconds")) if eta_s is not None else None,
            "basis": basis,
            "samples": len(hist["samples"]),
            "window_samples": n_win,
            "note": None,
        }
        if not prog:
            out["note"] = ("no progress marker found in %s; pass --log and/or "
                           "--pattern to point at a progress line" % (logpath or "(no log)"))
        elif eta_s is None:
            out["note"] = "need >=2 samples with a changing progress value"

        atomic_write_json(hist_path, hist)

        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            _print_estimate(out)
        printed = True

        if args.for_seconds is None:
            break
        if time.time() >= deadline:
            break
        if pid and not alive:
            break
        time.sleep(args.interval)
    return 0 if printed else 4


def _human(s):
    if s is None:
        return None
    s = int(max(0, s))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    if h:
        return "%dh%02dm" % (h, m)
    if m:
        return "%dm%02ds" % (m, sec)
    return "%ds" % sec


def _print_estimate(o):
    print("estimate for pid=%s  (alive=%s, %s samples, %ss elapsed)" % (
        o["pid"], o["alive"], o["samples"], o["elapsed_s"]))
    if o["progress"]:
        p = o["progress"]
        print("  progress: %s/%s  (%.1f%%)" % (p["current"], p["total"], p["percent"]))
    if o["eta_s"] is not None:
        print("  ETA: %s  (finish ~%s)" % (o["eta_human"], o["finish_at"]))
        print("  basis: %s" % o["basis"])
    else:
        print("  ETA: unknown")
    if o["note"]:
        print("  note: %s" % o["note"])


def cmd_watchproc(args):
    """Persistent ETA sampling for a process; exits when it dies."""
    args.json = True if not hasattr(args, "json") else args.json
    return cmd_estimate(args)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_gpu(args):
    snap = snapshot()
    if args.json:
        print(json.dumps(snap, indent=2, ensure_ascii=False))
    else:
        print(fmt_gpu(snap))
    return 0 if snap.get("gpus") else 4


def cmd_procs(args):
    snap = snapshot()
    procs = snap.get("procs") or []
    mem = load_memory()
    if args.json:
        print(json.dumps(procs, indent=2, ensure_ascii=False))
    elif not procs:
        print("no compute processes reported")
    else:
        for p in procs:
            note = ""
            for e in mem["entries"]:
                if not e.get("disabled") and proc_matches(e, p):
                    if e.get("kind") == "protected":
                        note = "   <-- do not disturb (%s)" % (
                            e.get("person") or e.get("because") or "remembered")
                    elif e.get("kind") == "profile":
                        note = "   <-- known: %s vram / %s compute" % (
                            (e.get("shape") or {}).get("vram"),
                            (e.get("shape") or {}).get("util"))
            print("pid=%-7s %-45s mem=%s%s" % (p["pid"], p["name"],
                                               _fmt(p["mem_mb"]), note))
    return 0


def cmd_list(args):
    q = load_queue()
    mem = load_memory()
    # ordering depends on which remembered profiles match what is on the card
    # right now, so `list` looks once
    snap = snapshot() if mem["entries"] else None
    if args.pending_only:
        q = dict(q, tasks=pending_tasks(q, mem, snap))
    print_queue(q, args.json, mem, snap)
    return 0


def cmd_add(args):
    with FileLock(LOCK_PATH):
        q = load_queue()
        tid = args.id or new_id(q, args.name or args.command)
        if find_task(q, tid):
            print("error: id already exists: %s" % tid, file=sys.stderr)
            return 4
        task = {
            "id": tid,
            "name": args.name or tid,
            "command": args.command,
            "workdir": args.workdir or os.getcwd(),
            "cost": {
                "vram_gb": args.vram,
                "gpus": args.gpus,
                "util_pct": args.util,
                "cpus": args.cpus,
            },
            "priority": args.priority,
            "pinned": bool(args.pin),
            "rank": args.rank,
            "est_minutes": args.est_min,
            "env": dict(kv.split("=", 1) for kv in (args.env or [])),
            "sandbox": args.sandbox,
            "status": "pending",
            "tags": args.tag or [],
            "note": args.note or "",
            "created_at": now_iso(),
        }
        q["tasks"].append(task)
        save_queue(q)
    print("added %s" % tid)
    print_queue(load_queue())
    return 0


def _apply_set(task, expr):
    if "=" not in expr:
        raise ValueError("expected field=value, got %r" % expr)
    k, v = expr.split("=", 1)
    k = k.strip()
    if k in ("vram_gb", "util_pct", "gpus", "cpus"):
        task.setdefault("cost", {})[k] = float(v)
    elif k in ("priority", "rank", "est_minutes"):
        task[k] = None if v.strip().lower() in ("none", "null", "") else float(v)
    elif k in ("pinned", "status", "workdir", "command", "name", "note",
               "sandbox"):
        if k == "pinned":
            task[k] = v.strip().lower() in ("1", "true", "yes", "on")
        else:
            task[k] = None if v.strip() in ("", "none", "null") else v
    elif k.startswith("env."):
        task.setdefault("env", {})[k[4:]] = v
    else:
        raise ValueError("unknown field: %s" % k)


def cmd_update(args):
    with FileLock(LOCK_PATH):
        q = load_queue()
        task = find_task(q, args.task)
        if task is None:
            print("error: no such task: %s" % args.task, file=sys.stderr)
            return 4
        try:
            for s in args.set or []:
                _apply_set(task, s)
        except ValueError as e:
            print("error: %s" % e, file=sys.stderr)
            return 4
        if args.pin:
            task["pinned"] = True
        if args.unpin:
            task["pinned"] = False
        if args.priority is not None:
            task["priority"] = args.priority
        if args.command:
            task["command"] = args.command
        if args.status:
            task["status"] = args.status
        task["updated_at"] = now_iso()
        save_queue(q)
    print_queue(load_queue())
    return 0


def cmd_remove(args):
    with FileLock(LOCK_PATH):
        q = load_queue()
        before = len(q["tasks"])
        q["tasks"] = [t for t in q["tasks"] if not any(
            t["id"] == tid or t["id"].startswith(tid) for tid in args.tasks)]
        removed = before - len(q["tasks"])
        if args.done:
            q["tasks"] = [t for t in q["tasks"] if t.get("status") != args.done]
        save_queue(q)
    print("removed %d task(s)" % removed)
    print_queue(load_queue())
    return 0


def cmd_move(args):
    with FileLock(LOCK_PATH):
        q = load_queue()
        task = find_task(q, args.task)
        if task is None:
            print("error: no such task: %s" % args.task, file=sys.stderr)
            return 4
        if args.top:
            task["rank"] = -1.0
        elif args.rank is not None:
            task["rank"] = float(args.rank)
        # renormalise ranks in the current effective order
        ordered = pending_tasks(q)
        for i, t in enumerate(ordered):
            if t.get("rank") is not None or args.renumber:
                t["rank"] = float(i)
        save_queue(q)
    print_queue(load_queue())
    return 0


def cmd_pin(args):
    with FileLock(LOCK_PATH):
        q = load_queue()
        task = find_task(q, args.task)
        if task is None:
            print("error: no such task: %s" % args.task, file=sys.stderr)
            return 4
        task["pinned"] = not args.unpin
        save_queue(q)
    print_queue(load_queue())
    return 0


def cmd_clear(args):
    with FileLock(LOCK_PATH):
        q = load_queue()
        q["tasks"] = [t for t in q["tasks"] if t.get("status") not in
                      set(args.status or ["done", "failed"])]
        save_queue(q)
    print_queue(load_queue())
    return 0


def cmd_policy(args):
    with FileLock(LOCK_PATH):
        q = load_queue()
        for s in args.set or []:
            if "=" not in s:
                print("error: expected key=value", file=sys.stderr)
                return 4
            k, v = s.split("=", 1)
            k = k.strip()
            if k not in DEFAULT_POLICY:
                print("error: unknown policy key %s (known: %s)"
                      % (k, ", ".join(sorted(DEFAULT_POLICY))), file=sys.stderr)
                return 4
            dv = DEFAULT_POLICY[k]
            if isinstance(dv, bool):
                q["policy"][k] = v.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(dv, int):
                q["policy"][k] = int(float(v))
            else:
                q["policy"][k] = float(v)
        save_queue(q)
    print_queue(load_queue())
    return 0


def cmd_probe(args):
    q = load_queue()
    dec = evaluate(q, args.task)
    if args.json:
        print(json.dumps(dec, indent=2, ensure_ascii=False))
    else:
        print(fmt_gpu(snapshot()))
        print("")
        print("task: %s" % (dec["task_id"] or "(none)"))
        print("runnable: %s" % dec["runnable"])
        for r in dec["reasons"]:
            print("  - %s" % r)
        for s in dec.get("skipped") or []:
            print("  skipped %s: %s" % (s["task_id"], s["reason"]))
        for n in dec["memory"].get("notes") or []:
            print("  memory: %s" % n)
        if dec.get("headroom"):
            h = dec["headroom"]
            print("  math: vram %.2f-%.2f=%.2f (reserve %.2f) | util %.1f-%.1f=%.1f (reserve %.1f)"
                  % (h["free_gb"], h["need_gb"], h["vram_after_gb"], h["vram_reserve_gb"],
                     h["idle_util_pct"], h["need_util_pct"], h["util_after_pct"],
                     h["util_reserve_pct"]))
    return 0 if dec["runnable"] else 2


def cmd_status(args):
    q = load_queue()
    snap = snapshot()
    mem = load_memory()
    dec = evaluate(q, args.task, snap, mem)
    st = read_json(WATCHER_STATE, {}) or {}
    if st:
        st["pid_alive"] = pid_alive(st.get("pid"))
    out = {
        "ts": now_iso(),
        "gpu": gpu_aggregate(snap),
        "gpu_detail": snap.get("gpus"),
        "procs": snap.get("procs"),
        "watcher": st,
        "pending": [t["id"] for t in pending_tasks(q, mem)],
        "running": [{"id": t["id"], "pid": t.get("pid"), "run_id": t.get("run_id")}
                    for t in running_tasks(q)],
        "next": dec,
        "memory": memory_report(mem, load_observations(), snap),
    }
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(fmt_gpu(snap))
        print("")
        print_queue(q, mem=mem)
        if st:
            print("watcher: pid=%s alive=%s verdict=%s task=%s"
                  % (st.get("pid"), st.get("pid_alive"), st.get("verdict"),
                     st.get("task_id")))
        if dec["task_id"]:
            print("next up: %s -> %s" % (dec["task_id"],
                                         "RUNNABLE" if dec["runnable"] else "waiting"))
            for r in dec["reasons"]:
                print("  - %s" % r)
        print_memory_summary(mem, snap)
    return 0


def print_memory_summary(mem, snap):
    now = datetime.now()
    blocked, soon = check_windows(mem, now)
    live = active_entries(mem, snap, now)
    if not mem["entries"]:
        print("memory: empty (nothing remembered about this machine)")
        return
    mine = [e for e in mem["entries"] if not e.get("shared")]
    theirs = [e for e in mem["entries"] if e.get("shared")]
    print("memory: %d entr%s%s"
          % (len(mem["entries"]),
             "y" if len(mem["entries"]) == 1 else "ies",
             (" (%d from other users)" % len(theirs)) if theirs else ""))
    for e, hits in live:
        if e.get("kind") == "protected":
            print("  ! do not disturb: %s%s (%s) -- %s"
                  % (e.get("person") or e.get("id"),
                     " [shared]" if e.get("shared") else "",
                     e.get("because") or "",
                     ", ".join("pid %s %s" % (h["pid"], os.path.basename(h["name"] or ""))
                               for h in hits) or "not currently running"))
        else:
            print("  ! reserved time in force: %s (%s)%s"
                  % (window_description(e), e.get("person") or "?",
                     " [shared]" if e.get("shared") else ""))
    if soon:
        print("  ~ a reserved window opens in %.0f min" % (soon / 60.0))
    for e in mem["entries"]:
        if e.get("kind") == "profile" and not e.get("disabled"):
            print("  . profile: %s -> %s compute / %s vram"
                  % (e.get("family") or e.get("id"),
                     (e.get("shape") or {}).get("util", "?"),
                     (e.get("shape") or {}).get("vram", "?")))
    n_off = sum(1 for e in mem["entries"] if e.get("disabled"))
    if n_off:
        print("  (%d disabled entr%s ignored)"
              % (n_off, "y" if n_off == 1 else "ies"))
    if not live and not soon:
        print("  (nothing currently in force)")
    print("  %d own entr%s, %d published by others"
          % (len(mine), "y" if len(mine) == 1 else "ies", len(theirs)))


def cmd_intent(args):
    """The intent board: what everybody is waiting for, read-only by default."""
    config = load_config()
    which = getattr(args, "board", None) or getattr(args, "intent", None) or "show"
    if which == "publish":
        b = publish_intent(config)
        if b is None:
            print("could not publish (no shared_dir configured)")
            return 4
        print("published %d waiting task(s) as %s@%s"
              % (len(b["tasks"]), b["owner"], b["host"]))
        return 0
    if which == "clear":
        try:
            os.unlink(intent_path(config))
            print("cleared our intent")
        except OSError:
            print("nothing to clear")
        return 0

    intents = load_intents(config)
    claims = load_claims(config)
    if args.json:
        print(json.dumps({"intent": [dict(i, _path=None) for i in intents],
                          "claims": [dict(c, _path=None) for c in claims]},
                         indent=2, ensure_ascii=False))
        return 0
    if not shared_dir(config):
        print("shared_dir is not set, so nobody can see anybody's intent.")
        print("set one with:  bider config --set shared_dir=/path/everyone/can/see")
        print("(every bider pointed at the same directory works together)")
        print("")
    if not intents:
        print("intent board: empty")
    else:
        print("intent board (who is waiting for what):")
        for i in intents:
            tasks = i.get("tasks") or []
            if not tasks:
                print("  %-14s (nothing waiting)" % (i.get("owner") or "?"))
                continue
            for t in tasks[:4]:
                print("  %-14s %-20s needs %.2fGB  waiting since %s"
                      % (i.get("owner") or "?", t.get("task_id"),
                         float(t.get("vram_gb") or 0), t.get("since") or "?"))
    live = [c for c in claims if c.get("owner") != config.get("owner")]
    if live:
        print("")
        print("claims by other people right now:")
        for c in live:
            print("  %-14s %-20s %.2fGB / %.0f%% for %.0fs"
                  % (c.get("owner"), c.get("task_id"), c.get("vram_gb") or 0,
                     c.get("util_pct") or 0,
                     now_epoch() - (c.get("at_epoch") or now_epoch())))
    return 0


def cmd_memory(args):
    mem = load_memory()
    snap = snapshot()

    if args.import_file:
        other = read_json(args.import_file)
        if not isinstance(other, dict):
            print("error: %s is not a memory file" % args.import_file, file=sys.stderr)
            return 4
        for e in other.get("entries", []):
            add_memory_entry(mem, e.get("kind"), **{k: v for k, v in e.items()
                                                    if k != "kind"})
        mem["advice"].extend(other.get("advice", []))
        save_memory(mem)
        print("imported %d entries" % len(other.get("entries", [])))
        return 0

    if args.export:
        export = dict(mem)
        if args.export != "-":
            atomic_write_json(args.export, export)
            print("wrote %s" % args.export)
        else:
            print(json.dumps(export, indent=2, ensure_ascii=False))
        return 0

    if args.forget:
        before = len(mem["entries"])
        mem["entries"] = [e for e in mem["entries"]
                          if e.get("shared")
                          or (e.get("id") != args.forget
                              and args.forget not in (e.get("person"),
                                                      e.get("name")))]
        n = before - len(mem["entries"])
        save_memory(mem)
        if shared_dir():
            publish_memory(mem)
        print("forgot %d entr%s" % (n, "y" if n == 1 else "ies"))
        if not n:
            return 4
        return 0

    # ---- writes ----------------------------------------------------------
    if args.protect:
        until = args.until
        if until:
            try:
                datetime.strptime(until, "%Y-%m-%d")
            except ValueError:
                print("error: --until wants YYYY-MM-DD", file=sys.stderr)
                return 4
        e = add_memory_entry(
            mem, "protected", person=args.protect, name=args.protect,
            match={"name": args.match or args.protect} if (args.match or args.protect) else {},
            pid=args.pid, task=args.task, private=bool(args.private),
            because=args.because or "user said so", until=until)
        save_memory(mem)
        where = publish_memory(mem)
        print("remembered: do not disturb %s%s"
              % (e.get("person"),
                 " (published to the shared directory)" if where else ""))
        if until:
            print("  (until %s; it will lapse on its own)" % until)

    if args.window:
        if not (args.from_ and args.to):
            print("error: --window needs --from and --to (HH:MM)", file=sys.stderr)
            return 4
        try:
            parse_clock(args.from_)
            parse_clock(args.to)
            parse_days(args.days)
        except ValueError as ex:
            print("error: %s" % ex, file=sys.stderr)
            return 4
        e = add_memory_entry(mem, "window", person=args.window, name=args.window,
                             private=bool(args.private),
                             **{"from": args.from_, "to": args.to,
                                "days": args.days or "mon-sun",
                                "because": args.because or "user said so"})
        save_memory(mem)
        where = publish_memory(mem)
        print("remembered: %s reserved %s-%s%s"
              % (args.window, args.from_, args.to,
                 " (published)" if where else ""))

    if args.profile:
        fam = args.profile
        e = add_memory_entry(
            mem, "profile", family=fam, name=fam, private=bool(args.private),
            shape={"vram": args.profile_vram or "heavy",
                   "util": args.profile_util or "light"},
            because=args.because or "user said so")
        save_memory(mem)
        where = publish_memory(mem)
        print("remembered: %s behaves like %s vram / %s compute%s"
              % (fam, e["shape"]["vram"], e["shape"]["util"],
                 " (published)" if where else ""))

    if args.suggest:
        obs = load_observations()
        added, obs = observe(snap)
        promoted = 0
        for fam in sorted(set(o.get("family") for o in obs["observations"]
                              if o.get("kind") == "shape" and o.get("family"))):
            if any(e.get("kind") == "profile" and e.get("family") == fam
                   for e in mem["entries"]):
                continue
            shape = shape_of_family(obs, fam, min_samples=args.min_samples)
            if not shape:
                continue
            e = add_memory_entry(
                mem, "profile", family=fam, name=fam, source="observed",
                shape={"vram": shape["vram"], "util": shape["util"]},
                because="%d samples: median %.0f%% util, %.1fGB held"
                        % (shape["samples"], shape["util_median_pct"],
                           shape["used_gb_median"]))
            print("learned: %s -> %s vram / %s compute (%s)"
                  % (fam, shape["vram"], shape["util"], e["because"]))
            promoted += 1
        if promoted:
            save_memory(mem)
            publish_memory(mem)
        else:
            print("no family has enough samples yet (need %d)" % args.min_samples)

    if args.clear:
        mem["entries"] = [e for e in mem["entries"]
                          if e.get("kind") != args.clear or e.get("shared")]
        save_memory(mem)
        if shared_dir():
            publish_memory(mem)
        print("cleared all own %s entries" % args.clear)

    # ---- reads -----------------------------------------------------------
    report = memory_report(mem, load_observations(), snap)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    print("memory=%s  (%d entries)" % (MEMORY_PATH, len(mem["entries"])))
    if not mem["entries"]:
        print("")
        print("nothing remembered yet. examples:")
        print('  bider.py memory --protect wei --because "deadline this week" --until 2026-10-22')
        print('  bider.py memory --window wei --from 14:00 --to 18:00 --days mon-fri')
        print('  bider.py memory --profile train --vram heavy --util light')
        print('  bider.py memory --suggest         # learn shapes from what it has seen')
        return 0
    for e in mem["entries"]:
        kind = e.get("kind")
        head = "%-9s %-18s" % (kind, e.get("id", ""))
        if kind == "window":
            body = "%s reserved %s" % (e.get("person"), window_description(e))
        elif kind == "profile":
            body = "%s behaves like %s vram / %s compute" % (
                e.get("family"), (e.get("shape") or {}).get("vram"),
                (e.get("shape") or {}).get("util"))
        else:
            body = "do not disturb %s" % (e.get("person") or e.get("id"))
        print("%s %s" % (head, body))
        why = e.get("because") or ""
        extra = []
        if e.get("until"):
            extra.append("until %s" % e["until"])
        if e.get("source") == "observed":
            extra.append("observed")
        if e.get("task"):
            extra.append("only when running %s" % e["task"])
        if why:
            extra.append(why)
        print("          %s" % "; ".join(extra))
    print("")
    print_memory_summary(mem, snap)
    if report["advice_now"]:
        print("")
        print("ordering advice in force right now:")
        for a in report["advice_now"]:
            print("  %s%+g  %s  (%s)"
                  % ("bias ", a["bias"], a["task"], a["why"]))
    if report["upcoming_windows"]:
        print("")
        print("upcoming reserved time:")
        for u in report["upcoming_windows"][:3]:
            print("  in %.0f min: %s" % (u["starts_in_min"], u["when"]))
    return 0


def cmd_observe(args):
    snap = snapshot()
    added, obs = observe(snap)
    print("gpu: %s" % json.dumps(gpu_aggregate(snap) or {}, ensure_ascii=False))
    procs = snap.get("procs") or []
    print("compute processes: %d" % len(procs))
    for p in procs:
        print("  pid=%-7s %s" % (p["pid"], p["name"]))
    if added:
        print("recorded %d new observation(s):" % len(added))
        for o in added:
            print("  %s" % json.dumps(o, ensure_ascii=False))
    else:
        print("nothing new")

    if args.suggest:
        mem = load_memory()
        promoted = 0
        for fam in sorted(set(o.get("family") for o in obs["observations"]
                              if o.get("kind") == "shape" and o.get("family"))):
            if any(e.get("kind") == "profile" and e.get("family") == fam
                   for e in mem["entries"]):
                continue
            shape = shape_of_family(obs, fam, min_samples=args.min_samples)
            if not shape:
                continue
            e = add_memory_entry(
                mem, "profile", family=fam, name=fam, source="observed",
                shape={"vram": shape["vram"], "util": shape["util"]},
                because="%d samples: median %.0f%% util, %.1fGB held"
                        % (shape["samples"], shape["util_median_pct"],
                           shape["used_gb_median"]))
            print("learned: %s -> %s vram / %s compute (%s)"
                  % (fam, shape["vram"], shape["util"], e["because"]))
            promoted += 1
        if promoted:
            save_memory(mem)
        else:
            print("no family has enough samples yet (need %d)" % args.min_samples)
    return 0


def cmd_logs(args):
    """A place to drop task logs so `estimate` can read them later."""
    ensure_dir(LOGS_DIR)
    if args.clean:
        n = 0
        for name in os.listdir(LOGS_DIR):
            p = os.path.join(LOGS_DIR, name)
            try:
                if os.path.isfile(p) and time.time() - os.path.getmtime(p) > args.older_than_days * 86400:
                    os.unlink(p)
                    n += 1
            except OSError:
                pass
        print("removed %d file(s) older than %d days" % (n, args.older_than_days))
        return 0
    files = []
    for name in os.listdir(LOGS_DIR):
        p = os.path.join(LOGS_DIR, name)
        try:
            files.append((os.path.getmtime(p), name, os.path.getsize(p)))
        except OSError:
            pass
    if not files:
        print("logs dir is empty: %s" % LOGS_DIR)
        print("")
        print("Put a task's own log here (or symlink it) and `estimate --pid`")
        print("will find it automatically, for your jobs and for other people's.")
        return 0
    for mtime, name, size in sorted(files, reverse=True)[:args.limit]:
        print("%-40s %8.1fMB  %s" % (
            name, size / 1048576.0,
            datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")))
    return 0


def cmd_config(args):
    c = load_config()
    for s in args.set or []:
        if "=" not in s:
            print("error: expected key=value", file=sys.stderr)
            return 4
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip()
        if k not in DEFAULT_CONFIG:
            print("error: unknown config key %s (known: %s)"
                  % (k, ", ".join(sorted(DEFAULT_CONFIG))), file=sys.stderr)
            return 4
        if k == "gpu_index":
            c[k] = None if v.lower() in ("", "none", "null") else int(v)
        else:
            c[k] = v
    save_config(c)
    if args.json:
        print(json.dumps(c, indent=2, ensure_ascii=False))
        return 0
    print("config: %s" % CONFIG_PATH)
    print("  owner       %s" % c["owner"])
    print("  shared_dir  %s" % (c["shared_dir"] or "(not set: single-user mode)"))
    print("  gpu_index   %s" % (c["gpu_index"] if c["gpu_index"] is not None
                                else "(all cards)"))
    print("  notify_cmd  %s" % (c["notify_cmd"] or "(none)"))
    if c["shared_dir"]:
        d = shared_dir(c)
        print("")
        print("  claims      %s" % shared_claims_dir(c))
        print("  published   %s" % shared_publish_dir(c))
        print("  exists      %s" % os.path.isdir(d))
        print("  readable    %s" % os.access(d, os.R_OK))
        print("  writable    %s" % os.access(d, os.W_OK))
        others = [c2 for c2 in load_claims(c) if c2.get("owner") != c["owner"]]
        print("  other users %d live claim(s)" % len(others))
    return 0


def cmd_ledger(args):
    rows = read_ledger(args.limit, set(args.event.split(",")) if args.event else None)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    if not rows:
        print("ledger is empty: %s" % LEDGER_PATH)
        return 0
    for e in rows:
        bits = []
        for k in ("task_id", "to_owner", "to_task", "exit_code", "duration_s",
                  "vram_gb", "claimed"):
            if e.get(k) is not None:
                bits.append("%s=%s" % (k, e[k]))
        why = ""
        if e.get("reasons"):
            why = "  <- " + str(e["reasons"][0])[:80]
        print("%s  %-14s %-20s %s%s"
              % (e.get("ts", "")[:19], e.get("event"), e.get("owner") or "",
                 " ".join(bits), why))
    return 0


def _human_duration(s):
    if s is None:
        return "?"
    s = int(max(0, s))
    if s < 90:
        return "%ds" % s
    if s < 5400:
        return "%.0f min" % (s / 60.0)
    return "%.1f h" % (s / 3600.0)


def cmd_explain(args):
    """Why is my job not running?  In words, with the numbers behind them.

    Every other scheduler makes you infer this.  We already carry the reasons
    in the decision, so the only work is prose.
    """
    q = load_queue()
    config = load_config()
    snap = snapshot()
    mem = load_memory()
    obs = load_observations()
    claims = load_claims(config)
    intents = load_intents(config)
    dec = evaluate(q, args.task, snap, mem, obs, config, claims, intents)

    waiting = None
    if args.task:
        waiting = find_task(q, args.task)
    elif dec.get("task"):
        waiting = dec["task"]
    else:
        pend = pending_tasks(q, mem, snap)
        waiting = pend[0] if pend else None

    lines = []
    if waiting is None:
        lines.append("There is nothing pending, so there is nothing to explain.")
    else:
        since = waiting.get("started_at") or waiting.get("created_at") or ""
        waited = None
        try:
            waited = (datetime.now(timezone.utc).astimezone()
                      - datetime.fromisoformat(since)).total_seconds()
        except Exception:
            pass
        c = waiting.get("cost") or {}
        head = "Task '%s' wants %.2fGB and %.0f%% of the card" % (
            waiting["id"], float(c.get("vram_gb", 0) or 0),
            float(c.get("util_pct", 0) or 0))
        if waited:
            head += ", and has been waiting %s" % _human_duration(waited)
        lines.append(head + ".")

        if dec["runnable"]:
            lines.append("It can start right now. Run: bider run %s" % waiting["id"])
        else:
            lines.append("It cannot start yet:")

        # memory blocks first: they outrank capacity
        for b in dec["memory"]["blocks"]:
            k = b.get("kind")
            if k == "window":
                lines.append("  - %s reserved this time (%s). We do not take the "
                             "card during a reserved window."
                             % (b.get("person") or "somebody", b.get("when") or ""))
            elif k == "window_imminent":
                lines.append("  - %s" % b["reason"])
            elif k == "protected":
                names = ", ".join("%s (pid %s)" % (os.path.basename(p["name"] or ""),
                                                   p["pid"])
                                  for p in b.get("procs") or [])
                lines.append("  - %s is running and is protected%s. Waiting for it "
                             "to finish is the point." % (names or "a protected process",
                                                          "" if not b.get("because")
                                                          else " (%s)" % b["because"]))
            else:
                lines.append("  - %s" % b.get("reason"))

        # capacity
        for r in dec["reasons"]:
            if r in [b.get("reason") for b in dec["memory"]["blocks"]]:
                continue
            if r.startswith("fits:"):
                continue
            lines.append("  - %s" % r)
        for s in dec.get("skipped") or []:
            if s["task_id"] != waiting["id"]:
                lines.append("  - task '%s' was tried first and skipped: %s"
                             % (s["task_id"], s["reason"]))

        # who is holding it
        others = dec["claims"]["others"]
        procs = foreign_procs(snap, config)
        if procs or others:
            lines.append("")
            lines.append("On the card right now:")
            for p in procs:
                lines.append("  - pid %-7s %s" % (p["pid"], _short_proc(p)))
            for c in others:
                lines.append("  - %s claimed %.2fGB %.0fs ago for task '%s'"
                             % (c.get("owner"), c.get("vram_gb") or 0,
                                c.get("age_s") or 0, c.get("task_id")))

        # intent
        yield_to = (dec.get("intent") or {}).get("yield_to")
        if yield_to:
            lines.append("")
            lines.append("Standing down for %s, who registered for %.2fGB at %s "
                         "- before us." % (yield_to.get("owner"),
                                           yield_to.get("vram_gb") or 0,
                                           yield_to.get("since")))
        sharing = [i for i in intents if i.get("owner") != config.get("owner")]
        if sharing:
            lines.append("")
            lines.append("Other people waiting:")
            for i in sharing:
                for t in (i.get("tasks") or [])[:3]:
                    lines.append("  - %-12s wants %.2fGB since %s"
                                 % (i.get("owner"), float(t.get("vram_gb") or 0),
                                    t.get("since") or "?"))

        # a projection, clearly labelled
        eta = dec.get("foreign_eta_s")
        if eta is not None:
            lines.append("")
            lines.append("From the progress in their log, the job holding the card "
                         "looks like it has about %s left." % _human_duration(eta))

    print("\n".join(lines))

    # and what actually happened recently, so "still waiting" is not the whole story
    hist = [e for e in read_ledger(500)
            if e.get("event") in ("yielded", "triggered", "refused", "started")
            and (not args.task or e.get("task_id") == args.task)]
    if hist:
        print("")
        print("Recent history for this queue:")
        for e in hist[-6:]:
            extra = ""
            if e.get("event") == "yielded":
                extra = " -> %s" % e.get("to_owner")
            elif e.get("reasons"):
                extra = "  (%s)" % str(e["reasons"][0])[:70]
            print("  %s  %-10s %s%s" % (e.get("ts", "")[:19], e.get("event"),
                                        e.get("task_id") or "", extra))
    if args.json:
        print("")
        print(json.dumps(dec, indent=2, ensure_ascii=False))
    return 0


def _short_proc(p):
    name = os.path.basename((p.get("name") or "").replace("\\", "/"))
    return name


def cmd_stats(args):
    """What the card has actually been doing: the honest utilisation number.

    Read from the ledger, so it answers 'does this thing help?' with events
    rather than with an estimate.
    """
    rows = read_ledger(args.limit)
    if not rows:
        print("no ledger yet: %s" % LEDGER_PATH)
        print("")
        print("The watcher writes to it as it runs. Start one with:")
        print("  bider watch --stay-alive --daemon")
        return 0

    started = [e for e in rows if e["event"] == "started"]
    finished = [e for e in rows if e["event"] == "finished"]
    triggered = [e for e in rows if e["event"] == "triggered"]
    yielded = [e for e in rows if e["event"] == "yielded"]
    refused = [e for e in rows if e["event"] == "refused"]

    span_s = rows[-1].get("at_epoch", 0) - rows[0].get("at_epoch", 0)
    days = max(span_s / 86400.0, 1e-9)
    gpu_seconds = sum(e.get("duration_s") or 0 for e in finished)
    util = gpu_seconds / (days * 86400.0) * 100.0 if span_s > 0 else 0.0

    print("ledger: %s" % LEDGER_PATH)
    print("window: %s -> %s  (%.1f days)"
          % (rows[0].get("ts", "")[:19], rows[-1].get("ts", "")[:19], days))
    print("")
    print("  runs started        %d" % len(started))
    print("  runs finished       %d" % len(finished))
    if finished:
        durs = sorted(e.get("duration_s") or 0 for e in finished)
        print("  median run length   %s" % _human_duration(durs[len(durs) // 2]))
        print("  GPU time we used    %.1f h" % (gpu_seconds / 3600.0))
    print("  triggers fired      %d" % len(triggered))
    print("  yielded to others   %d" % len(yielded))
    print("  refusals            %d" % len(refused))
    if finished:
        print("")
        print("  our share of wall-clock GPU time: %.1f%%" % util)
        print("  (a lower number is not bad: it means we only took what was free)")
    owners = {}
    for e in started:
        owners[e.get("owner") or "?"] = owners.get(e.get("owner") or "?", 0) + 1
    if len(owners) > 1:
        print("")
        print("  starts by owner:")
        for o, n in sorted(owners.items(), key=lambda kv: -kv[1]):
            print("    %-16s %d" % (o, n))
    if yielded:
        print("")
        print("  stood down for:")
        for e in yielded[-5:]:
            print("    %s  %s" % (e.get("ts", "")[:19], e.get("to_owner")))
    return 0


def cmd_runs(args):
    if not os.path.isdir(RUNS_DIR):
        print("no runs yet")
        return 0
    rows = []
    for d in sorted(os.listdir(RUNS_DIR), reverse=True):
        m = read_json(os.path.join(RUNS_DIR, d, "meta.json"))
        if m:
            rows.append(m)
    for m in rows[:args.limit]:
        print("%-26s %-14s rc=%-5s %-9s %ss  %s" % (
            m.get("run_id"), m.get("task_id"), _fmt(m.get("exit_code")),
            m.get("status"), m.get("duration_s"), (m.get("command") or "")[:50]))
    return 0


def cmd_log(args):
    d = args.run
    if not os.path.isdir(os.path.join(RUNS_DIR, d)):
        cands = [x for x in os.listdir(RUNS_DIR) if x.startswith(d)] if os.path.isdir(RUNS_DIR) else []
        if len(cands) == 1:
            d = cands[0]
        else:
            print("error: run not found: %s" % args.run, file=sys.stderr)
            return 4
    meta = os.path.join(RUNS_DIR, d, "meta.json")
    path = os.path.join(RUNS_DIR, d, "stdout.log")
    if not os.path.isfile(path):
        path = (read_json(meta, {}) or {}).get("log") or path
    if args.follow:
        pos = 0
        while True:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                    pos = fh.tell()
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                time.sleep(1.0)
            except KeyboardInterrupt:
                return 0
            except OSError:
                time.sleep(1.0)
    print_tail(path, args.tail)
    return 0


def cmd_selftest(args):
    """Deterministic checks for the trigger logic using a fake GPU."""
    fails = []
    home = args.home_dir or tempfile.mkdtemp(prefix="bider-selftest-")
    ensure_dir(home)
    old_home = HOME
    set_home(home)

    def check(name, cond, extra=None):
        detail = ""
        if extra is not None and not cond:
            detail = "   got: %r" % (extra,)
        print("%-46s %s%s" % (name, "ok" if cond else "FAIL", detail))
        if not cond:
            fails.append(name)

    CFG = dict(DEFAULT_CONFIG, owner="me")

    def q_with(tasks, **policy):
        p = dict(DEFAULT_POLICY)
        p.update(policy)
        return {"version": 1, "policy": p, "tasks": tasks}

    def task(tid, vram, util, **kw):
        t = {"id": tid, "name": tid, "command": "echo hi", "status": "pending",
             "cost": {"vram_gb": vram, "gpus": 1, "util_pct": util},
             "priority": 0, "created_at": "2026-01-01T00:00:00+00:00"}
        t.update(kw)
        return t

    def snap(free_gb, util, procs=()):
        return {"ts": time.time(), "source": "fake", "error": None,
                "gpus": [{"index": 0, "name": "fake", "driver": "0",
                          "mem_total_mb": 6144, "mem_used_mb": 6144 - free_gb * 1024,
                          "mem_free_mb": free_gb * 1024, "util_pct": util, "temp_c": 50}],
                "procs": list(procs)}

    NOMEM = {"version": 1, "entries": [], "advice": []}
    NOOBS = {"version": 1, "observations": []}

    def decide(q, tid, s, memo=None, claims=None, intents=None, cfg=None):
        """evaluate against clean in-memory state, never the user's files."""
        return evaluate(q, tid, s, memo or NOMEM, NOOBS, cfg or CFG,
                        claims if claims is not None else [],
                        intents if intents is not None else [])

    def raises(fn):
        try:
            fn()
            return False
        except ValueError:
            return True

    def claim(owner, vram, util=10.0, **kw):
        c = {"owner": owner, "host": "somewhere", "pid": 4242,
             "task_id": "theirs", "vram_gb": vram, "util_pct": util,
             "gpus": 1, "at_epoch": now_epoch() - 5,
             "expires_at": now_epoch() + 60}
        c.update(kw)
        return c

    old = os.environ.get("BIDER_FAKE_GPU")
    try:
        # --- ordering -----------------------------------------------------
        q = q_with([task("heavy", 4, 80), task("light", 0.5, 10), task("mid", 2, 30)])
        order = [t["id"] for t in pending_tasks(q)]
        check("lightest requirement goes first", order == ["light", "mid", "heavy"])

        q = q_with([task("heavy", 4, 80), task("light", 0.5, 10, pinned=True)])
        check("pinned beats lighter", [t["id"] for t in pending_tasks(q)][0] == "heavy" or
              [t["id"] for t in pending_tasks(q)][0] == "light")
        check("pinned is first", pending_tasks(q)[0]["id"] == "light")

        q = q_with([task("a", 1, 10), task("b", 3, 30, rank=0.0)])
        check("explicit rank overrides footprint", pending_tasks(q)[0]["id"] == "b")

        q = q_with([task("a", 1, 10, priority=99), task("b", 2, 20)])
        check("priority breaks footprint ties only", pending_tasks(q)[0]["id"] == "a")

        # --- capacity -----------------------------------------------------
        q = q_with([task("light", 0.5, 10)])
        dec = decide(q, None, snap(6.0, 0))
        check("light task runs on an idle gpu", dec["runnable"])

        q = q_with([task("big", 5.0, 10)])
        dec = decide(q, None, snap(6.0, 0))
        check("task exceeding reserve is refused", not dec["runnable"])
        check("refusal names the vram shortfall",
              any("vram short" in r for r in dec["reasons"]))

        q = q_with([task("busy", 1.0, 90)])
        dec = decide(q, None, snap(6.0, 0))
        check("task needing >idle util is refused", not dec["runnable"])
        check("refusal names the util shortfall",
              any("util short" in r for r in dec["reasons"]))

        q = q_with([task("light", 0.5, 10)])
        dec = decide(q, None, snap(1.0, 0))
        check("busy gpu (low free vram) blocks", not dec["runnable"])

        # --- reserve math -------------------------------------------------
        q = q_with([task("t", 2.0, 10)], reserve_vram_gb=2.0)
        dec = decide(q, None, snap(4.0, 0))
        check("exactly hitting the reserve is allowed", dec["runnable"])
        dec = decide(q, None, snap(3.9, 0))
        check("just under the reserve is refused", not dec["runnable"])

        # --- concurrency --------------------------------------------------
        r = task("running", 1, 10, status="running")
        q = q_with([r, task("next", 0.5, 10)])
        check("second task blocked while one runs",
              not decide(q, None, snap(6, 0))["runnable"])
        q2 = q_with([r, task("next", 0.5, 10)], allow_concurrent=True)
        check("allow_concurrent lets them overlap",
              decide(q2, None, snap(6, 0))["runnable"])

        # --- first fit: a head that stops fitting must not stall the queue --
        q = q_with([task("big", 5.0, 10, pinned=True), task("small", 0.5, 10)])
        dec = decide(q, None, snap(6.0, 0))
        check("a pinned task that no longer fits does not stall the queue",
              dec["runnable"] and dec["task_id"] == "small", dec["task_id"])
        check("says how many it skipped",
              any("skipped" in r for r in dec["reasons"]), dec["reasons"])
        check("records why the skipped task was skipped",
              dec["skipped"] and dec["skipped"][0]["task_id"] == "big",
              dec.get("skipped"))

        q = q_with([task("big", 5.0, 10)], first_fit=False)
        check("first_fit=false restores strict head-of-queue",
              not decide(q, None, snap(6.0, 0))["runnable"])

        # --- change detection ---------------------------------------------
        p0 = [{"pid": 100, "name": "other.exe", "mem_mb": None}]
        chg = change_events(snap(1.0, 90, p0), snap(6.0, 0))
        kinds = set(e["kind"] for e in chg)
        check("detects a foreign process exiting", "proc_exited" in kinds)
        check("detects freed vram", "vram_freed" in kinds)
        check("detects utilization dropping", "util_dropped" in kinds)
        check("reports which pid left",
              any(e["kind"] == "proc_exited" and e["procs"][0]["pid"] == 100
                  for e in chg))

        # --- memory: time windows ------------------------------------------
        def win(frm, to, days="mon-sun", person="person"):
            return {"kind": "window", "id": "w", "person": person,
                    "from": frm, "to": to, "days": days}

        check("parses clock values", parse_clock("14:30") == 870)
        check("rejects a bad clock", raises(lambda: parse_clock("25:00")))
        check("parses a weekday range", parse_days("mon-fri") == set([0, 1, 2, 3, 4]))
        check("parses a wrapping weekday range",
              parse_days("fri-mon") == set([4, 5, 6, 0]), sorted(parse_days("fri-mon")))
        check("empty days means every day", parse_days("") == set(range(7)))

        w = win("14:00", "18:00", "mon-fri")
        check("weekday window is active inside it",
              window_active(w, datetime(2026, 10, 19, 15, 0)))
        check("weekday window is idle at the weekend",
              not window_active(w, datetime(2026, 10, 17, 15, 0)))
        check("weekday window is idle before it",
              not window_active(w, datetime(2026, 10, 19, 9, 0)))

        w = win("22:00", "02:00")
        check("overnight window active before midnight",
              window_active(w, datetime(2026, 10, 19, 23, 30)))
        check("overnight window active after midnight",
              window_active(w, datetime(2026, 10, 20, 1, 0)))
        check("overnight window idle mid-afternoon",
              not window_active(w, datetime(2026, 10, 20, 15, 0)))

        memo = {"version": 1, "entries": [win("00:00", "23:59", person="wei")],
                "advice": []}
        q = q_with([task("light", 0.5, 10)])
        dec = evaluate(q, None, snap(6.0, 0), memo, NOOBS)
        check("a reserved window blocks everything", not dec["runnable"])
        check("the block is reported as memory, not as capacity",
              dec["memory"]["blocks"] and
              dec["memory"]["blocks"][0]["kind"] == "window",
              dec["memory"]["blocks"])
        check("the block names the person",
              "wei" in dec["memory"]["blocks"][0]["reason"])

        mem = {"version": 1, "entries": [
            {"kind": "protected", "id": "wei", "person": "wei", "name": "wei",
             "match": {"name": "weijob"}, "because": "deadline"}], "advice": []}
        q = q_with([task("light", 0.5, 10)])
        dec = evaluate(q, None, snap(6.0, 0, [{"pid": 77, "name": "weijob.exe"}]),
                       mem, NOOBS)
        check("a protected process blocks the queue", not dec["runnable"])
        check("the block names the live pid",
              dec["memory"]["blocks"] and
              dec["memory"]["blocks"][0]["procs"][0]["pid"] == 77)

        # an unrelated person
        dec = evaluate(q, None, snap(6.0, 0, [{"pid": 9, "name": "someone_else.exe"}]),
                       mem, NOOBS)
        check("an unrelated process does not block", dec["runnable"])

        # a protection that has lapsed by date
        lapsed = {"version": 1, "entries": [
            {"kind": "protected", "id": "old", "person": "old", "name": "old",
             "match": {"name": "weijob"}, "until": "2020-01-01"}], "advice": []}
        dec = evaluate(q, None, snap(6.0, 0, [{"pid": 77, "name": "weijob.exe"}]),
                       lapsed, NOOBS)
        check("a lapsed protection does not block", dec["runnable"], dec["reasons"])

        # a protection the user pinned to one of our own tasks
        scoped = {"version": 1, "entries": [
            {"kind": "protected", "id": "wei", "person": "wei", "name": "wei",
             "match": {"name": "weijob"}, "task": "bigjob"}], "advice": []}
        q_scope = q_with([task("small", 0.5, 10), task("bigjob", 1.0, 10)])
        dec = evaluate(q_scope, None, snap(6.0, 0, [{"pid": 77, "name": "weijob.exe"}]),
                       scoped, NOOBS)
        check("a task-scoped protection still gates the queue", not dec["runnable"])

        # --- memory: profile advice -----------------------------------------
        memo = {"version": 1, "entries": [
            {"kind": "profile", "id": "evalfam", "family": "evalfam",
             "shape": {"vram": "heavy", "util": "light"}}], "advice": []}
        busy = [{"pid": 8, "name": "evalfam_worker.exe"}]
        q = q_with([task("vram-hog", 5.0, 10), task("compute-hog", 0.6, 80)])
        advice, notes = profile_advice(memo, NOOBS, snap(6.0, 5, busy),
                                       q["tasks"])
        bias = dict((a["task"], a["bias"]) for a in advice)
        check("advice favours compute-heavy when vram-heavy holds the card",
              bias.get("compute-hog", 0) < 0 and bias.get("vram-hog", 0) > 0, bias)
        check("advice explains itself",
              any("VRAM" in n or "vram" in n for n in notes), notes)
        check("advice reorders the queue",
              pending_tasks(q, memo, snap(6.0, 5, busy))[0]["id"] == "compute-hog",
              [t["id"] for t in pending_tasks(q, memo, snap(6.0, 5, busy))])
        q_pin = q_with([task("vram-hog", 5.0, 10, pinned=True),
                        task("compute-hog", 0.6, 80)])
        check("a pinned task outranks memory advice",
              pending_tasks(q_pin, memo, snap(6.0, 5, busy))[0]["id"] == "vram-hog")

        dec = evaluate(q, None, snap(6.0, 5, busy), memo, NOOBS)
        check("advice is surfaced in the decision",
              dec["memory"]["advice"] and dec["memory"]["notes"], dec["memory"])

        # --- memory: building entries ---------------------------------------
        memo2 = {"version": 1, "entries": [], "advice": []}
        add_memory_entry(memo2, "window", person="wei",
                         **{"from": "09:00", "to": "10:00", "days": "mon-fri",
                            "because": "standup"})
        check("add_memory_entry assigns an id",
              memo2["entries"][0]["id"].startswith("window"))
        check("find_memory finds it by person", find_memory(memo2, "wei") is not None)

        check("matches a process by partial name",
              proc_matches({"name": "weijob"}, {"name": "D:\\x\\weijob_main.exe"}))
        check("matches a process by pid",
              proc_matches({"pid": 42}, {"pid": 42, "name": "z"}))
        check("does not match an unrelated name",
              not proc_matches({"name": "weijob"}, {"name": "torchrun.exe"}))

        # a dormant protection: remembered, but nothing to match right now
        dormant = {"version": 1, "entries": [
            {"kind": "protected", "id": "wei", "person": "wei", "name": "wei",
             "match": {"name": "weijob"}}], "advice": []}
        q1 = q_with([task("light", 0.5, 10)])
        dec = evaluate(q1, None, snap(6.0, 0), dormant, NOOBS)
        check("a dormant protection does not block", dec["runnable"], dec["reasons"])
        check("a dormant protection is still remembered",
              len(dormant_entries(dormant)) == 1)

        check("families ignore worker indices",
              _family("D:\\x\\train_worker_3.exe") == "train",
              _family("D:\\x\\train_worker_3.exe"))
        check("families ignore the extension",
              _family("/opt/eval.py") == "eval", _family("/opt/eval.py"))

        # --- memory: shape learning -----------------------------------------
        obs = {"version": 1, "observations": [
            {"kind": "shape", "family": "evalfam", "gpu_util_pct": u,
             "gpu_used_gb": 4.5, "samples": 1, "at": now_iso()}
            for u in (8, 12, 9, 11)]}
        obs["observations"][0]["samples"] = 4        # one family, four polls
        shape = shape_of_family(obs, "evalfam", min_samples=3)
        check("learns a shape from repeated samples",
              shape and shape["util"] == "light" and shape["vram"] == "heavy",
              shape)
        check("counts samples, not rows",
              shape and shape["samples"] == 7, shape)
        check("refuses to learn from too few samples",
              shape_of_family(obs, "evalfam", min_samples=99) is None)
        check("a freshly seen family is not profiled",
              shape_of_family({"version": 1, "observations": [
                  {"kind": "shape", "family": "new", "gpu_util_pct": 5,
                   "gpu_used_gb": 5.0, "samples": 1}]}, "new", min_samples=3) is None)

        # --- progress parsing ---------------------------------------------
        p = _scan_progress(" 42%|####      | 21/50 [00:10<00:13, 2.1it/s]")
        check("parses tqdm percentage", p and abs(p[2] - 0.42) < 0.01)
        p = _scan_progress("Epoch 3/10\nloss=0.5")
        check("parses epoch counters", p and p[0] == 3 and p[1] == 10)
        p = _scan_progress("step 250/1000")
        check("parses step counters", p and abs(p[2] - 0.25) < 1e-6)
        p = _scan_progress("no numbers here")
        check("returns None without a marker", p is None)
        p = _scan_progress("epoch 5/10\nepoch 7/10")
        check("takes the latest marker", p and p[0] == 7)
        check("humanises durations", _human(3720) == "1h02m" and _human(90) == "1m30s")

        # --- ETA projection -------------------------------------------------
        def samples(pairs):
            return [{"t": float(t), "frac": f, "cur": int(f * 100), "tot": 100}
                    for t, f in pairs]

        eta, basis, n = _eta_from_samples(samples([(0, 0.10), (10, 0.20)]), 600)
        check("projects an ETA from the steady rate",
              eta is not None and abs(eta - 80) < 1.5, eta)
        check("the ETA basis names the rate", basis and "%/min" in basis, basis)

        eta, basis, n = _eta_from_samples(samples([(0, 0.10), (10, 0.20),
                                                   (20, 0.60)]), 12)
        check("a short window projects on the recent rate only",
              eta is not None and abs(eta - 10) < 1.5, eta)

        eta, basis, n = _eta_from_samples(samples([(0, 0.60), (10, 0.05),
                                                   (20, 0.10)]), 600)
        check("ignores samples from before a restart",
              eta is not None and abs(eta - 180) < 3, eta)
        check("uses everything inside the window",
              _eta_from_samples(samples([(0, 0.10), (500, 0.20)]), 600)[0] is not None)
        check("drops samples outside the window",
              _eta_from_samples(samples([(0, 0.10), (500, 0.20)]), 60)[0] is None)
        check("no ETA when progress is not moving",
              _eta_from_samples(samples([(0, 0.10), (10, 0.10)]), 600)[0] is None)
        check("no ETA from a single sample",
              _eta_from_samples(samples([(0, 0.5)]), 600)[0] is None)
        check("no ETA from nothing at all",
              _eta_from_samples([], 600)[0] is None)

        # --- claims: the reservation that stops two bider users colliding ------
        q = q_with([task("mine", 1.0, 20)])
        dec = decide(q, None, snap(6.0, 0))
        check("no claims means the whole card is free", dec["runnable"])

        dec = decide(q, None, snap(6.0, 0), claims=[claim("wei", 4.0, 10)])
        check("somebody else's claim is subtracted from free vram",
              not dec["runnable"], dec["reasons"])
        check("the shortfall says the space is reserved",
              any("reserved" in r for r in dec["reasons"]), dec["reasons"])
        check("the claim is reported with its owner",
              dec["claims"]["others"] and
              dec["claims"]["others"][0]["owner"] == "wei",
              dec["claims"]["others"])

        dec = decide(q, None, snap(6.0, 0), claims=[claim("wei", 0.5, 5)])
        check("a claim that leaves room does not block", dec["runnable"])

        dec = decide(q, None, snap(6.0, 0), claims=[claim("me", 4.0, 10)])
        check("our own claim also counts against the card", not dec["runnable"])

        dec = decide(q, None, snap(6.0, 0),
                     claims=[claim("wei", 4.0, 10, pid=1,
                                   host=host_name(),
                                   at_epoch=now_epoch() - 3600,
                                   expires_at=now_epoch() - 10)])
        check("an expired claim is ignored", dec["runnable"], dec["reasons"])

        # a claim whose process died on this host, past the grace period
        dec = decide(q, None, snap(6.0, 0),
                     claims=[claim("wei", 4.0, 10, pid=999999,
                                   host=host_name(),
                                   at_epoch=now_epoch() - 300,
                                   expires_at=now_epoch() + 600)])
        check("a claim for a dead local pid is ignored", dec["runnable"],
              dec["reasons"])

        # ...but a claim made seconds ago is honoured, because the process may
        # be starting up
        dec = decide(q, None, snap(6.0, 0),
                     claims=[claim("wei", 4.0, 10, pid=999999,
                                   host=host_name(), at_epoch=now_epoch(),
                                   expires_at=now_epoch() + 600)])
        check("a claim made seconds ago is honoured even if the pid is unseen",
              not dec["runnable"], dec["reasons"])

        # --- intent: read-only, and only yields when it should --------------
        ours = task("mine", 1.0, 20, created_at="2026-06-02T00:00:00+00:00")
        their_board = [{"owner": "wei", "host": "box2",
                        "tasks": [{"task_id": "big", "vram_gb": 5.0,
                                   "since": "2026-06-01T00:00:00+00:00"}]}]
        y = competes_with_us(snap(6.0, 0)["gpus"] and
                             {"mem_free_gb": 6.0}, ours, CFG, their_board)
        check("yields to somebody who declared earlier for the same room",
              y is not None and y["owner"] == "wei", y)

        their_board2 = [{"owner": "wei", "host": "box2",
                         "tasks": [{"task_id": "big", "vram_gb": 5.0,
                                    "since": "2026-07-01T00:00:00+00:00"}]}]
        check("does not yield to somebody who declared later",
              competes_with_us({"mem_free_gb": 6.0}, ours, CFG, their_board2) is None)

        their_board3 = [{"owner": "wei", "host": "box2",
                         "tasks": [{"task_id": "small", "vram_gb": 0.2,
                                    "since": "2026-06-01T00:00:00+00:00"}]}]
        check("does not yield when both tasks fit anyway",
              competes_with_us({"mem_free_gb": 6.0}, ours, CFG, their_board3) is None)

        mine_board = [{"owner": "me", "host": "box2",
                       "tasks": [{"task_id": "big", "vram_gb": 5.0,
                                  "since": "2026-06-01T00:00:00+00:00"}]}]
        check("never yields to ourselves",
              competes_with_us({"mem_free_gb": 6.0}, ours, CFG, mine_board) is None)

        # --- multi-card -----------------------------------------------------
        multi = {"ts": time.time(), "source": "fake", "error": None,
                 "gpus": [dict(snap(6.0, 0)["gpus"][0], index=0, uuid="GPU-a"),
                          dict(snap(2.0, 90)["gpus"][0], index=1, uuid="GPU-b")],
                 "procs": [{"gpu_uuid": "GPU-a", "pid": 1, "name": "a.exe"},
                           {"gpu_uuid": "GPU-b", "pid": 2, "name": "b.exe"}]}
        one = _restrict_to_gpu(multi, 0)
        check("restricting to one card drops the others",
              len(one["gpus"]) == 1 and one["gpus"][0]["index"] == 0)
        check("restricting also drops other cards' processes",
              [p["pid"] for p in one["procs"]] == [1], one["procs"])
        check("an unknown index leaves the snapshot alone",
              len(_restrict_to_gpu(multi, 9)["gpus"]) == 2)

        # --- the ledger -----------------------------------------------------
        record("started", task_id="t1", owner="me")
        record("yielded", task_id="t1", to_owner="wei")
        rows = read_ledger(10)
        check("the ledger records events", len(rows) >= 2, len(rows))
        check("the ledger keeps its fields",
              any(r.get("to_owner") == "wei" for r in rows), rows[-1])
        check("the ledger can be filtered",
              all(r["event"] == "yielded" for r in read_ledger(10, set(["yielded"]))))

        # --- config ---------------------------------------------------------
        c = load_config()
        check("config gets an owner by default", bool(c.get("owner")), c)
        c["shared_dir"] = os.path.join(home, "shared")
        save_config(c)
        check("shared_dir round-trips", load_config()["shared_dir"] == c["shared_dir"])
        check("claims live under shared_dir/claims",
              shared_claims_dir().replace("\\", "/").endswith("shared/claims"),
              shared_claims_dir())
        check("no shared_dir means local-only", shared_dir(dict(c, shared_dir="")) == "")

        # --- family naming ---------------------------------------------------
        check("families ignore worker indices",
              _family("D:\\x\\train_worker_3.exe") == "train",
              _family("D:\\x\\train_worker_3.exe"))
        check("families ignore the extension",
              _family("/opt/eval.py") == "eval", _family("/opt/eval.py"))

        # --- a full shared-directory round trip -------------------------------
        shared = os.path.join(home, "shared2")
        c = load_config()
        c["shared_dir"] = shared
        c["owner"] = "me"
        save_config(c)
        mem_mine = {"version": 1, "entries": [
            {"kind": "protected", "id": "mine-prot", "person": "me",
             "match": {"name": "myjob"}},
            {"kind": "window", "id": "mine-win", "person": "me",
             "from": "09:00", "to": "10:00", "days": "mon-fri"},
            {"kind": "protected", "id": "secret", "person": "me",
             "private": True, "match": {"name": "privatejob"}}], "advice": []}
        save_memory(mem_mine)
        p = publish_memory(mem_mine)
        check("publishing writes one file per owner", p and os.path.isfile(p), p)
        published = read_json(p)
        check("private entries are not published",
              all(e.get("id") != "secret" for e in published["entries"]),
              [e.get("id") for e in published["entries"]])
        check("the published file names its owner",
              published.get("owner") == "me", published.get("owner"))

        seen = load_memory()
        check("our own memory reads back", len(seen["entries"]) == 3, len(seen["entries"]))
        # a second user, reading the same directory
        c2 = dict(c, owner="wei")
        shared_files = shared_memory_files(c2)
        check("another user can see the published file", len(shared_files) == 1,
              shared_files)
        their_mem = read_json(shared_files[0])
        check("and sees the non-private entries",
              len(their_mem["entries"]) == 2, their_mem["entries"])

        # a claim we take, seen by a different owner
        c = load_config()
        c["owner"] = "me"
        save_config(c)
        cl = take_claim({"id": "t1", "cost": {"vram_gb": 3.0, "util_pct": 40}},
                        c, pid=os.getpid())
        check("taking a claim writes a file", os.path.isfile(cl["_path"]), cl["_path"])
        check("our own claim is visible to us", len(load_claims(c)) == 1)
        other_view = load_claims(dict(c, owner="wei"))
        check("another owner sees our claim", len(other_view) == 1, other_view)
        check("the claim carries what it reserved",
              other_view[0]["vram_gb"] == 3.0 and other_view[0]["util_pct"] == 40)
        check("claims add up", claims_totals(other_view)["vram_gb"] == 3.0)
        release_claim(cl)
        check("releasing removes it", len(load_claims(c)) == 0)

        # intent publishing
        save_queue(q_with([task("waiting", 2.0, 30)]))
        b = publish_intent(c)
        check("intent publishing lists what we wait for",
              b and b["tasks"] and b["tasks"][0]["task_id"] == "waiting", b)
        check("intent is readable by others", len(load_intents(dict(c, owner="wei"))) == 1)
        check("intent is never a claim",
              not [x for x in load_claims(dict(c, owner="wei"))
                   if x.get("task_id") == "waiting"])

        # --- adaptive polling ------------------------------------------------
        pol_q = q_with([task("t", 1.0, 10)])
        d_run = decide(pol_q, None, snap(6.0, 0))
        check("polling speeds up when a task can start",
              _adaptive_interval(20.0, d_run, 2, 20.0, CFG) <= 6.0,
              _adaptive_interval(20.0, d_run, 2, 20.0, CFG))
        d_busy = decide(pol_q, None, snap(0.5, 95))
        check("polling slows down when the card is busy and unknowable",
              _adaptive_interval(20.0, d_busy, 2, 20.0, CFG) >= 20.0,
              _adaptive_interval(20.0, d_busy, 2, 20.0, CFG))
        d_busy["foreign_eta_s"] = 40.0
        check("polling tracks a projected finish time",
              _adaptive_interval(20.0, d_busy, 2, 20.0, CFG) <= 10.0,
              _adaptive_interval(20.0, d_busy, 2, 20.0, CFG))

        # --- fake source --------------------------------------------------
        os.environ["BIDER_FAKE_GPU"] = json.dumps(snap(3.0, 5))
        s = snapshot()
        check("BIDER_FAKE_GPU source is used", s["source"] == "fake" and
              abs(gpu_aggregate(s)["mem_free_gb"] - 3.0) < 0.01)
    finally:
        if old is None:
            os.environ.pop("BIDER_FAKE_GPU", None)
        else:
            os.environ["BIDER_FAKE_GPU"] = old
        set_home(old_home)
        if not args.home_dir:
            shutil.rmtree(home, ignore_errors=True)

    print("")
    if fails:
        print("SELFTEST FAILED: %d check(s)" % len(fails))
        for f in fails:
            print("  - %s" % f)
        return 1
    print("SELFTEST OK")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="bider",
        description="A polite GPU job queue: wait for the card, never take "
                    "anybody else's.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `bider selftest` to verify the scheduling logic without "
               "touching the GPU.")
    p.add_argument("--version", action="version", version="bider " + VERSION)
    p.add_argument("--home", dest="home_dir", metavar="DIR",
                   help="state directory (default $BIDER_HOME or ~/.bider)")
    sub = p.add_subparsers(dest="cmd")

    def sp(name, help_, **kw):
        s = sub.add_parser(name, help=help_, **kw)
        s.add_argument("--json", action="store_true", help="machine-readable output")
        return s

    s = sp("gpu", "show the current GPU snapshot")
    s.set_defaults(func=cmd_gpu)

    s = sp("procs", "list processes holding the GPU")
    s.set_defaults(func=cmd_procs)

    s = sp("list", "show the queue in run order", aliases=["ls"])
    s.add_argument("--pending-only", action="store_true")
    s.set_defaults(func=cmd_list)

    s = sp("add", "add an experiment to the queue")
    s.add_argument("command", help="shell command to run")
    s.add_argument("--name", help="human label")
    s.add_argument("--id", help="explicit id (default: slug of --name)")
    s.add_argument("--workdir", help="cwd for the command (default: current dir)")
    s.add_argument("--vram", type=float, default=1.0, help="GB it will occupy")
    s.add_argument("--gpus", type=int, default=1, help="how many GPUs it needs")
    s.add_argument("--util", type=float, default=10.0, help="%% of GPU utilization it needs")
    s.add_argument("--cpus", type=int, default=1, help="informational only")
    s.add_argument("--est-min", type=float, default=None, help="expected minutes")
    s.add_argument("--priority", type=float, default=0.0,
                   help="tie-breaker only; see --rank/--pin to reorder for real")
    s.add_argument("--pin", action="store_true", help="run this before everything else")
    s.add_argument("--rank", type=float, default=None, help="explicit position (0 = first)")
    s.add_argument("--env", action="append", metavar="K=V", help="extra env var (repeatable)")
    s.add_argument("--sandbox", metavar="DEVICES",
                   help="value for CUDA_VISIBLE_DEVICES on a multi-card machine")
    s.add_argument("--tag", action="append", help="free-form tag (repeatable)")
    s.add_argument("--note", help="free-form note")
    s.set_defaults(func=cmd_add)

    s = sp("update", "change a task in place")
    s.add_argument("task")
    s.add_argument("--set", action="append", metavar="FIELD=VALUE",
                   help="vram_gb|util_pct|gpus|cpus|priority|rank|est_minutes|"
                        "pinned|status|workdir|command|name|note|env.KEY")
    s.add_argument("--pin", action="store_true")
    s.add_argument("--unpin", action="store_true")
    s.add_argument("--priority", type=float)
    s.add_argument("--command")
    s.add_argument("--status", choices=["pending", "running", "done", "failed", "skipped"])
    s.set_defaults(func=cmd_update)

    s = sp("remove", "delete tasks from the queue", aliases=["rm"])
    s.add_argument("tasks", nargs="*", help="ids (prefixes ok)")
    s.add_argument("--done", help="also drop all tasks with this status")
    s.set_defaults(func=cmd_remove)

    s = sp("move", "reorder a task")
    s.add_argument("task")
    s.add_argument("--top", action="store_true", help="put it first")
    s.add_argument("--rank", type=float, help="explicit position")
    s.add_argument("--renumber", action="store_true", help="pin the current order explicitly")
    s.set_defaults(func=cmd_move)

    s = sp("pin", "run this task before anything else")
    s.add_argument("task")
    s.add_argument("--unpin", action="store_true")
    s.set_defaults(func=cmd_pin)

    s = sp("clear", "drop finished tasks")
    s.add_argument("--status", action="append")
    s.set_defaults(func=cmd_clear)

    s = sp("policy", "read or change scheduling policy")
    s.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="reserve_vram_gb|reserve_util_pct|poll_seconds|"
                        "poll_floor_seconds|poll_ceiling_seconds|settle_ticks|"
                        "allow_concurrent|first_fit|require_claim|claim_ttl_s")
    s.set_defaults(func=cmd_policy)

    s = sp("config", "who you are, and where the shared directory is")
    s.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="owner|shared_dir|gpu_index|notify_cmd|timezone")
    s.set_defaults(func=cmd_config)

    s = sp("intent", "what everybody on this card is waiting for")
    s.add_argument("board", nargs="?", default="show",
                   choices=["show", "publish", "clear"],
                   help="show (default) | publish ours | clear ours")
    s.set_defaults(func=cmd_intent)

    s = sp("explain", "why is my job not running?")
    s.add_argument("--task")
    s.set_defaults(func=cmd_explain)

    s = sp("stats", "what the card has actually been doing")
    s.add_argument("--limit", type=int, default=5000)
    s.set_defaults(func=cmd_stats)

    s = sp("ledger", "the audit trail of starts, yields and refusals")
    s.add_argument("--limit", type=int, default=60)
    s.add_argument("--event", help="comma-separated events to show")
    s.set_defaults(func=cmd_ledger)

    s = sp("logs", "the shared place to leave logs for ETA estimates")
    s.add_argument("--clean", action="store_true", help="remove old files")
    s.add_argument("--older-than-days", type=float, default=14.0)
    s.add_argument("--limit", type=int, default=30)
    s.set_defaults(func=cmd_logs)

    s = sp("memory", "what this machine has learned about its other users",
        aliases=["mem"])
    s.add_argument("--protect", metavar="PERSON",
                   help="do not disturb this person's processes")
    s.add_argument("--match", help="process name/path substring (default: the person)")
    s.add_argument("--pid", type=int, help="protect one specific pid")
    s.add_argument("--task", help="apply the protection only while this task runs")
    s.add_argument("--window", metavar="PERSON",
                   help="reserve recurring time for this person")
    s.add_argument("--from", dest="from_", metavar="HH:MM", help="window start")
    s.add_argument("--to", metavar="HH:MM", help="window end (may cross midnight)")
    s.add_argument("--days", help="mon-fri / sat,sun / fri-mon (default: every day)")
    s.add_argument("--until", metavar="YYYY-MM-DD",
                   help="let a protection lapse after this date")
    s.add_argument("--because", help="why - shown back to you and to Claude Code")
    s.add_argument("--profile", metavar="FAMILY",
                   help="remember how a process family behaves")
    s.add_argument("--profile-vram", choices=["heavy", "light"])
    s.add_argument("--profile-util", choices=["heavy", "light"])
    s.add_argument("--private", action="store_true",
                   help="keep this entry out of the shared directory")
    s.add_argument("--suggest", action="store_true",
                   help="learn process shapes from what has been observed")
    s.add_argument("--min-samples", type=int, default=3,
                   help="samples needed before --suggest believes a shape")
    s.add_argument("--forget", metavar="ID_OR_PERSON")
    s.add_argument("--clear", choices=list(MEMORY_KINDS))
    s.add_argument("--export", metavar="FILE", nargs="?", const="-",
                   help="write the memory out (default: stdout)")
    s.add_argument("--import-file", metavar="FILE", dest="import_file")
    s.set_defaults(func=cmd_memory)

    s = sp("observe", "record what the GPU is being used for right now")
    s.add_argument("--suggest", action="store_true",
                   help="also promote well-sampled shapes into memory")
    s.add_argument("--min-samples", type=int, default=3)
    s.set_defaults(func=cmd_observe)

    s = sp("probe", "one-shot check: can the next task start?")
    s.add_argument("--task", help="check a specific task instead of the head")
    s.set_defaults(func=cmd_probe)

    s = sp("status", "overview: gpu + queue + watcher")
    s.add_argument("--task")
    s.set_defaults(func=cmd_status)

    s = sp("watch", "monitor the GPU and report when a task becomes runnable")
    s.add_argument("--task", help="watch for one specific task")
    s.add_argument("--interval", type=float, default=None, help="seconds between polls")
    s.add_argument("--timeout-min", type=float, default=60.0,
                   help="give up after this long (default 60; 0 = never)")
    s.add_argument("--settle-ticks", type=int, default=None,
                   help="consecutive good polls required (default from policy)")
    s.add_argument("--require-change", action="store_true",
                   help="only trigger if something actually changed since start")
    s.add_argument("--stay-alive", action="store_true",
                   help="keep watching even when there is nothing pending")
    s.add_argument("--no-observe", action="store_true",
                   help="do not record process shapes while watching")
    s.add_argument("--no-eta", action="store_true",
                   help="do not try to project when other people's jobs finish")
    s.add_argument("--no-intent", action="store_true",
                   help="do not publish or read the intent board")
    s.add_argument("--no-sleep", action="store_true",
                   help="poll as fast as possible (testing; busy-waits)")
    s.add_argument("--treat-empty-as-timeout", action="store_true",
                   help="return the timeout code instead of NOTHING_PENDING")
    s.add_argument("--daemon", action="store_true", help="detach and keep watching")
    s.add_argument("--stop", action="store_true", help="stop the running watcher")
    s.add_argument("--status", action="store_true", help="show watcher state")
    s.add_argument("--intent", nargs="?", const="show", default=None,
                   choices=["show", "publish", "clear"],
                   help="read or publish the intent board (alias for `bider intent`)")
    s.add_argument("--on-trigger", choices=["none", "run", "claude"], default="none",
                   help="daemon only: none=just report, run=start the task, "
                        "claude=hand it to a headless claude session")
    s.set_defaults(func=cmd_watch)

    s = sp("run", "start a task from the queue")
    s.add_argument("task")
    s.add_argument("--no-check", action="store_true", help="skip the resource check")
    s.add_argument("--force", action="store_true", help="run even if not pending")
    s.add_argument("--dry-run", action="store_true", help="show what would run")
    s.add_argument("--detach", action="store_true", help="do not wait for it")
    s.add_argument("--timeout-min", type=float, default=None, help="kill it after N minutes")
    s.set_defaults(func=cmd_run)

    s = sp("runs", "list past runs")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_runs)

    s = sp("log", "show a run's output")
    s.add_argument("run")
    s.add_argument("--tail", type=int, default=60)
    s.add_argument("--follow", action="store_true")
    s.set_defaults(func=cmd_log)

    s = sp("estimate", "estimate when a running program will finish")
    s.add_argument("--pid", type=int, help="process id to watch")
    s.add_argument("--run", help="our own run id instead of --pid")
    s.add_argument("--log", help="log file to parse for progress")
    s.add_argument("--pattern", help="extra progress regex")
    s.add_argument("--total", type=float, help="total units when the log omits it")
    s.add_argument("--interval", type=float, default=10.0, help="sample period (seconds)")
    s.add_argument("--for-seconds", type=float, default=None,
                   help="keep sampling for this long (default: one shot)")
    s.add_argument("--window-seconds", type=float, default=None,
                   help="how much recent history the ETA uses (default: max(60, 6*interval))")
    s.add_argument("--eta-cpu-hours", action="store_true",
                   help="include the cpu-duty diagnostic")
    s.set_defaults(func=cmd_estimate)

    s = sp("watchproc", "keep sampling a running task until it ends")
    s.add_argument("--pid", type=int)
    s.add_argument("--run")
    s.add_argument("--log")
    s.add_argument("--pattern")
    s.add_argument("--total", type=float)
    s.add_argument("--interval", type=float, default=30.0)
    s.add_argument("--for-seconds", type=float, default=None)
    s.add_argument("--window-seconds", type=float, default=None)
    s.add_argument("--eta-cpu-hours", action="store_true")
    s.set_defaults(func=cmd_watchproc)

    s = sp("selftest", "verify scheduling and parsing logic")
    s.set_defaults(func=cmd_selftest)
    return p


def main(argv=None):
    _fix_stdout()
    parser = build_parser()
    # a first pass only to learn --home, so every path below is right
    pre, _ = parser.parse_known_args(argv)
    if getattr(pre, "home_dir", None):
        set_home(pre.home_dir)
    ensure_dir(HOME)
    ensure_dir(STATE_DIR)
    ensure_dir(RUNS_DIR)

    if getattr(pre, "cmd", None) == "selftest" and not pre.home_dir:
        pass  # cmd_selftest sets up its own throwaway home
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except RuntimeError as e:
        print("error: %s" % e, file=sys.stderr)
        return 4
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
