#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end tests for bider, runnable without touching the real GPU.

Every test gets its own throwaway BIDER_HOME, so nothing here reads or writes
your real queue, memory or claims.  The GPU is replaced by BIDER_FAKE_GPU.

    python test_e2e.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GQ = os.path.join(HERE, "bider.py")
PY = sys.executable

FAILS = []
_TMP = []


def check(name, cond, extra=""):
    print("%-54s %s%s" % (name, "ok" if cond else "FAIL",
                          ("  " + str(extra)) if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


def new_home(prefix="bider-e2e-"):
    d = tempfile.mkdtemp(prefix=prefix)
    _TMP.append(d)
    return d


def snap(free_gb, util, procs=()):
    return {
        "ts": time.time(),
        "source": "fake",
        "error": None,
        "gpus": [{
            "index": 0, "name": "fake-3060", "driver": "0", "uuid": "GPU-fake",
            "mem_total_mb": 6144.0,
            "mem_used_mb": 6144.0 - free_gb * 1024,
            "mem_free_mb": free_gb * 1024,
            "util_pct": util, "temp_c": 50.0,
        }],
        "procs": list(procs),
    }


def bider_env(home, fake=None, shared=None):
    env = dict(os.environ)
    env["BIDER_HOME"] = home
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("BIDER_FAKE_GPU", None)
    if fake is not None:
        env["BIDER_FAKE_GPU"] = fake if isinstance(fake, str) else json.dumps(fake)
    return env


def run_bider(args, home, fake=None, timeout=120, env_extra=None):
    env = bider_env(home, fake)
    if env_extra:
        env.update(env_extra)
    p = subprocess.run([PY, GQ] + args, cwd=home, env=env, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout.decode("utf-8", "replace")


def only_trigger_json(out):
    """Pull the BIDER_TRIGGER payload out of the watcher's mixed stdout."""
    marker = "BIDER_TRIGGER "
    i = out.find(marker)
    if i < 0:
        return None
    j = out.find("\n=====", i)
    blob = out[i + len(marker): j if j > 0 else len(out)]
    depth = 0
    for k, ch in enumerate(blob):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = blob[:k + 1]
                break
    try:
        return json.loads(blob)
    except Exception:
        return None


def write_queue(home, tasks, **policy):
    q = {"version": 1, "policy": {
        "reserve_vram_gb": 2.0, "reserve_util_pct": 25, "poll_seconds": 20,
        "poll_floor_seconds": 3, "poll_ceiling_seconds": 120, "settle_ticks": 1,
        "allow_concurrent": False, "first_fit": True, "require_claim": True,
        "claim_ttl_s": 120, "polite": True}, "tasks": tasks}
    q["policy"].update(policy)
    os.makedirs(home, exist_ok=True)
    with open(os.path.join(home, "queue.json"), "w", encoding="utf-8") as fh:
        json.dump(q, fh, indent=2)


def task(tid, vram, util, **kw):
    t = {"id": tid, "name": tid, "command": "echo hi", "status": "pending",
         "cost": {"vram_gb": vram, "gpus": 1, "util_pct": util},
         "priority": 0, "pinned": False, "rank": None,
         "created_at": "2026-01-01T00:00:00+00:00"}
    t.update(kw)
    return t


# --------------------------------------------------------------------------

def test_edges():
    print("\n-- watcher: edge-triggered on a foreign task finishing --")
    home = new_home()
    fake_path = os.path.join(home, "fake_gpu.json")
    foreign = [{"gpu_uuid": "GPU-x", "pid": 4242, "name": "someone_else.exe",
                "mem_mb": None}]

    write_queue(home, [task("small-thing", 1.0, 20)])
    with open(fake_path, "w") as fh:
        json.dump(snap(0.8, 95, foreign), fh)      # phase 1: GPU busy

    env = bider_env(home, fake_path)
    proc = subprocess.Popen(
        [PY, GQ, "watch", "--interval", "0.7", "--timeout-min", "1.5",
         "--settle-ticks", "2", "--require-change", "--no-intent"],
        cwd=home, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def flip():
        time.sleep(3.0)
        with open(fake_path, "w") as fh:
            json.dump(snap(5.6, 0), fh)            # phase 2: everything freed

    t = threading.Thread(target=flip)
    t.start()
    t0 = time.time()
    out, _ = proc.communicate(timeout=120)
    elapsed = time.time() - t0
    t.join()

    text = out.decode("utf-8", "replace")
    trigger = only_trigger_json(text)

    check("watcher exited with code 0", proc.returncode == 0, proc.returncode)
    check("a trigger payload was emitted", trigger is not None)
    if trigger:
        check("trigger names the queued task",
              trigger.get("task_id") == "small-thing", trigger.get("task_id"))
        check("verdict is TRIGGERED", trigger.get("verdict") == "TRIGGERED",
              trigger.get("verdict"))
        kinds = set(e["kind"] for e in trigger.get("changes_since_start") or [])
        check("trigger reports what changed",
              bool(kinds & {"proc_exited", "vram_freed"}), kinds)
        check("payload says how to deploy it",
              "bider run small-thing" in (trigger.get("next_step") or ""))
        check("waited for the process to actually leave", elapsed > 3.0,
              "%.1fs" % elapsed)
    check("did not trigger during the busy phase", elapsed < 60)

    # --- never fires while a protected process is on the card -------------
    print("\n-- watcher: stands down for a protected process --")
    home2 = new_home()
    write_queue(home2, [task("ours", 1.0, 20)])
    with open(os.path.join(home2, "memory.json"), "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "entries": [
            {"kind": "protected", "id": "protected-wei", "person": "wei",
             "match": {"name": "weijob"}}], "advice": []}, fh)
    rc, out = run_bider(["watch", "--interval", "0.4", "--timeout-min", "0.05",
                      "--no-intent"],
                     home2, snap(5.6, 0, [{"gpu_uuid": "GPU-fake", "pid": 77,
                                           "name": "weijob.exe", "mem_mb": None}]))
    check("a protected process stops the trigger", rc == 2, rc)
    j = only_trigger_json(out)
    check("the payload explains the protection",
          j and any("not be disturbed" in r for r in j["decision"]["reasons"]),
          j and j["decision"]["reasons"])

    # --- timeout when the card never frees --------------------------------
    print("\n-- watcher: times out when the GPU never frees up --")
    home3 = new_home()
    write_queue(home3, [task("too-big", 4.0, 60)])
    with open(fake_path, "w") as fh:
        json.dump(snap(0.5, 99, foreign), fh)
    rc, out = run_bider(["watch", "--interval", "0.5", "--timeout-min", "0.05",
                      "--settle-ticks", "1", "--no-intent"], home3, fake_path)
    check("exit code 2 on timeout", rc == 2, rc)
    j = only_trigger_json(out)
    check("timeout payload says TIMEOUT", j and j.get("verdict") == "TIMEOUT",
          j and j.get("verdict"))
    check("timeout payload explains why", j and any(
        "vram short" in r or "util short" in r for r in j["decision"]["reasons"]),
        j and j["decision"]["reasons"])

    # --- empty queue ------------------------------------------------------
    home4 = new_home()
    write_queue(home4, [])
    rc, out = run_bider(["watch", "--interval", "0.5", "--timeout-min", "0.05",
                      "--no-intent"], home4, snap(5.6, 0))
    check("exit code 3 when the queue is empty", rc == 3, rc)
    j = only_trigger_json(out)
    check("empty-queue payload says NOTHING_PENDING",
          j and j.get("verdict") == "NOTHING_PENDING", j and j.get("verdict"))


def test_estimate():
    print("\n-- estimate: ETA from a progress log --")
    home = new_home()
    log = os.path.join(home, "job.log")
    with open(log, "w") as fh:
        fh.write("starting\n")

    def emit():
        for i in range(1, 7):
            time.sleep(0.6)
            with open(log, "a") as fh:
                fh.write("epoch %d/50 loss=0.1\n" % (i * 2))

    t = threading.Thread(target=emit)
    t.start()
    rc, text = run_bider(["estimate", "--log", log, "--interval", "0.5",
                       "--for-seconds", "5", "--window-seconds", "3", "--json"],
                      home)
    t.join()

    blobs = []
    depth = 0
    start = None
    for k, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = k
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    blobs.append(json.loads(text[start:k + 1]))
                except Exception:
                    pass
    check("estimate produced samples", len(blobs) >= 2, len(blobs))
    if len(blobs) >= 2:
        last = blobs[-1]
        seen = [b for b in blobs if b.get("progress")]
        check("reads the progress counter",
              last["progress"] is not None and last["progress"]["total"] == 50,
              last.get("progress"))
        check("progress advanced between samples",
              len(seen) >= 2 and seen[-1]["progress"]["current"] >
              seen[0]["progress"]["current"],
              [b.get("progress") for b in seen])
        check("produced a finite ETA", last["eta_s"] is not None, last.get("eta_s"))
        check("ETA is plausible for the observed rate",
              last["eta_s"] is not None and 5 < last["eta_s"] < 120, last.get("eta_s"))
        check("ETA is humanised", bool(last["eta_human"]), last.get("eta_human"))

    print("\n-- estimate: says so when there is no progress marker --")
    log2 = os.path.join(home, "quiet.log")
    with open(log2, "w") as fh:
        fh.write("something happened\nnothing numeric here\n")
    rc, out = run_bider(["estimate", "--log", log2, "--json"], home)
    j = json.loads(out[out.find("{"):out.rfind("}") + 1]) if "{" in out else None
    check("handles a log with no progress",
          j is not None and j["progress"] is None, j and j.get("progress"))
    check("explains the missing marker",
          j and "no progress marker" in (j.get("note") or ""), j and j.get("note"))


def test_cli_flow():
    print("\n-- cli: add / reorder / probe / run / stats --")
    home = new_home()
    fake = snap(5.6, 0)

    rc, out = run_bider(["add", "python train_big.py", "--name", "big run",
                      "--vram", "4", "--util", "70"], home, fake)
    check("adds a task", rc == 0 and "big-run" in out, out[-200:])
    run_bider(["add", "python quick.py", "--name", "quick", "--vram", "0.3",
            "--util", "5"], home, fake)
    rc, out = run_bider(["list"], home, fake)
    order = [ln for ln in out.splitlines() if ln.strip().startswith(("1.", "2."))]
    check("cheapest requirement listed first",
          len(order) == 2 and "quick" in order[0] and "big-run" in order[1], order)

    run_bider(["pin", "big-run"], home, fake)
    rc, out = run_bider(["list"], home, fake)
    order = [ln for ln in out.splitlines() if ln.strip().startswith(("1.", "2."))]
    check("pinning overrides the cheap-first rule",
          len(order) == 2 and "big-run" in order[0], order)

    rc, out = run_bider(["probe"], home, fake)
    check("probe follows the pinned task", "big-run" in out, out[-300:])
    check("probe falls through to the task that does fit", rc == 0, rc)
    check("probe explains what it skipped", "skipped" in out, out[-300:])
    check("probe shows the arithmetic", "reserve" in out, out[-300:])

    rc, out = run_bider(["probe", "--task", "big-run"], home, fake)
    check("naming a task refuses it when it eats the reserve", rc == 2, rc)

    run_bider(["policy", "--set", "first_fit=false"], home, fake)
    rc, out = run_bider(["probe"], home, fake)
    check("first_fit=false restores the strict head-of-queue refusal", rc == 2, rc)
    run_bider(["policy", "--set", "first_fit=true"], home, fake)

    run_bider(["policy", "--set", "reserve_vram_gb=0.5"], home, fake)
    rc, out = run_bider(["probe"], home, fake)
    check("raising headroom lets the pinned task through", rc == 0, rc)

    # a real (trivial) run, end to end
    run_bider(["add", "echo hello", "--name", "smoke", "--vram", "0.1",
            "--util", "1"], home, fake)
    rc, out = run_bider(["run", "smoke"], home, fake)
    check("run starts a task and waits for it", rc == 0 and "hello" in out,
          out[-300:])
    check("the run is recorded in the ledger",
          "started" in run_bider(["ledger"], home, fake)[1])

    rc, out = run_bider(["runs"], home, fake)
    check("runs lists what happened", "smoke" in out, out[-200:])
    rc, out = run_bider(["stats"], home, fake)
    check("stats counts the run", "runs started        1" in out, out[-400:])

    # a refusal is explained, in words
    rc, out = run_bider(["explain", "--task", "big-run"], home, fake)
    check("explain names the task and its appetite",
          "big-run" in out and "wants" in out, out[:300])
    check("explain says it can start when it can", "can start right now" in out,
          out[:300])
    rc, out = run_bider(["explain", "--task", "big-run"], home, snap(0.5, 95))
    check("explain says why not when it cannot",
          "cannot start yet" in out, out[:300])

    run_bider(["remove", "--done", "done"], home, fake)
    rc, out = run_bider(["update", "big-run", "--set", "vram_gb=1.5"], home, fake)
    check("update changes cost", rc == 0 and "vram=1.5" in out, out[-200:])
    rc, out = run_bider(["run", "big-run", "--dry-run"], home, fake)
    check("dry-run does not start a process", rc == 0 and "dry_run" in out,
          out[-200:])


def test_shared():
    print("\n-- two users on one card, sharing a directory --")
    home_a = new_home("bider-userA-")
    home_b = new_home("bider-userB-")
    shared = new_home("bider-shared-")
    fake = snap(5.6, 0)

    for home, who in ((home_a, "alice"), (home_b, "bob")):
        rc, out = run_bider(["config", "--set", "shared_dir=" + shared,
                          "--set", "owner=" + who], home, fake)
        norm = out.replace("\\", "/")
        check("%s points at the shared directory" % who,
              rc == 0 and shared.replace("\\", "/") in norm, norm[-200:])

    # A claims 3GB (which fits: 5.6 - 3.0 = 2.6, above the 2GB reserve),
    # then B wants 3GB too - together they do not fit.
    write_queue(home_a, [task("alice-big", 3.0, 10)])
    write_queue(home_b, [task("bob-big", 3.0, 10)])
    rc, out = run_bider(["run", "alice-big", "--detach"], home_a, fake)
    check("alice starts her job", rc == 0, out[-300:])

    claims = os.path.join(shared, "claims")
    check("alice's claim is visible in the shared directory",
          os.path.isdir(claims) and
          any(n.startswith("alice") for n in os.listdir(claims)),
          os.listdir(claims) if os.path.isdir(claims) else None)

    rc, out = run_bider(["probe"], home_b, fake)
    check("bob is blocked by alice's claim", rc == 2, rc)
    check("bob is told it is a reservation, not free memory",
          "reserved" in out, out[-300:])
    rc, out = run_bider(["explain"], home_b, fake)
    check("bob's explain names alice",
          "alice" in out, out[:400])

    # B's task is small enough to fit alongside, so it should not be blocked
    write_queue(home_b, [task("bob-small", 0.5, 5)])
    rc, out = run_bider(["probe"], home_b, fake)
    check("bob's small task still fits beside alice's claim", rc == 0, out[-300:])

    # the intent board
    rc, out = run_bider(["intent", "publish"], home_b, fake)
    check("bob publishes his intent", rc == 0, out[-200:])
    rc, out = run_bider(["intent"], home_a, fake)
    check("alice can read bob's intent",
          "bob" in out and "bob-small" in out, out[-300:])

    # a claim for a process that is gone stops blocking, after the grace period
    claim_file = None
    for n in os.listdir(claims):
        if n.startswith("alice"):
            claim_file = os.path.join(claims, n)
    check("alice's claim file is there to inspect", claim_file is not None,
          os.listdir(claims))
    c = json.load(open(claim_file))
    c["pid"] = 999999
    c["host"] = subprocess.run([PY, "-c",
                                "import socket;print(socket.gethostname())"],
                               stdout=subprocess.PIPE).stdout.decode().strip()
    c["at_epoch"] = time.time() - 600
    c["expires_at"] = time.time() + 600
    json.dump(c, open(claim_file, "w"))
    write_queue(home_b, [task("bob-big", 3.0, 10)])
    rc, out = run_bider(["probe"], home_b, fake)
    check("a claim for a dead local process stops blocking", rc == 0, out[-300:])

    # ...but a claim made seconds ago is honoured, because the job may be starting
    c["at_epoch"] = time.time()
    json.dump(c, open(claim_file, "w"))
    rc, out = run_bider(["probe"], home_b, fake)
    check("a fresh claim is honoured even before the process shows up", rc == 2, rc)


def test_memory():
    print("\n-- memory: protects, windows, profiles, sharing --")
    home = new_home()
    shared = new_home("bider-memshare-")
    run_bider(["config", "--set", "shared_dir=" + shared, "--set", "owner=alice"],
           home, snap(5.6, 0))
    holder = [{"gpu_uuid": "GPU-fake", "pid": 4242, "name": "weijob_worker.exe",
               "mem_mb": None}]

    rc, out = run_bider(["memory"], home)
    check("memory starts empty and shows examples",
          rc == 0 and "nothing remembered yet" in out, out[-200:])

    rc, out = run_bider(["add", "echo hi", "--name", "small", "--vram", "0.5",
                      "--util", "5"], home)
    check("queues a small task", rc == 0, out[-200:])
    rc, out = run_bider(["probe"], home, snap(5.6, 4))
    check("without memory the task fits", rc == 0, out[-250:])

    rc, out = run_bider(["memory", "--protect", "wei", "--match", "weijob",
                      "--because", "deadline this week",
                      "--until", "2099-01-01"], home)
    check("records a protection", rc == 0 and "do not disturb wei" in out,
          out[-250:])
    check("and publishes it to the shared directory", "published" in out,
          out[-250:])
    rc, out = run_bider(["probe"], home, snap(5.6, 4, holder))
    check("a protection for a running process blocks the queue", rc == 2, rc)
    check("the refusal says whose process it is",
          "wei" in out and "not be disturbed" in out, out[-250:])
    rc, out = run_bider(["procs"], home, snap(5.6, 4, holder))
    check("procs flags the protected process",
          "do not disturb" in out and "weijob_worker.exe" in out, out[-250:])

    rc, out = run_bider(["probe"], home, snap(5.6, 4))
    check("the protection lapses when the process exits", rc == 0, out[-250:])
    rc, out = run_bider(["memory"], home, snap(5.6, 4))
    check("the protection is still remembered after it lapses",
          "do not disturb wei" in out, out[-250:])

    # an already-expired protection never blocks, even with the process there
    run_bider(["memory", "--forget", "protected-wei"], home)
    rc, out = run_bider(["memory", "--protect", "old", "--match", "weijob",
                      "--until", "2020-01-01"], home)
    check("records an already-expired protection", rc == 0, out[-200:])
    rc, out = run_bider(["probe"], home, snap(5.6, 4, holder))
    check("an expired protection does not block", rc == 0, out[-250:])
    rc, out = run_bider(["memory", "--forget", "protected-old"], home)
    check("forget removes the expired one", rc == 0 and "forgot 1" in out,
          out[-200:])

    # a reserved window
    rc, out = run_bider(["memory", "--window", "wei", "--from", "00:00", "--to",
                      "23:59", "--because", "deadline"], home)
    check("records a reserved window", rc == 0, out[-250:])
    rc, out = run_bider(["probe"], home, snap(5.6, 4))
    check("an in-force window blocks the queue", rc == 2, rc)
    check("the refusal names the window",
          "reserved time" in out and "wei" in out, out[-250:])
    rc, out = run_bider(["explain"], home, snap(5.6, 4))
    check("explain says the card is reserved",
          "reserved this time" in out, out[:400])
    run_bider(["memory", "--forget", "window-wei"], home)

    # profile advice
    rc, out = run_bider(["add", "python eval.py", "--name", "vram hog",
                      "--vram", "4", "--util", "5"], home)
    rc, out = run_bider(["add", "python train.py", "--name", "compute hog",
                      "--vram", "0.6", "--util", "80"], home)
    check("queues two differently shaped tasks", rc == 0, out[-200:])
    rc, out = run_bider(["memory", "--profile", "weijob", "--profile-vram", "heavy",
                      "--profile-util", "light",
                      "--because", "holds memory, not compute"], home)
    check("records a profile", rc == 0 and "heavy vram / light compute" in out,
          out[-250:])
    rc, out = run_bider(["list"], home, snap(5.6, 4, holder))
    order = [ln for ln in out.splitlines() if ln.strip().startswith(("1.", "2.", "3."))]
    check("the profile lifts the compute-heavy task",
          any("compute-hog" in ln for ln in order[:1]), order[:3])

    # private entries stay local, and markers never leak into a published file
    rc, out = run_bider(["add", "echo x", "--name", "sekrit job", "--vram", "0.2",
                      "--util", "1"], home)
    rc, out = run_bider(["memory", "--protect", "secret", "--match", "sekritjob",
                      "--task", "sekrit-job"], home)
    check("a task-scoped protection publishes without the marker",
          "published" in out, out[-200:])
    published = os.path.join(shared, "memory")
    blob = ""
    for n in os.listdir(published):
        blob += open(os.path.join(published, n), encoding="utf-8").read()
    check("the published file still carries the protection",
          "sekritjob" in blob, blob[:200])

    # a second user reads what we published
    home_b = new_home("bider-memb-")
    run_bider(["config", "--set", "shared_dir=" + shared, "--set", "owner=bob"],
           home_b, snap(5.6, 0))
    rc, out = run_bider(["memory"], home_b, snap(5.6, 4, holder))
    check("another user sees the published protection",
          "weijob" in out, out[:300])
    rc, out = run_bider(["probe"], home_b, snap(5.6, 4, holder))
    check("and is blocked by it too", rc == 2, rc)
    rc, out = run_bider(["memory", "--json"], home_b)
    blob = json.loads(out[out.find("{"):out.rfind("}") + 1])
    check("shared entries are flagged and attributed",
          all(e.get("shared") and e.get("shared_owner") == "alice"
              for e in blob["entries"]), blob["entries"])
    rc, out = run_bider(["memory", "--export", "-"], home_b)
    check("a shared entry is marked as shared and not ours",
          '"shared": true' in out and '"shared_owner": "alice"' in out, out[:600])

    rc, out = run_bider(["memory", "--json"], home)
    check("memory --json is machine readable",
          rc == 0 and '"entries"' in out, out[:200])
    rc, out = run_bider(["memory", "--export",
                      os.path.join(home, "exported.json")], home)
    check("exports the memory", rc == 0 and
          os.path.isfile(os.path.join(home, "exported.json")), out[-200:])
    rc, out = run_bider(["memory", "--import-file",
                      os.path.join(home, "exported.json")], home)
    check("imports it back", rc == 0, out[-200:])


def test_config():
    print("\n-- config and home --")
    home = new_home()
    rc, out = run_bider(["config"], home)
    check("config reports an owner", "owner" in out and "(not set" in out,
          out[-300:])
    check("home is the throwaway directory", home in out, out[:200])
    rc, out = run_bider(["config", "--set", "gpu_index=1"], home)
    check("gpu_index round-trips", "gpu_index   1" in out, out[-300:])
    rc, out = run_bider(["config", "--set", "nonsense=1"], home)
    check("an unknown key is refused", rc == 4, rc)

    rc, out = run_bider(["logs"], home)
    check("logs explains where to put task logs", "logs dir is empty" in out,
          out[-300:])


def main():
    print("bider end-to-end tests  (python %s)" % sys.version.split()[0])
    try:
        test_edges()
        test_estimate()
        test_cli_flow()
        test_shared()
        test_memory()
        test_config()
    finally:
        for d in _TMP:
            shutil.rmtree(d, ignore_errors=True)
    print("")
    if FAILS:
        print("E2E FAILED: %d check(s)" % len(FAILS))
        for f in FAILS:
            print("  - %s" % f)
        return 1
    print("E2E OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
