#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Install gpu-bider.

Run this once after cloning:

    python install.py            # set up this machine (recommended)
    python install.py --check    # diagnostics only, change nothing
    python install.py --skill    # also copy the Claude Code skill, in place
    python install.py --user     # ... into ~/.claude/skills instead

What it does:

  1. checks Python, nvidia-smi, and whether the card also drives a display;
  2. runs `bider selftest` so you know the logic works before trusting it;
  3. seeds ~/.bider with a queue and a config (the sample memory entries are
     disabled and teach the format without changing anything);
  4. with --skill, copies skill/gpu-bider/ into .claude/skills/ so Claude Code
     picks it up.

If `pip install .` was used, steps 1-3 are already done by the entry point and
this script is only needed for the skill.  It writes nothing outside ~/.bider and
the project directory.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GQ = os.path.join(HERE, "bider.py")
SKILL_SRC = os.path.join(HERE, "skill", "gpu-bider")
SKILL_NAME = "gpu-bider"

MIN_PY = (3, 8)


def c(s, code=None):
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return s
    codes = {"ok": "32", "warn": "33", "bad": "31", "dim": "2", "b": "1"}
    if code:
        return "\033[%sm%s\033[0m" % (codes.get(code, "0"), s)
    return s


def line(label, status, detail=""):
    marks = {"ok": c("ok", "ok"), "warn": c("!!", "warn"), "bad": c("XX", "bad"),
             "info": c("--", "dim")}
    print("  %-4s %-34s %s" % (marks.get(status, status), label, detail))


def _is_store_stub(path):
    return "windowsapps" in (path or "").lower()


def bider_home():
    return os.path.abspath(os.environ.get("BIDER_HOME")
                           or os.path.join(os.path.expanduser("~"), ".bider"))


def check_python():
    ok = True
    v = sys.version_info
    if (v.major, v.minor) >= MIN_PY:
        line("python", "ok", "%d.%d.%d at %s" % (v.major, v.minor, v.micro,
                                                 sys.executable))
    else:
        line("python", "bad", "need >= %d.%d, found %d.%d"
             % (MIN_PY + (v.major, v.minor)))
        ok = False
    if _is_store_stub(sys.executable):
        line("interpreter sanity", "bad",
             "that is the Microsoft Store stub; run this with a real Python "
             "(e.g. C:\\anaconda\\python.exe)")
        ok = False
    return ok


def find_smi():
    p = shutil.which("nvidia-smi")
    if p:
        return p
    for cand in (os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                              "System32", "nvidia-smi.exe"),
                 r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        if os.path.isfile(cand):
            return cand
    return None


def check_nvidia_smi():
    p = find_smi()
    if not p:
        line("nvidia-smi", "bad", "not found -- the queue cannot see the GPU")
        return False
    try:
        out = subprocess.check_output(
            [p, "--query-gpu=name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader"], stderr=subprocess.DEVNULL, timeout=15)
        line("nvidia-smi", "ok",
             out.decode("utf-8", "replace").strip().splitlines()[0])
    except Exception as e:
        line("nvidia-smi", "warn", "found at %s but querying failed: %s" % (p, e))
    return True


def check_display():
    p = find_smi()
    if not p:
        return
    try:
        out = subprocess.check_output(
            [p, "--query-gpu=count,display_active", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=15)
        row = out.decode("utf-8", "replace").strip().splitlines()[0].split(",")
        count = row[0].strip()
        active = row[1].strip().lower() if len(row) > 1 else "?"
        if active == "enabled":
            line("display shares the gpu", "warn",
                 "%s card(s); the default 2GB/25%% reserve protects it" % count)
        else:
            line("display shares the gpu", "ok", "%s card(s), display inactive"
                 % count)
    except Exception:
        pass


def check_skill_source():
    md = os.path.join(SKILL_SRC, "SKILL.md")
    if not os.path.isfile(md):
        line("skill source", "bad", "missing %s" % md)
        return False
    with open(md, "r", encoding="utf-8") as fh:
        head = fh.read(400)
    if not head.lstrip().startswith("---") or "name:" not in head:
        line("skill source", "warn", "SKILL.md has no frontmatter")
        return False
    line("skill source", "ok", os.path.relpath(SKILL_SRC, HERE))
    return True


def check_selftest():
    if not os.path.isfile(GQ):
        line("selftest", "bad", "bider.py missing")
        return False
    p = subprocess.run([sys.executable, GQ, "selftest"], cwd=HERE,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode == 0:
        line("selftest", "ok", "scheduling, claims and memory logic verified")
        return True
    line("selftest", "bad", "see `bider selftest`")
    for t in p.stdout.decode("utf-8", "replace").strip().splitlines()[-4:]:
        print("        %s" % t)
    return False


def install_skill(target_root, label):
    dest = os.path.join(target_root, SKILL_NAME)
    if os.path.abspath(dest) == os.path.abspath(SKILL_SRC):
        line("skill -> %s" % label, "info", "source is already in place")
        return True
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copytree(SKILL_SRC, dest)
    n = sum(len(f) for _, _, f in os.walk(dest))
    line("skill -> %s" % label, "ok", "%s (%d files)" % (dest, n))
    return True


def write_wrappers():
    """bider / bider.cmd for people who did not pip install."""
    exe = sys.executable
    cmd_path = os.path.join(HERE, "bider.cmd")
    with open(cmd_path, "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write("@echo off\r\n")
        fh.write('"%s" "%s" %%*\r\n' % (exe, GQ))
    line("wrapper bider.cmd", "ok", "Windows / cmd / Claude Code")

    sh_path = os.path.join(HERE, "bider")
    with open(sh_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("#!/bin/sh\n")
        fh.write("# generated by install.py - calls bider.py with a known-good python\n")
        fh.write('exec "%s" "%s" "$@"\n'
                 % (exe.replace("\\", "/"), GQ.replace("\\", "/")))
    try:
        os.chmod(sh_path, 0o755)
    except OSError:
        pass
    line("wrapper bider", "ok", "bash / git-bash")


SAMPLE_MEMORY = {"version": 1, "entries": [
    {"kind": "window", "id": "window-example", "person": "example",
     "name": "example", "disabled": True,
     "from": "14:00", "to": "18:00", "days": "mon-fri",
     "because": "EXAMPLE - a recurring block reserved for one person. "
                "disabled=true, so it does nothing; drop that key to arm it."},
    {"kind": "protected", "id": "protected-example", "person": "example",
     "name": "example", "disabled": True,
     "match": {"name": "their_job"},
     "because": "EXAMPLE - never disturb this process. Applies only while it "
                "is actually on the card. Replace it with your own: "
                "bider memory --protect <name> --match <process>"},
    {"kind": "profile", "id": "profile-example", "family": "example",
     "name": "example", "disabled": True,
     "shape": {"vram": "heavy", "util": "light"},
     "because": "EXAMPLE - this family holds memory but little compute, so "
                "put compute-heavy tasks next to it."},
], "advice": []}

DEFAULT_QUEUE = {"version": 1, "policy": {
    "reserve_vram_gb": 2.0, "reserve_util_pct": 25, "poll_seconds": 20,
    "poll_floor_seconds": 3, "poll_ceiling_seconds": 120, "settle_ticks": 2,
    "allow_concurrent": False, "first_fit": True, "require_claim": True,
    "claim_ttl_s": 120, "polite": True}, "tasks": []}


def seed(memory_mode, force=False):
    home = bider_home()
    os.makedirs(home, exist_ok=True)
    line("home", "ok", home)

    qp = os.path.join(home, "queue.json")
    if os.path.isfile(qp) and not force:
        line("queue.json", "info", "kept (already exists)")
    else:
        with open(qp, "w", encoding="utf-8") as fh:
            json.dump(DEFAULT_QUEUE, fh, indent=2)
            fh.write("\n")
        line("queue.json", "ok", "created, 2GB / 25% held back for the desktop")

    mp = os.path.join(home, "memory.json")
    if os.path.isfile(mp) and not force:
        line("memory.json", "info", "kept (already exists)")
        return home
    mem = {"version": 1, "entries": [], "advice": []} \
        if memory_mode == "empty" else SAMPLE_MEMORY
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(mem, fh, indent=2)
        fh.write("\n")
    line("memory.json", "ok",
         "created empty" if memory_mode == "empty"
         else "created with 3 disabled example entries")
    return home


def main():
    ap = argparse.ArgumentParser(description="Install gpu-bider.")
    ap.add_argument("--check", action="store_true",
                    help="run diagnostics only, change nothing")
    ap.add_argument("--skill", action="store_true",
                    help="copy the skill into ./.claude/skills")
    ap.add_argument("--user", action="store_true",
                    help="copy the skill into ~/.claude/skills instead")
    ap.add_argument("--also-user", action="store_true",
                    help="copy the skill to both places")
    ap.add_argument("--no-wrapper", action="store_true",
                    help="skip the bider / bider.cmd wrappers")
    ap.add_argument("--memory", choices=["examples", "empty"],
                    default="examples",
                    help="seed memory.json with disabled examples (default) "
                         "or empty")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing queue.json / memory.json")
    args = ap.parse_args()

    print(c("gpu-bider installer", "b"))
    print("  project: %s" % HERE)
    print("")
    print(c("diagnostics", "b"))

    ok = check_python()
    smi = check_nvidia_smi()
    check_display()
    have_skill = check_skill_source()
    ok = check_selftest() and ok

    if args.check:
        print("")
        print(c("check only -- nothing was changed.", "dim"))
        return 0 if (ok and smi) else 1

    print("")
    print(c("installing", "b"))
    home = seed(args.memory, args.force)
    if not args.no_wrapper:
        write_wrappers()

    if have_skill and (args.skill or args.user or args.also_user):
        if args.user or args.also_user:
            install_skill(os.path.join(os.path.expanduser("~"), ".claude",
                                       "skills"),
                          "~/.claude/skills  (all projects)")
        if not args.user:
            install_skill(os.path.join(HERE, ".claude", "skills"),
                          ".claude/skills  (this project)")
    elif have_skill:
        print(c("  skill source is here but not installed; pass --skill or "
                "--user to copy it", "dim"))

    print("")
    print(c("next steps", "b"))
    print("  bider status                     # gpu + queue + what it remembers")
    print("  bider selftest                   # the logic, without touching the gpu")
    print('  bider add "python train.py" --name first --vram 2 --util 40')
    print("  bider watch --stay-alive         # wait for the card (Ctrl-C to stop)")
    print("")
    print("  To share the card with colleagues, point every bider at one")
    print("  directory they can all see:")
    print("      bider config --set shared_dir=/path/we/all/share")
    print("      bider config --set owner=yourname")
    print("")
    if args.skill or args.user or args.also_user:
        print("  The skill loads when Claude Code starts in this directory.")
        print("  If you installed it mid-session, restart Claude Code.")
    if not ok or not smi:
        print("")
        print(c("Some diagnostics failed -- see above.", "warn"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
