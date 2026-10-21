# Release checklist

The things that must be true before a public release. Kept in the repo so a
later change cannot quietly break one of them.

## Non-negotiables

- [ ] `bider selftest` passes on Linux, macOS and Windows (CI does this).
- [ ] `python test_e2e.py` passes on all three (CI does this).
- [ ] Neither suite touches a real GPU. Every test sets `BIDER_FAKE_GPU`.
- [ ] Nothing in the codebase sends data anywhere. No telemetry, no phone-home,
      no network calls except the user's own `notify_cmd`.
- [ ] `bider` works with no Claude Code, no API key and no account.
- [ ] The tool never sends a signal to, or otherwise touches, a process it did
      not itself start. Grep for `kill_tree` and `taskkill` and confirm every
      call site is on a pid from our own `meta.json`.
- [ ] `kill_tree` is called in exactly one place: stopping *our own* run on
      `--timeout-min`, and stopping *our own* watcher on `watch --stop`.

## Before tagging

- [ ] `VERSION` in `bider.py` and `version` in `pyproject.toml` agree.
- [ ] `README.md` limitations section still matches the code. In particular:
      the Windows `[N/A]` note, the NFS claim caveat, and the "cannot
      preempt" limit.
- [ ] `skill/gpu-bider/SKILL.md` mentions every user-facing command. Run
      `bider --help` and diff the subcommand list against the skill by eye.
- [ ] `python install.py --check` reports honestly on a machine with no GPU
      (CI checks this, and that it exits 1 rather than crashing).
- [ ] Fresh-clone smoke test:

      git clone <repo> /tmp/bider-fresh && cd /tmp/bider-fresh
      python install.py --memory empty
      bider selftest
      python test_e2e.py

- [ ] `pip install .` in a clean venv gives a working `bider` entry point.
- [ ] `python -m build` produces a wheel with just `bider.py` in it, and no
      runtime state.

## Deliberately not in scope

Recording these so they are decisions, not oversights:

- **Preemption.** Out of scope by design, permanently. bider never stops another
  process. See the README's "honest limit".
- **A server.** Sharing works through a directory. No daemon to run, no port to
  open, nothing to be granted access to.
- **MIG / MPS.** Real partitioning is a different tool. bider manages time, not
  hardware slices.
- **Multi-tenant fair-share scheduling.** If a group needs quotas and rotation,
  they want Slurm. bider's niche is the small shared box that has no admin.
- **Static analysis to predict VRAM.** Unreliable. Estimates come from the
  user, and are corrected from what the card actually reports afterwards.

## Known rough edges

- `estimate` cannot project another person's job without a readable log.
  `~/.bider/logs/` exists because a symlink there is the one thing a colleague
  can do to make this work.
- `observations.json` grows monotonically; the bounded families are: shape
  samples are merged rather than appended, and the list is trimmed to 500.
- Protected-process matching is name-based, so a rename defeats it. It fails
  open (does not block), which is the safe direction for a tool that must never
  wedge a card.
