#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from supervise_command_on_user_busy import DEFAULT_IGNORED_COMM, match_busy_processes, user_exists


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_stamp()}] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait until a watched user stays idle for a required duration, "
            "then exec supervise_command_on_user_busy.py with the provided arguments."
        )
    )
    parser.add_argument("--watch-user", required=True)
    parser.add_argument(
        "--mode",
        choices=("terminal", "gpu", "any", "all-processes"),
        default="any",
    )
    parser.add_argument("--initial-idle-seconds", type=float, default=1800.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--busy-hold-seconds", type=float, default=10.0)
    parser.add_argument("--idle-hold-seconds", type=float, default=1800.0)
    parser.add_argument("--term-grace-seconds", type=float, default=20.0)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command to supervise")
    return args


def main() -> int:
    args = parse_args()
    if not user_exists(args.watch_user):
        raise SystemExit(f"user not found: {args.watch_user}")

    ignored_comm = set(DEFAULT_IGNORED_COMM)
    shutting_down = False

    def handle_signal(signum: int, _frame) -> None:
        nonlocal shutting_down
        shutting_down = True
        log(f"received signal {signum}, shutting down before supervisor launch")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    idle_since: float | None = None
    last_state = ""
    log(
        f"waiting for user {args.watch_user} to stay idle for "
        f"{args.initial_idle_seconds:.1f}s before launching supervised job"
    )

    while not shutting_down:
        sample = match_busy_processes(args.watch_user, args.mode, ignored_comm)
        now = time.monotonic()
        if sample.busy:
            idle_since = None
            summary = sample.summary()
            if summary != last_state:
                log(f"{args.watch_user} busy: {summary}")
                last_state = summary
        else:
            if idle_since is None:
                idle_since = now
                log(f"{args.watch_user} is idle, starting initial idle timer")
            else:
                elapsed = now - idle_since
                if elapsed >= args.initial_idle_seconds:
                    break
                rounded_elapsed = int(elapsed)
                state = f"idle:{rounded_elapsed}"
                if state != last_state and rounded_elapsed % 60 == 0:
                    remaining = max(0, int(args.initial_idle_seconds - elapsed))
                    log(f"{args.watch_user} idle for {rounded_elapsed}s, remaining {remaining}s before launch")
                    last_state = state
        time.sleep(args.poll_interval)

    if shutting_down:
        return 1

    supervise_script = PROJECT_ROOT / "supervise_command_on_user_busy.py"
    command = [
        sys.executable,
        str(supervise_script),
        "--watch-user",
        args.watch_user,
        "--mode",
        args.mode,
        "--poll-interval",
        str(args.poll_interval),
        "--busy-hold-seconds",
        str(args.busy_hold_seconds),
        "--idle-hold-seconds",
        str(args.idle_hold_seconds),
        "--term-grace-seconds",
        str(args.term_grace_seconds),
    ]
    if args.pid_file is not None:
        command.extend(["--pid-file", str(args.pid_file)])
    command.append("--")
    command.extend(args.command)
    log(f"initial idle requirement satisfied, launching supervisor: {' '.join(command)}")
    os_exec = __import__("os").execv
    os_exec(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
