#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pwd
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_IGNORED_COMM = (
    "systemd",
    "(sd-pam)",
    "tmux: server",
    "tmux",
    "bash",
    "sh",
    "zsh",
    "fish",
    "login",
    "sshd",
)


@dataclass
class ProcessInfo:
    pid: int
    tty: str
    elapsed_seconds: int
    comm: str


@dataclass
class GpuProcessInfo:
    pid: int
    process_name: str
    used_memory_mb: int | None


@dataclass
class BusySample:
    process_matches: list[ProcessInfo]
    gpu_matches: list[GpuProcessInfo]

    @property
    def busy(self) -> bool:
        return bool(self.process_matches or self.gpu_matches)

    def summary(self) -> str:
        parts: list[str] = []
        if self.process_matches:
            processes = ", ".join(
                f"pid={info.pid}:{info.comm}@{info.tty}" for info in self.process_matches[:5]
            )
            suffix = "" if len(self.process_matches) <= 5 else f" ... +{len(self.process_matches) - 5}"
            parts.append(f"terminal-processes [{processes}{suffix}]")
        if self.gpu_matches:
            gpu_processes = ", ".join(
                f"pid={info.pid}:{info.process_name}:{info.used_memory_mb or '?'}MB"
                for info in self.gpu_matches[:5]
            )
            suffix = "" if len(self.gpu_matches) <= 5 else f" ... +{len(self.gpu_matches) - 5}"
            parts.append(f"gpu-processes [{gpu_processes}{suffix}]")
        return "; ".join(parts) if parts else "idle"


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_stamp()}] {message}", file=sys.stderr, flush=True)


def normalize_command(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        return command[1:]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a command only while another user's terminal/GPU activity is idle. "
            "When that user becomes busy, the child command is terminated and restarted later."
        )
    )
    parser.add_argument("--watch-user", required=True, help="Username to monitor, for example caochunshui.")
    parser.add_argument(
        "--mode",
        choices=("terminal", "gpu", "any", "all-processes"),
        default="any",
        help=(
            "terminal: only non-shell TTY processes; gpu: only GPU compute processes; "
            "any: terminal or GPU activity; all-processes: any non-ignored process or GPU activity."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds. Default: 5.",
    )
    parser.add_argument(
        "--busy-hold-seconds",
        type=float,
        default=10.0,
        help="How long activity must persist before the child is stopped. Default: 10.",
    )
    parser.add_argument(
        "--idle-hold-seconds",
        type=float,
        default=30.0,
        help="How long the watched user must stay idle before the child restarts. Default: 30.",
    )
    parser.add_argument(
        "--term-grace-seconds",
        type=float,
        default=20.0,
        help="How long to wait after SIGTERM before SIGKILL. Default: 20.",
    )
    parser.add_argument(
        "--ignore-comm",
        action="append",
        default=[],
        help="Process names to ignore. Can be passed more than once.",
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        help="Optional file that stores the currently running child PID.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run. Put -- before the command to stop option parsing.",
    )
    args = parser.parse_args()
    args.command = normalize_command(args.command)
    if not args.command:
        parser.error("missing command to supervise")
    return args


def user_exists(username: str) -> bool:
    try:
        pwd.getpwnam(username)
    except KeyError:
        return False
    return True


def write_pid_file(path: Path | None, pid: int | None) -> None:
    if path is None:
        return
    if pid is None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n", encoding="utf-8")


def load_processes(username: str) -> list[ProcessInfo]:
    completed = subprocess.run(
        ["ps", "-u", username, "-o", "pid=", "-o", "tty=", "-o", "etimes=", "-o", "comm="],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []

    processes: list[ProcessInfo] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 3)
        if len(parts) != 4:
            continue
        pid_text, tty, elapsed_text, comm = parts
        try:
            processes.append(
                ProcessInfo(
                    pid=int(pid_text),
                    tty=tty,
                    elapsed_seconds=int(elapsed_text),
                    comm=comm.strip(),
                )
            )
        except ValueError:
            continue
    return processes


def pid_owner(pid: int) -> str | None:
    proc_path = Path("/proc") / str(pid)
    try:
        uid = proc_path.stat().st_uid
    except FileNotFoundError:
        return None
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return None


def load_gpu_processes(username: str) -> list[GpuProcessInfo]:
    if not shutil_which("nvidia-smi"):
        return []

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []

    gpu_processes: list[GpuProcessInfo] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid_owner(pid) != username:
            continue
        used_memory_mb: int | None = None
        if len(parts) >= 3 and parts[2]:
            try:
                used_memory_mb = int(parts[2])
            except ValueError:
                used_memory_mb = None
        gpu_processes.append(
            GpuProcessInfo(
                pid=pid,
                process_name=parts[1],
                used_memory_mb=used_memory_mb,
            )
        )
    return gpu_processes


def comm_is_ignored(comm: str, ignored_comm: set[str]) -> bool:
    for ignored in ignored_comm:
        if comm == ignored:
            return True
        if comm.startswith(f"{ignored}:"):
            return True
    return False


def match_busy_processes(
    username: str,
    mode: str,
    ignored_comm: set[str],
) -> BusySample:
    processes = load_processes(username)
    gpu_matches: list[GpuProcessInfo] = []
    process_matches: list[ProcessInfo] = []

    if mode in {"gpu", "any", "all-processes"}:
        gpu_matches = load_gpu_processes(username)

    if mode in {"terminal", "any", "all-processes"}:
        for info in processes:
            if comm_is_ignored(info.comm, ignored_comm):
                continue
            if mode in {"terminal", "any"} and info.tty == "?":
                continue
            process_matches.append(info)

    return BusySample(process_matches=process_matches, gpu_matches=gpu_matches)


def shutil_which(program: str) -> str | None:
    for path_part in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(path_part) / program
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def start_child(command: list[str], pid_file: Path | None) -> subprocess.Popen[str]:
    log(f"starting child: {shlex.join(command)}")
    child = subprocess.Popen(
        command,
        start_new_session=True,
        text=True,
    )
    write_pid_file(pid_file, child.pid)
    log(f"child pid={child.pid}")
    return child


def terminate_process_group(child: subprocess.Popen[str], grace_seconds: float) -> None:
    try:
        pgid = os.getpgid(child.pid)
    except ProcessLookupError:
        return

    log(f"stopping child process group pgid={pgid} with SIGTERM")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if child.poll() is not None:
            return
        time.sleep(0.5)

    if child.poll() is None:
        log(f"child still alive after {grace_seconds:.1f}s, sending SIGKILL to pgid={pgid}")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return


def reap_child(child: subprocess.Popen[str], timeout: float = 1.0) -> None:
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def main() -> int:
    args = parse_args()
    if not user_exists(args.watch_user):
        raise SystemExit(f"user not found: {args.watch_user}")

    ignored_comm = set(DEFAULT_IGNORED_COMM)
    ignored_comm.update(args.ignore_comm)

    child: subprocess.Popen[str] | None = None
    busy_since: float | None = None
    idle_since: float | None = None
    last_busy_summary = ""
    shutting_down = False
    wait_for_idle_window = False

    def handle_signal(signum: int, _frame) -> None:
        nonlocal shutting_down, child
        shutting_down = True
        log(f"received signal {signum}, shutting down")
        if child is not None:
            terminate_process_group(child, args.term_grace_seconds)
            reap_child(child)
            child = None
            write_pid_file(args.pid_file, None)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    log(
        "watching user "
        f"{args.watch_user} in mode={args.mode}, busy_hold={args.busy_hold_seconds:.1f}s, "
        f"idle_hold={args.idle_hold_seconds:.1f}s"
    )

    while not shutting_down:
        sample = match_busy_processes(args.watch_user, args.mode, ignored_comm)
        now = time.monotonic()

        if sample.busy:
            wait_for_idle_window = True
            idle_since = None
            if busy_since is None:
                busy_since = now
                last_busy_summary = sample.summary()
                log(f"{args.watch_user} became busy: {last_busy_summary}")
            elif sample.summary() != last_busy_summary:
                last_busy_summary = sample.summary()
                log(f"{args.watch_user} busy details changed: {last_busy_summary}")

            if child is not None and now - busy_since >= args.busy_hold_seconds:
                terminate_process_group(child, args.term_grace_seconds)
                reap_child(child)
                child = None
                write_pid_file(args.pid_file, None)
        else:
            busy_since = None
            last_busy_summary = ""

            if child is None:
                if not wait_for_idle_window:
                    child = start_child(args.command, args.pid_file)
                else:
                    if idle_since is None:
                        idle_since = now
                        log(f"{args.watch_user} is idle")
                    if now - idle_since >= args.idle_hold_seconds:
                        child = start_child(args.command, args.pid_file)
                        wait_for_idle_window = False
                        idle_since = None
            else:
                idle_since = None
                wait_for_idle_window = False

        if child is not None:
            returncode = child.poll()
            if returncode is not None:
                write_pid_file(args.pid_file, None)
                if sample.busy:
                    log(f"child exited with code {returncode} while {args.watch_user} was busy; waiting to restart")
                    child = None
                    idle_since = None
                    wait_for_idle_window = True
                else:
                    log(f"child exited with code {returncode}")
                    return returncode

        time.sleep(args.poll_interval)

    return 128 + signal.SIGTERM


if __name__ == "__main__":
    raise SystemExit(main())
