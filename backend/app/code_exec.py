"""Runs tests/lint inside an indexed repository.

This is real arbitrary code execution against a cloned third-party repository —
that's inherently risky regardless of how it's wrapped, so it's off by default
(ENABLE_CODE_EXECUTION=false) and, even when enabled, only allows a fixed
allowlist of read-mostly commands (pytest, a linter, etc.) run with a timeout,
no shell interpolation, and no network/filesystem access beyond the repo
directory. This is a starting point for a sandboxed tool, not a hardened one —
for real deployments this belongs in an isolated container/VM, not the API
process.
"""
import shlex
import subprocess
from dataclasses import dataclass
from typing import List

from . import config

ALLOWED_COMMANDS = {
    "pytest": ["python3", "-m", "pytest", "-q"],
    "ruff": ["python3", "-m", "ruff", "check", "."],
    "flake8": ["python3", "-m", "flake8", "."],
    "npm_test": ["npm", "test", "--silent"],
    "go_test": ["go", "test", "./..."],
}


@dataclass
class ExecResult:
    command: str
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


class CodeExecutionDisabled(Exception):
    pass


class UnknownCommand(Exception):
    pass


def run(repo_dir: str, command_key: str, extra_args: List[str] = None, timeout: int = 60) -> ExecResult:
    if not config.ENABLE_CODE_EXECUTION:
        raise CodeExecutionDisabled(
            "Code execution is disabled by default. Set ENABLE_CODE_EXECUTION=true "
            "to enable it, and only do so for repositories you trust — this runs "
            "code from the cloned repo."
        )
    if command_key not in ALLOWED_COMMANDS:
        raise UnknownCommand(f"'{command_key}' is not in the allowlist: {list(ALLOWED_COMMANDS)}")

    cmd = list(ALLOWED_COMMANDS[command_key])
    if extra_args:
        # still allowlisted-command-only; extra args are appended, not shell-interpreted
        cmd += [shlex.quote(a) for a in extra_args]

    try:
        result = subprocess.run(
            cmd, cwd=repo_dir, capture_output=True, text=True, timeout=timeout,
        )
        return ExecResult(
            command=" ".join(cmd), stdout=result.stdout[-5000:], stderr=result.stderr[-5000:],
            returncode=result.returncode, timed_out=False,
        )
    except subprocess.TimeoutExpired as e:
        return ExecResult(
            command=" ".join(cmd), stdout=(e.stdout or "")[-5000:], stderr=(e.stderr or "")[-5000:],
            returncode=-1, timed_out=True,
        )
    except FileNotFoundError as e:
        return ExecResult(command=" ".join(cmd), stdout="", stderr=str(e), returncode=-1, timed_out=False)
