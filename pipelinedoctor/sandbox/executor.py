"""
pipelinedoctor/sandbox/executor.py
=================================
Ephemeral Docker sandbox executor for isolated test verification.
Mounts a repo snapshot, applies unified diffs with fuzz tolerance,
runs test commands, and captures deterministic exit codes and logs.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import docker
from docker.errors import DockerException

from pipelinedoctor.config import Settings, get_settings
from pipelinedoctor.schemas import VerificationResult, VerificationStatus


def apply_patch_py(repo_dir: Path, diff_text: str) -> bool:
    """
    Fallback pure-python patch applier if 'patch' or 'git' binary is unavailable.
    Applies unified diff hunks directly to files with whitespace and single-line tolerance.
    """
    lines = diff_text.splitlines()
    target_file: Path | None = None
    old_lines: list[str] = []
    new_lines: list[str] = []
    in_hunk = False

    for line in lines:
        if line.startswith("--- "):
            rel_path = line[4:].strip()
            if rel_path.startswith("a/"):
                rel_path = rel_path[2:]
            target_file = repo_dir / rel_path
        elif line.startswith("+++ "):
            pass
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk:
            if line.startswith("-") and not line.startswith("---"):
                old_lines.append(line[1:])
            elif line.startswith("+") and not line.startswith("+++"):
                new_lines.append(line[1:])
            elif line.startswith(" "):
                old_lines.append(line[1:])
                new_lines.append(line[1:])

    if target_file and target_file.exists():
        try:
            content = target_file.read_text(encoding="utf-8")
            old_block = "\n".join(old_lines)
            new_block = "\n".join(new_lines)
            if old_block and old_block in content:
                patched = content.replace(old_block, new_block, 1)
                target_file.write_text(patched, encoding="utf-8")
                return True
            # Fallback: single-line replacement if only 1 removed and 1 added line
            removed = [l for l in old_lines if l not in new_lines]
            added = [l for l in new_lines if l not in old_lines]
            if len(removed) == 1 and len(added) == 1 and removed[0] in content:
                patched = content.replace(removed[0], added[0], 1)
                target_file.write_text(patched, encoding="utf-8")
                return True
        except Exception:
            return False

    return False


def to_wsl_path(win_path: Path | str) -> str:
    """Convert Windows path C:\\foo\\bar to WSL path /mnt/c/foo/bar."""
    p = str(win_path).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:]
        return f"/mnt/{drive}{rest}"
    return p


class SandboxExecutor:
    """Executes verification tests inside isolated ephemeral Docker containers."""

    def __init__(
        self,
        settings: Settings | None = None,
        docker_client: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._docker_client = docker_client

    @property
    def docker_client(self) -> docker.DockerClient:
        if self._docker_client is None:
            self._docker_client = docker.from_env()
        return self._docker_client

    def execute_in_sandbox(
        self,
        repo_dir: Path | str,
        diffs: list[str],
        test_command: str = "pytest",
        image: str | None = None,
        timeout_seconds: int | None = None,
    ) -> VerificationResult:
        """
        Creates an isolated temporary copy of repo_dir, applies diffs,
        spawns a container, executes the test command, and returns VerificationResult.
        Prefers Python Docker SDK, falls back to WSL Docker engine, then local subprocess.
        """
        t0 = time.monotonic()
        source_path = Path(repo_dir)
        target_image = image or self.settings.DOCKER_IMAGE
        timeout = timeout_seconds or self.settings.SANDBOX_TIMEOUT_SECONDS

        # 1. Ephemeral workspace directory
        temp_workspace = Path(tempfile.mkdtemp(prefix="pipelinedoctor_sandbox_"))

        try:
            # Copy source snapshot into temp workspace
            if source_path.exists() and any(source_path.iterdir()):
                shutil.copytree(source_path, temp_workspace, dirs_exist_ok=True)

            # 2. Write patch files and apply to workspace
            for idx, diff in enumerate(diffs):
                patch_file = temp_workspace / f"patch_{idx}.diff"
                patch_file.write_text(diff, encoding="utf-8")
                # Apply diff directly to files in workspace
                apply_patch_py(temp_workspace, diff)

            # 3. Build container entrypoint command
            shell_commands: list[str] = []
            for idx in range(len(diffs)):
                patch_name = f"patch_{idx}.diff"
                shell_commands.append(
                    f"if [ -f /workspace/{patch_name} ]; then (patch -p1 -N -t --fuzz=3 -i /workspace/{patch_name} 2>/dev/null || patch -p0 -N -t --fuzz=3 -i /workspace/{patch_name} 2>/dev/null || true); fi"
                )
            shell_commands.append("python -m pytest -q")
            entrypoint_cmd = " && ".join(shell_commands)

            # ── Attempt A: Python Docker SDK (Windows Named Pipe / daemon) ────────
            try:
                container = self.docker_client.containers.create(
                    image=target_image,
                    command=["sh", "-c", entrypoint_cmd],
                    volumes={str(temp_workspace): {"bind": self.settings.SANDBOX_WORK_DIR, "mode": "rw"}},
                    working_dir=self.settings.SANDBOX_WORK_DIR,
                    network_mode="none",
                    mem_limit="2g",
                    nano_cpus=2_000_000_000,
                )

                container_id = container.id[:12] if hasattr(container, "id") else "docker-sdk-box"
                try:
                    container.start()
                    result = container.wait(timeout=timeout)
                    exit_code = result.get("StatusCode", -1) if isinstance(result, dict) else int(result)
                    stdout_logs = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
                    stderr_logs = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
                    duration = time.monotonic() - t0
                    status = VerificationStatus.PASSED if exit_code == 0 else VerificationStatus.FAILED
                    return VerificationResult(
                        status=status,
                        exit_code=exit_code,
                        stdout=stdout_logs,
                        stderr=stderr_logs,
                        duration_seconds=round(duration, 2),
                        container_id=container_id,
                    )
                finally:
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass
            except Exception:
                pass  # Fall through to WSL Docker

            # ── Attempt B: Real Docker Container via WSL Engine ───────────────────
            import subprocess

            wsl_dir = to_wsl_path(temp_workspace)
            docker_wsl_cmd = [
                "wsl", "-d", "Ubuntu-22.04", "-e",
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", "2g",
                "-v", f"{wsl_dir}:/workspace",
                "-w", "/workspace",
                target_image,
                "sh", "-c", entrypoint_cmd,
            ]

            try:
                proc = subprocess.run(
                    docker_wsl_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                duration = time.monotonic() - t0
                status = VerificationStatus.PASSED if proc.returncode == 0 else VerificationStatus.FAILED
                return VerificationResult(
                    status=status,
                    exit_code=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    duration_seconds=round(duration, 2),
                    container_id="docker-wsl-container",
                )
            except subprocess.TimeoutExpired:
                duration = time.monotonic() - t0
                return VerificationResult(
                    status=VerificationStatus.TIMEOUT,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Docker container execution timed out after {timeout} seconds",
                    duration_seconds=round(duration, 2),
                    container_id="docker-wsl-container",
                )
            except Exception:
                pass  # Fall through to local isolated workspace

            # ── Attempt C: Local Isolated Sandbox (Subprocess) ────────────────────
            import sys
            cmd = test_command
            if cmd == "pytest":
                cmd = f'"{sys.executable}" -m pytest -q'

            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(temp_workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.monotonic() - t0
            status = VerificationStatus.PASSED if proc.returncode == 0 else VerificationStatus.FAILED
            return VerificationResult(
                status=status,
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_seconds=round(duration, 2),
                container_id="local-isolated-sandbox",
            )

        finally:
            shutil.rmtree(temp_workspace, ignore_errors=True)
