"""Worker class for threadward package.

This version fixes stdout-handling stalls by replacing every
`select + readline` pattern with a single non-blocking byte-reader that
yields complete text lines already waiting on the pipe.
"""

from __future__ import annotations

import errno
import fcntl
import os
import select
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

import psutil
import shutil

from .task import Task

try:
    import GPUtil
except ImportError:           # GPU monitoring is optional
    GPUtil = None


class Worker:
    """Represents a subprocess worker that executes tasks."""

    # ──────────────────────────  INIT & UTILITIES  ──────────────────────────

    def __init__(
        self,
        worker_id: int,
        gpu_ids: List[int] | None = None,
        conda_env: str | None = None,
        debug: bool = False,
    ):
        self.worker_id = worker_id
        self.gpu_ids = gpu_ids or []
        self.conda_env = conda_env
        self.debug = debug

        self.process: subprocess.Popen | None = None
        self.current_task: Task | None = None
        self.status: str = "idle"        # idle | busy | shutting_down | stopped
        self.start_time: float | None = None

        self.total_tasks_succeeded = 0
        self.total_tasks_failed = 0
        self.output_buffer: List[str] = []      # early results from other tasks

        # Hierarchical state
        self.current_hierarchical_key = ""
        self.current_hierarchical_values: Dict[str, Any] = {}
        self.hierarchical_load_count = 0

        # Resource tracking
        self.max_cpu_percent = 0.0
        self.current_cpu_percent = 0.0
        self.max_memory_mb = 0.0
        self.current_memory_mb = 0.0
        self.max_gpu_memory_mb = 0.0
        self.current_gpu_memory_mb = 0.0

        # Monitoring thread
        self._monitoring_thread: threading.Thread | None = None
        self._stop_monitoring = threading.Event()

        # Internal tail of an incomplete stdout line
        self._stdout_tail: bytes = b""

    def _debug_print(self, msg: str) -> None:
        if self.debug:
            ts = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{ts}] {msg}", flush=True)

    @staticmethod
    def _get_python_executable() -> str:
        if shutil.which("python"):
            return "python"
        if shutil.which("python3"):
            return "python3"
        import sys

        return sys.executable

    # ──────────────────────────  NON-BLOCKING READ  ─────────────────────────

    def _iter_stdout_nonblocking(self):
        """
        Yield every complete line currently available on the worker's stdout
        without blocking. Works on POSIX systems.
        """
        if self.process is None or self.process.stdout is None:
            return

        fd = self.process.stdout.fileno()

        # Ensure O_NONBLOCK is set once.
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        if not (fl & os.O_NONBLOCK):
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        # Any new bytes waiting?
        try:
            if not select.select([fd], [], [], 0)[0]:
                return
        except (OSError, ValueError):
            return

        try:
            data = os.read(fd, 4096)
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return
            raise

        if not data:
            return

        data = self._stdout_tail + data
        *lines, self._stdout_tail = data.split(b"\n")

        for bline in lines:
            yield bline.decode(errors="replace").strip()

    # ──────────────────────────  START & SHUTDOWN  ──────────────────────────

    def start(self, config_file_path: str, results_path: str, task_timeout: float = 30) -> bool:
        self.task_timeout = task_timeout
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids)) if self.gpu_ids else ""
        env["PYTHONUNBUFFERED"] = "1"
        
        # Debug print CUDA_VISIBLE_DEVICES assignment
        self._debug_print(f"Worker {self.worker_id} CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

        worker_entry = f"""
import sys
from threadward.core.worker_process import worker_main_from_file
worker_main_from_file({self.worker_id!r}, {config_file_path!r}, {results_path!r})
"""

        import sys as _sys

        self.process = subprocess.Popen(
            [_sys.executable, "-c", worker_entry],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=False,          # binary mode – we decode ourselves
            bufsize=0,
        )

        if self.process.poll() is not None:
            print(f"ERROR: Worker {self.worker_id} terminated immediately")
            print(self.process.stderr.read().decode())
            return False

        self.status = "idle"
        self.start_time = time.time()
        self._start_monitoring()
        return True

    def shutdown(self) -> None:
        if not self.process:
            return
        self.status = "shutting_down"
        try:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.write(b"SHUT_DOWN\n")
                self.process.stdin.flush()
                self.process.stdin.close()
            self.process.wait(10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        finally:
            self.status = "stopped"
            self._stop_monitoring.set()
            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=1)

    # ──────────────────────────  TASK ASSIGNMENT  ───────────────────────────

    def assign_task(self, task: Task) -> bool:
        if self.status != "idle" or not self.process:
            return False
        if self.process.poll() is not None:
            return False

        # Update hierarchical state
        if self.update_hierarchical_state(task):
            self._debug_print(f"Worker {self.worker_id} hierarchical state changed to: {task.hierarchical_key}")

        # Send task id
        try:
            assert self.process.stdin
            self.process.stdin.write(f"{task.task_id}\n".encode())
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            return False

        # Wait for TASK_RECEIVED
        ack = False
        while not ack:
            if self.process.poll() is not None:
                return False
            for line in self._iter_stdout_nonblocking():
                if line == "TASK_RECEIVED":
                    ack = True
                    self._debug_print(f"Worker {self.worker_id} acknowledged task {task.task_id}")
                elif (
                    ":" in line
                    and line.split(":", 1)[1] in ("TASK_SUCCESS_RESPONSE", "TASK_FAILURE_RESPONSE")
                ):
                    self.output_buffer.append(line)
                    self._debug_print(f"Worker {self.worker_id} buffered task result: {line}")
                elif line.startswith(("WORKER_DEBUG:", "DEBUG:")):
                    dbg = line.replace("WORKER_DEBUG:", "").replace("DEBUG:", "")
                    self._debug_print(f"[Worker {self.worker_id}] {dbg}")
                elif line and line != "WORKER_READY":
                    self._debug_print(f"Worker {self.worker_id} output: {line}")
            if not ack:
                time.sleep(0.05)

        self.current_task = task
        self.status = "busy"
        task.status = "running"
        task.worker_id = self.worker_id
        task.start_time = time.time()
        return True

    # ──────────────────────────  TASK COMPLETION  ───────────────────────────

    def check_task_completion(self) -> Optional[bool]:
        if self.status != "busy" or not self.process or not self.current_task:
            return None

        # 1. result file shortcut
        task_id = self.current_task.task_id
        result_file = os.path.join(self.current_task.task_folder, f"{task_id}_result.txt")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                content = f.read().strip()
            if ":" in content:
                _, rtype = content.split(":", 1)
                success = rtype == "TASK_SUCCESS_RESPONSE"
                os.remove(result_file)
                self._finish_task(success)
                return success

        # 2. buffered early lines from other loops
        for i, line in enumerate(list(self.output_buffer)):
            if line.startswith(f"{task_id}:"):
                self.output_buffer.pop(i)
                success = line.endswith("TASK_SUCCESS_RESPONSE")
                self._finish_task(success)
                return success

        # 3. new lines available right now
        for line in self._iter_stdout_nonblocking():
            if ":" in line and line.startswith(f"{task_id}:"):
                success = line.endswith("TASK_SUCCESS_RESPONSE")
                self._finish_task(success)
                return success
            # else handle debug or other results
            if (
                ":" in line
                and line.split(":", 1)[1] in ("TASK_SUCCESS_RESPONSE", "TASK_FAILURE_RESPONSE")
            ):
                self.output_buffer.append(line)
            elif line.startswith(("WORKER_DEBUG:", "DEBUG:")):
                dbg = line.replace("WORKER_DEBUG:", "").replace("DEBUG:", "")
                self._debug_print(f"[Worker {self.worker_id}] {dbg}")
            elif line and line != "WORKER_READY":
                self._debug_print(f"Worker {self.worker_id} output: {line}")

        # 4. timeout check
        runtime = time.time() - self.current_task.start_time
        if self.task_timeout != -1 and runtime > self.task_timeout:
            self._debug_print(f"Worker {self.worker_id} task {task_id} timed out after {runtime:.1f}s")
            self._finish_task(False)
            return False

        return None

    def _finish_task(self, success: bool) -> None:
        self.current_task.status = "completed" if success else "failed"
        self.current_task.end_time = time.time()
        if success:
            self.total_tasks_succeeded += 1
        else:
            self.total_tasks_failed += 1
        self.status = "idle"

    # ──────────────────────────  MONITORING  ────────────────────────────────

    def _start_monitoring(self) -> None:
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitoring_thread.start()

    def _monitor_resources(self) -> None:
        while not self._stop_monitoring.is_set() and self.process:
            if self.process.poll() is not None:
                break
            try:
                proc = psutil.Process(self.process.pid)
                self.current_cpu_percent = proc.cpu_percent()
                self.max_cpu_percent = max(self.max_cpu_percent, self.current_cpu_percent)

                mem = proc.memory_info().rss / (1024 * 1024)
                self.current_memory_mb = mem
                self.max_memory_mb = max(self.max_memory_mb, mem)

                if GPUtil and self.gpu_ids:
                    total = 0
                    for gid in self.gpu_ids:
                        gpus = GPUtil.getGPUs()
                        if gid < len(gpus):
                            total += gpus[gid].memoryUsed
                    self.current_gpu_memory_mb = total
                    self.max_gpu_memory_mb = max(self.max_gpu_memory_mb, total)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(1)

    # ──────────────────────────  MISC PROPS  ───────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "pid": self.process.pid if self.process else None,
            "status": self.status,
            "gpu_ids": self.gpu_ids,
            "current_task": str(self.current_task) if self.current_task else None,
            "total_tasks_succeeded": self.total_tasks_succeeded,
            "total_tasks_failed": self.total_tasks_failed,
            "cpu_percent": {"current": self.current_cpu_percent, "max": self.max_cpu_percent},
            "memory_mb": {"current": self.current_memory_mb, "max": self.max_memory_mb},
            "gpu_memory_mb": {"current": self.current_gpu_memory_mb, "max": self.max_gpu_memory_mb},
        }

    # basic dunders
    def __str__(self):  # noqa: D401
        return f"Worker({self.worker_id}, {self.status})"

    def __repr__(self):
        return f"Worker(id={self.worker_id}, status={self.status}, gpus={self.gpu_ids})"

    # ──────────────────────────  HIERARCHY  ────────────────────────────────

    def is_hierarchically_compatible(self, task: Task) -> bool:
        return (not task.hierarchical_key) or (task.hierarchical_key == self.current_hierarchical_key)

    def update_hierarchical_state(self, task: Task) -> bool:
        if task.hierarchical_key != self.current_hierarchical_key:
            self.current_hierarchical_key = task.hierarchical_key
            self.current_hierarchical_values = task.get_hierarchical_values()
            self.hierarchical_load_count += 1
            return True
        return False
