"""Worker class for threadward package."""

import subprocess
import os
import time
import psutil
import threading
from typing import Optional, List, Dict, Any
from .task import Task

try:
    import GPUtil
except ImportError:
    GPUtil = None


class Worker:
    """Represents a subprocess worker that executes tasks."""
    
    def __init__(self, worker_id: int, gpu_ids: List[int] = None, 
                 conda_env: Optional[str] = None):
        """Initialize a worker.
        
        Args:
            worker_id: Unique identifier for this worker
            gpu_ids: List of GPU IDs assigned to this worker
            conda_env: Name of conda environment to use
        """
        self.worker_id = worker_id
        self.gpu_ids = gpu_ids or []
        self.conda_env = conda_env
        self.process: Optional[subprocess.Popen] = None
        self.current_task: Optional[Task] = None
        self.status = "idle"  # idle, busy, shutting_down, stopped
        self.start_time: Optional[float] = None
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        
        # Resource monitoring
        self.max_cpu_percent = 0.0
        self.current_cpu_percent = 0.0
        self.max_memory_mb = 0.0
        self.current_memory_mb = 0.0
        self.max_gpu_memory_mb = 0.0
        self.current_gpu_memory_mb = 0.0
        
        # Monitoring thread
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
    
    def start(self, worker_script_path: str) -> bool:
        """Start the worker subprocess.
        
        Args:
            worker_script_path: Path to the worker script to execute
            
        Returns:
            True if worker started successfully, False otherwise
        """
        try:
            print(f"DEBUG: Starting worker {self.worker_id} with script: {worker_script_path}")
            
            # Prepare environment variables
            env = os.environ.copy()
            
            # Set CUDA_VISIBLE_DEVICES if GPUs are assigned
            if self.gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            
            # Prepare command
            if self.conda_env:
                # Use conda environment
                cmd = [
                    "conda", "run", "-n", self.conda_env,
                    "python", worker_script_path, str(self.worker_id)
                ]
            else:
                # Use current Python environment
                cmd = ["python", worker_script_path, str(self.worker_id)]
            
            print(f"DEBUG: Worker {self.worker_id} command: {cmd}")
            
            # Start the subprocess
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1
            )
            
            print(f"DEBUG: Worker {self.worker_id} subprocess started with PID: {self.process.pid}")
            
            # Check if process started successfully
            if self.process.poll() is not None:
                print(f"ERROR: Worker {self.worker_id} process terminated immediately")
                stderr_output = self.process.stderr.read()
                print(f"ERROR: Worker {self.worker_id} stderr: {stderr_output}")
                return False
            
            self.status = "idle"
            self.start_time = time.time()
            
            # Start monitoring thread
            self._start_monitoring()
            
            print(f"DEBUG: Worker {self.worker_id} started successfully")
            return True
            
        except Exception as e:
            print(f"Failed to start worker {self.worker_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def assign_task(self, task: Task) -> bool:
        """Assign a task to this worker.
        
        Args:
            task: Task to assign
            
        Returns:
            True if task was assigned successfully, False otherwise
        """
        print(f"DEBUG: Attempting to assign task {task.task_id} to worker {self.worker_id}")
        print(f"DEBUG: Worker {self.worker_id} status: {self.status}, process: {self.process is not None}")
        
        if self.status != "idle" or self.process is None:
            print(f"DEBUG: Worker {self.worker_id} not ready - status: {self.status}, process exists: {self.process is not None}")
            return False
        
        # Check if process is still alive
        if self.process.poll() is not None:
            print(f"ERROR: Worker {self.worker_id} process has terminated (return code: {self.process.returncode})")
            stderr_output = self.process.stderr.read()
            print(f"ERROR: Worker {self.worker_id} stderr: {stderr_output}")
            return False
        
        try:
            print(f"DEBUG: Sending task ID '{task.task_id}' to worker {self.worker_id}")
            
            # Send task ID to worker via stdin
            self.process.stdin.write(f"{task.task_id}\n")
            self.process.stdin.flush()
            
            print(f"DEBUG: Task {task.task_id} sent successfully to worker {self.worker_id}")
            
            self.current_task = task
            self.status = "busy"
            task.status = "running"
            task.worker_id = self.worker_id
            task.start_time = time.time()
            
            return True
            
        except Exception as e:
            print(f"Failed to assign task to worker {self.worker_id}: {e}")
            print(f"DEBUG: Process state - poll: {self.process.poll()}, stdin: {self.process.stdin}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_task_completion(self) -> Optional[bool]:
        """Check if the current task has completed.
        
        Returns:
            True if task succeeded, False if failed, None if still running
        """
        if self.status != "busy" or self.process is None or self.current_task is None:
            return None
        
        # Check if process has output available
        if self.process.poll() is not None:
            # Process has terminated - this shouldn't happen during normal task execution
            self.status = "idle"
            self.current_task.status = "failed"
            self.current_task.end_time = time.time()
            self.total_tasks_failed += 1
            self.current_task = None
            return False
        
        # Check for task completion signal (worker should send result via stdout)
        try:
            # Try to use select for non-blocking read (Unix/Linux/macOS)
            import select
            if select.select([self.process.stdout], [], [], 0)[0]:
                result_line = self.process.stdout.readline().strip()
                if result_line:
                    success = result_line == "SUCCESS"
                    
                    self.current_task.status = "completed" if success else "failed"
                    self.current_task.end_time = time.time()
                    
                    if success:
                        self.total_tasks_completed += 1
                    else:
                        self.total_tasks_failed += 1
                    
                    self.current_task = None
                    self.status = "idle"
                    
                    return success
        except (ImportError, OSError):
            # On Windows or systems without select, use a simpler approach
            # Check if process has new output by attempting a quick peek
            try:
                # Use subprocess's built-in poll mechanism
                if hasattr(self.process.stdout, 'peek'):
                    # Try to peek at the buffer without consuming it
                    peeked = self.process.stdout.peek(1)
                    if peeked:
                        result_line = self.process.stdout.readline().strip()
                        if result_line:
                            success = result_line == "SUCCESS"
                            
                            self.current_task.status = "completed" if success else "failed"
                            self.current_task.end_time = time.time()
                            
                            if success:
                                self.total_tasks_completed += 1
                            else:
                                self.total_tasks_failed += 1
                            
                            self.current_task = None
                            self.status = "idle"
                            
                            return success
            except:
                # Fallback: task completion will be detected when process terminates
                pass
        
        return None
    
    def shutdown(self) -> None:
        """Gracefully shutdown the worker."""
        if self.process is None:
            return
        
        self.status = "shutting_down"
        
        try:
            # Send shutdown signal
            self.process.stdin.write("SHUT_DOWN\n")
            self.process.stdin.flush()
            
            # Wait for process to terminate
            self.process.wait(timeout=10)
            
        except subprocess.TimeoutExpired:
            # Force terminate if it doesn't shut down gracefully
            self.process.terminate()
            self.process.wait(timeout=5)
            
        except:
            # Force kill if terminate fails
            self.process.kill()
            
        finally:
            self.status = "stopped"
            self._stop_monitoring.set()
            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=1)
    
    def _start_monitoring(self) -> None:
        """Start the resource monitoring thread."""
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitor_resources)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
    
    def _monitor_resources(self) -> None:
        """Monitor worker resource usage."""
        while not self._stop_monitoring.is_set() and self.process is not None:
            try:
                if self.process.poll() is None:  # Process is still running
                    # Get process info
                    proc = psutil.Process(self.process.pid)
                    
                    # CPU and Memory
                    self.current_cpu_percent = proc.cpu_percent()
                    self.max_cpu_percent = max(self.max_cpu_percent, self.current_cpu_percent)
                    
                    memory_info = proc.memory_info()
                    self.current_memory_mb = memory_info.rss / 1024 / 1024
                    self.max_memory_mb = max(self.max_memory_mb, self.current_memory_mb)
                    
                    # GPU Memory (if GPUtil is available and GPUs are assigned)
                    if GPUtil and self.gpu_ids:
                        try:
                            gpus = GPUtil.getGPUs()
                            total_gpu_memory = 0
                            for gpu_id in self.gpu_ids:
                                if gpu_id < len(gpus):
                                    gpu = gpus[gpu_id]
                                    total_gpu_memory += gpu.memoryUsed
                            
                            self.current_gpu_memory_mb = total_gpu_memory
                            self.max_gpu_memory_mb = max(self.max_gpu_memory_mb, total_gpu_memory)
                        except:
                            pass
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except Exception:
                pass
            
            time.sleep(1)  # Monitor every second
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "pid": self.process.pid if self.process else None,
            "status": self.status,
            "gpu_ids": self.gpu_ids,
            "current_task": str(self.current_task) if self.current_task else None,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "cpu_percent": {
                "current": self.current_cpu_percent,
                "max": self.max_cpu_percent
            },
            "memory_mb": {
                "current": self.current_memory_mb,
                "max": self.max_memory_mb
            },
            "gpu_memory_mb": {
                "current": self.current_gpu_memory_mb,
                "max": self.max_gpu_memory_mb
            }
        }
    
    def __str__(self) -> str:
        """String representation of the worker."""
        return f"Worker({self.worker_id}, {self.status})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the worker."""
        return f"Worker(id={self.worker_id}, status={self.status}, gpus={self.gpu_ids})"