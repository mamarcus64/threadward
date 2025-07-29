"""Worker class for threadward package."""

import subprocess
import os
import time
import psutil
import threading
import pickle
from typing import Optional, List, Dict, Any
from .task import Task
import shutil

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
        self.total_tasks_succeeded = 0
        self.total_tasks_failed = 0
        
        # Hierarchical state tracking
        self.current_hierarchical_key: str = ""
        self.current_hierarchical_values: Dict[str, Any] = {}
        self.hierarchical_load_count: int = 0
        
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
    
    @staticmethod
    def _get_python_executable() -> str:
        """Get the Python executable, preferring 'python' but falling back to 'python3'."""
        if shutil.which("python") is not None:
            return "python"
        elif shutil.which("python3") is not None:
            return "python3"
        else:
            # As a last resort, use sys.executable
            import sys
            return sys.executable
    
    def start(self, config_file_path: str, results_path: str) -> bool:
        """Start the worker subprocess.
        
        Args:
            config_file_path: Path to the configuration file
            results_path: Path to the results directory
            
        Returns:
            True if worker started successfully, False otherwise
        """
        try:
            # Prepare environment variables
            env = os.environ.copy()
            
            # Set CUDA_VISIBLE_DEVICES if GPUs are assigned
            if self.gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            
            # Create worker entry script that imports and runs the worker process
            worker_entry = f'''
import sys
from threadward.core.worker_process import worker_main_from_file

# Worker parameters
worker_id = {self.worker_id}
config_file_path = {repr(config_file_path)}
results_path = {repr(results_path)}

# Run the worker
worker_main_from_file(worker_id, config_file_path, results_path)
'''
            
            # Prepare command to run the worker entry code
            python_executable = self._get_python_executable()
            
            if self.conda_env:
                # Use conda environment
                cmd = [
                    "conda", "run", "-n", self.conda_env,
                    python_executable, "-c", worker_entry
                ]
            else:
                # Use current Python environment
                cmd = [python_executable, "-c", worker_entry]
            
            # Start the subprocess
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=0,  # Unbuffered for immediate output
                universal_newlines=True
            )
            
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
        if self.status != "idle" or self.process is None:
            return False
        
        # Check if process is still alive
        if self.process.poll() is not None:
            print(f"ERROR: Worker {self.worker_id} process has terminated (return code: {self.process.returncode})")
            stderr_output = self.process.stderr.read()
            print(f"ERROR: Worker {self.worker_id} stderr: {stderr_output}")
            return False
        
        try:
<<<<<<< HEAD
            # Update hierarchical state if needed
            state_changed = self.update_hierarchical_state(task)
            if state_changed:
                print(f"DEBUG: Worker {self.worker_id} hierarchical state changed to: {task.hierarchical_key}")
            
            print(f"DEBUG: Sending task ID '{task.task_id}' to worker {self.worker_id}")
            
=======
>>>>>>> b4da4bfd29a091af18fd4f2429364fc07840e1f2
            # Send task ID to worker via stdin
            self.process.stdin.write(f"{task.task_id}\n")
            self.process.stdin.flush()
            
            self.current_task = task
            self.status = "busy"
            task.status = "running"
            task.worker_id = self.worker_id
            task.start_time = time.time()
            
            return True
            
        except Exception as e:
            print(f"Failed to assign task to worker {self.worker_id}: {e}")
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
            # Don't clear current_task yet - the caller needs it
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
                        self.total_tasks_succeeded += 1
                    else:
                        self.total_tasks_failed += 1
                    
                    # Don't clear current_task here - let the main loop do it
                    self.status = "idle"
                    
                    return success
        except (ImportError, OSError) as e:
            # Windows fallback using threading with timeout
            try:
                import threading
                import queue
                
                def read_line_with_timeout(process, timeout=0.1):
                    """Read a line from process stdout with timeout."""
                    result_queue = queue.Queue()
                    
                    def reader():
                        try:
                            line = process.stdout.readline()
                            if line:
                                result_queue.put(line.strip())
                        except:
                            pass
                    
                    thread = threading.Thread(target=reader, daemon=True)
                    thread.start()
                    thread.join(timeout)
                    
                    try:
                        return result_queue.get_nowait()
                    except queue.Empty:
                        return None
                
                # Try to read a line with a short timeout
                result_line = read_line_with_timeout(self.process, 0.1)
                if result_line and result_line in ["SUCCESS", "FAILURE"]:
                    success = result_line == "SUCCESS"
                    
                    self.current_task.status = "completed" if success else "failed"
                    self.current_task.end_time = time.time()
                    
                    if success:
                        self.total_tasks_succeeded += 1
                    else:
                        self.total_tasks_failed += 1
                    
                    # Don't clear current_task here - let the main loop do it
                    self.status = "idle"
                    
                    return success
                        
            except Exception as e:
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
            "total_tasks_succeeded": self.total_tasks_succeeded,
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
    
    def is_hierarchically_compatible(self, task: Task) -> bool:
        """Check if a task is compatible with the worker's current hierarchical state.
        
        Args:
            task: Task to check compatibility for
            
        Returns:
            True if the task matches the worker's hierarchical state or if no hierarchy is defined
        """
        if not task.hierarchical_key:
            return True  # No hierarchy defined, all tasks are compatible
        
        return task.hierarchical_key == self.current_hierarchical_key
    
    def update_hierarchical_state(self, task: Task) -> bool:
        """Update the worker's hierarchical state based on a task.
        
        Args:
            task: Task to update state from
            
        Returns:
            True if hierarchical state changed, False otherwise
        """
        if task.hierarchical_key != self.current_hierarchical_key:
            self.current_hierarchical_key = task.hierarchical_key
            self.current_hierarchical_values = task.get_hierarchical_values()
            self.hierarchical_load_count += 1
            return True
        return False