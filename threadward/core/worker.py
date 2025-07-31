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
                 conda_env: Optional[str] = None, debug: bool = False):
        """Initialize a worker.
        
        Args:
            worker_id: Unique identifier for this worker
            gpu_ids: List of GPU IDs assigned to this worker
            conda_env: Name of conda environment to use
            debug: Enable debug output (default: False)
        """
        self.worker_id = worker_id
        self.gpu_ids = gpu_ids or []
        self.conda_env = conda_env
        self.debug = debug
        self.process: Optional[subprocess.Popen] = None
        self.current_task: Optional[Task] = None
        self.status = "idle"  # idle, busy, shutting_down, stopped
        self.start_time: Optional[float] = None
        self.total_tasks_succeeded = 0
        self.total_tasks_failed = 0
        self.output_buffer = []  # Buffer for output lines read while waiting for acknowledgment
        self.pending_result = None  # Result from a previous task that arrived late
        
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
    
    def _debug_print(self, message: str):
        """Print debug message if debug mode is enabled."""
        if self.debug:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{timestamp}] {message}", flush=True)
    
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
    
    def start(self, config_file_path: str, results_path: str, task_timeout: float = 30) -> bool:
        """Start the worker subprocess.
        
        Args:
            config_file_path: Path to the configuration file
            results_path: Path to the results directory
            task_timeout: Timeout in seconds for task completion (-1 for no timeout)
            
        Returns:
            True if worker started successfully, False otherwise
        """
        self.task_timeout = task_timeout
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
            
            # Use the same Python executable as the parent process
            # This inherits the exact same environment (conda, virtualenv, etc.)
            import sys
            cmd = [sys.executable, "-c", worker_entry]
            self._debug_print(f"Worker {self.worker_id} using parent python executable: {sys.executable}")
            
            self._debug_print(f"Worker {self.worker_id} conda_env: {self.conda_env}")
            self._debug_print(f"Worker {self.worker_id} python_executable: {python_executable}")
            
            # Start the subprocess with proper unbuffering
            # Set PYTHONUNBUFFERED to ensure Python doesn't buffer output
            env["PYTHONUNBUFFERED"] = "1"
            
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
            
            # Give worker a brief moment to initialize, then start sending tasks
            self._debug_print(f"Worker {self.worker_id} waiting 1 second for basic initialization")
            time.sleep(1)
            
            # Check if process is still alive after initialization period
            if self.process.poll() is not None:
                print(f"ERROR: Worker {self.worker_id} process terminated during initialization")
                self._debug_print(f"Worker {self.worker_id} return code: {self.process.returncode}")
                stderr_output = self.process.stderr.read()
                stdout_output = self.process.stdout.read()
                print(f"ERROR: Worker {self.worker_id} stderr: {stderr_output}")
                self._debug_print(f"Worker {self.worker_id} stdout: {stdout_output}")
                if self.conda_env:
                    self._debug_print(f"Conda environment '{self.conda_env}' may not exist or be accessible")
                    self._debug_print(f"Try running: conda info --envs")
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
            self._debug_print(f"Worker {self.worker_id} process pid: {self.process.pid}")
            self._debug_print(f"Worker {self.worker_id} current task: {self.current_task.task_id if self.current_task else 'None'}")
            stderr_output = self.process.stderr.read()
            print(f"ERROR: Worker {self.worker_id} stderr: {stderr_output}")
            # Try to read any remaining stdout as well
            try:
                stdout_output = self.process.stdout.read()
                if stdout_output:
                    self._debug_print(f"Worker {self.worker_id} final stdout: {stdout_output}")
            except:
                pass
            return False
        
        try:
            # Update hierarchical state if needed
            state_changed = self.update_hierarchical_state(task)
            if state_changed:
                self._debug_print(f"Worker {self.worker_id} hierarchical state changed to: {task.hierarchical_key}")
            
            self._debug_print(f"Sending task ID '{task.task_id}' to worker {self.worker_id}")
            
            # Send task ID to worker via stdin
            if self.process.stdin and not self.process.stdin.closed:
                try:
                    self.process.stdin.write(f"{task.task_id}\n")
                    self.process.stdin.flush()
                except BrokenPipeError as e:
                    print(f"ERROR: Worker {self.worker_id} broken pipe when writing task ID: {e}")
                    self._debug_print(f"Worker {self.worker_id} process poll: {self.process.poll()}")
                    self._debug_print(f"Worker {self.worker_id} process pid: {self.process.pid}")
                    # Check if process is still alive
                    if self.process.poll() is not None:
                        self._debug_print(f"Worker {self.worker_id} process has terminated with return code: {self.process.poll()}")
                        # Try to read any remaining stderr
                        try:
                            stderr_output = self.process.stderr.read()
                            if stderr_output:
                                self._debug_print(f"Worker {self.worker_id} final stderr: {stderr_output}")
                        except:
                            pass
                    return False
                except Exception as e:
                    print(f"ERROR: Worker {self.worker_id} unexpected error writing to stdin: {e}")
                    return False
            else:
                print(f"ERROR: Worker {self.worker_id} stdin is closed or None")
                self._debug_print(f"Worker {self.worker_id} stdin state: {self.process.stdin}")
                self._debug_print(f"Worker {self.worker_id} process poll: {self.process.poll()}")
                return False
            
            # Wait for acknowledgment from worker that task was received
            ack_received = False
            start_time = time.time()
            last_warning_time = start_time
            warning_interval = 20  # Start with 20 second intervals
            warning_count = 0
            
            while not ack_received:
                if self.process.poll() is not None:
                    # Process died
                    return False
                
                # Try to read acknowledgment
                import select
                if hasattr(select, 'select'):
                    ready_to_read, _, _ = select.select([self.process.stdout], [], [], 0.1)
                    if ready_to_read:
                        try:
                            line = self.process.stdout.readline().strip()
                            if line == "TASK_RECEIVED":
                                ack_received = True
                                self._debug_print(f"Worker {self.worker_id} acknowledged task {task.task_id}")
                            elif line:
                                # Check if it's a task result with ID
                                if ":" in line and line.split(":", 1)[1] in ["TASK_SUCCESS_RESPONSE", "TASK_FAILURE_RESPONSE"]:
                                    self.output_buffer.append(line)
                                    self._debug_print(f"Worker {self.worker_id} buffered task result: {line}")
                                elif line.startswith("WORKER_DEBUG:") or line.startswith("DEBUG:"):
                                    # Handle debug messages from worker - print directly to main console
                                    debug_msg = line.replace("WORKER_DEBUG:", '').replace("DEBUG:", '')
                                    self._debug_print(f"[Worker {self.worker_id}] {debug_msg}")
                                elif "DEBUG:" not in line and line != "WORKER_READY":
                                    # Log non-debug output for debugging
                                    self._debug_print(f"Worker {self.worker_id} output: {line}")
                        except Exception as e:
                            self._debug_print(f"Worker {self.worker_id} error reading stdout: {e}")
                
                time.sleep(0.1)
            
            # ack_received is guaranteed to be True when we exit the while loop
            
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
        
        self._debug_print(f"Worker {self.worker_id} checking completion for task {self.current_task.task_id}, buffer size: {len(self.output_buffer)}")
        if self.output_buffer:
            self._debug_print(f"Worker {self.worker_id} buffer contents: {self.output_buffer}")
        
        # Check if process has output available
        if self.process.poll() is not None:
            # Process has terminated - this shouldn't happen during normal task execution
            self.status = "idle"
            self.current_task.status = "failed"
            self.current_task.end_time = time.time()
            self.total_tasks_failed += 1
            # Don't clear current_task yet - the caller needs it
            return False
        
        # First check if there's a result file for the current task
        task_id = self.current_task.task_id
        result_file = os.path.join(self.current_task.task_folder, f"{task_id}_result.txt")
        
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    result_content = f.read().strip()
                
                if ":" in result_content:
                    result_task_id, result_type = result_content.split(":", 1)
                    if result_task_id == task_id and result_type in ["TASK_SUCCESS_RESPONSE", "TASK_FAILURE_RESPONSE"]:
                        success = result_type == "TASK_SUCCESS_RESPONSE"
                        print(f"Worker {self.worker_id} found result file for {task_id}: {result_type}")
                        
                        # Remove the result file after reading
                        try:
                            os.remove(result_file)
                            self._debug_print(f"DEBUG: Deleted result file {result_file} for task {task_id}")
                        except Exception as e:
                            self._debug_print(f"DEBUG: Failed to delete result file {result_file}: {e}")
                        
                        self.current_task.status = "completed" if success else "failed"
                        self.current_task.end_time = time.time()
                        
                        if success:
                            self.total_tasks_succeeded += 1
                        else:
                            self.total_tasks_failed += 1
                        
                        self.status = "idle"
                        return success
            except Exception as e:
                self._debug_print(f"Worker {self.worker_id} error reading result file: {e}")
        
        # Then check buffered results (keep as fallback)
        for i, line in enumerate(self.output_buffer):
            if ":" in line:
                result_task_id, result_type = line.split(":", 1)
                if result_task_id == task_id and result_type in ["TASK_SUCCESS_RESPONSE", "TASK_FAILURE_RESPONSE"]:
                    # Found the result for current task
                    self.output_buffer.pop(i)
                    success = result_type == "TASK_SUCCESS_RESPONSE"
                    self._debug_print(f"Worker {self.worker_id} found buffered result for {task_id}: {result_type}")
                    
                    self.current_task.status = "completed" if success else "failed"
                    self.current_task.end_time = time.time()
                    
                    if success:
                        self.total_tasks_succeeded += 1
                    else:
                        self.total_tasks_failed += 1
                    
                    self.status = "idle"
                    return success
        
        # Check for task completion signal (worker should send result via stdout)
        try:
            # Try to use select for non-blocking read (Unix/Linux/macOS)
            import select
            # Keep checking until we reach the task timeout
            task_runtime = time.time() - self.current_task.start_time if self.current_task.start_time else 0
            remaining_timeout = (self.task_timeout - task_runtime) if self.task_timeout != -1 else 30
            
            # If we've already exceeded timeout, use 0 (non-blocking check)
            if remaining_timeout <= 0:
                check_timeout = 0
            else:
                # Use a short check interval but keep trying until timeout
                check_timeout = min(0.5, remaining_timeout)
            
            if select.select([self.process.stdout], [], [], check_timeout)[0]:
                # Read all available lines to avoid blocking
                lines_read = []
                try:
                    while select.select([self.process.stdout], [], [], 0)[0]:
                        line = self.process.stdout.readline().strip()
                        if line:
                            lines_read.append(line)
                        else:
                            break
                except:
                    pass
                
                # Process all the lines we read
                for result_line in lines_read:
                    if result_line:
                        # Check if it's a task result
                        if ":" in result_line and (":TASK_SUCCESS_RESPONSE" in result_line or ":TASK_FAILURE_RESPONSE" in result_line):
                            result_task_id, result_type = result_line.split(":", 1)
                            if result_task_id == task_id:
                                success = result_type == "TASK_SUCCESS_RESPONSE"
                                self._debug_print(f"Worker {self.worker_id} found immediate result for {task_id}: {result_type}")
                                
                                self.current_task.status = "completed" if success else "failed"
                                self.current_task.end_time = time.time()
                                
                                if success:
                                    self.total_tasks_succeeded += 1
                                else:
                                    self.total_tasks_failed += 1
                                
                                # Don't clear current_task here - let the main loop do it
                                self.status = "idle"
                                
                                return success
                            else:
                                # Result for a different task - buffer it
                                self.output_buffer.append(result_line)
                                self._debug_print(f"Worker {self.worker_id} buffered result for different task: {result_line}")
                        elif result_line.startswith("WORKER_DEBUG:"):
                            # Handle debug messages from worker - print directly to main console
                            debug_msg = result_line[13:]  # Remove "WORKER_DEBUG:" prefix
                            print(f"[Worker {self.worker_id}] {debug_msg}")
                        elif "DEBUG:" not in result_line:
                            # Non-debug output that's not a result
                            self._debug_print(f"Worker {self.worker_id} output: {result_line}")
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
                # Use configured timeout (-1 means no timeout)
                timeout = None if self.task_timeout == -1 else self.task_timeout
                result_line = read_line_with_timeout(self.process, timeout)
                if result_line and result_line in ["TASK_SUCCESS_RESPONSE", "TASK_FAILURE_RESPONSE"]:
                    success = result_line == "TASK_SUCCESS_RESPONSE"
                    
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
        
        # Check if we've exceeded the task timeout
        task_runtime = time.time() - self.current_task.start_time if self.current_task.start_time else 0
        if self.task_timeout != -1 and task_runtime > self.task_timeout:
            # Task timed out - do one final check for results with a short timeout
            self._debug_print(f"Worker {self.worker_id} task {self.current_task.task_id} checking for late results after {task_runtime:.1f}s")
            
            # One last attempt to read the result
            import select
            if select.select([self.process.stdout], [], [], 0.5)[0]:
                result_line = self.process.stdout.readline().strip()
                if result_line and ":" in result_line and (":TASK_SUCCESS_RESPONSE" in result_line or ":TASK_FAILURE_RESPONSE" in result_line):
                    result_task_id, result_type = result_line.split(":", 1)
                    if result_task_id == self.current_task.task_id:
                        # Found the result just after timeout!
                        success = result_type == "TASK_SUCCESS_RESPONSE"
                        self._debug_print(f"Worker {self.worker_id} found late result for {self.current_task.task_id}: {result_type}")
                        
                        self.current_task.status = "completed" if success else "failed"
                        self.current_task.end_time = time.time()
                        
                        if success:
                            self.total_tasks_succeeded += 1
                        else:
                            self.total_tasks_failed += 1
                        
                        self.status = "idle"
                        return success
            
            # Really timed out
            self._debug_print(f"Worker {self.worker_id} task {self.current_task.task_id} timed out after {task_runtime:.1f}s")
            self.current_task.status = "failed"
            self.current_task.end_time = time.time()
            self.total_tasks_failed += 1
            self.status = "idle"
            return False
        
        return None
    
    def shutdown(self) -> None:
        """Gracefully shutdown the worker."""
        if self.process is None:
            return
        
        self.status = "shutting_down"
        
        try:
            # Send shutdown signal if stdin is still open
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.write("SHUT_DOWN\n")
                self.process.stdin.flush()
                self.process.stdin.close()
            
            # Wait for process to terminate
            self.process.wait(timeout=10)
            
        except subprocess.TimeoutExpired:
            # Force terminate if it doesn't shut down gracefully
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if terminate fails
                self.process.kill()
            
        except Exception as e:
            # Handle any other exceptions during shutdown
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
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