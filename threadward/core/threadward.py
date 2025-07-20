"""Main Threadward class for coordinating workers and tasks."""

import json
import os
import time
import threading
from typing import List, Dict, Any, Optional
from queue import Queue, Empty
from .task import Task
from .worker import Worker
from .variable_set import VariableSet

try:
    import GPUtil
except ImportError:
    GPUtil = None


class Threadward:
    """Main coordinator class for threadward execution."""
    
    def __init__(self, project_path: str = "."):
        """Initialize Threadward coordinator.
        
        Args:
            project_path: Path to the project directory containing threadward folder
        """
        self.project_path = os.path.abspath(project_path)
        self.threadward_path = os.path.join(project_path, "threadward")
        self.task_queue_path = os.path.join(self.threadward_path, "task_queue")
        
        # Task management
        self.tasks: List[Task] = []
        self.task_queue: Queue = Queue()
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        
        # Worker management
        self.workers: List[Worker] = []
        self.worker_assignment_lock = threading.Lock()
        
        # Execution state
        self.start_time: Optional[float] = None
        self.is_running = False
        self.should_stop = False
        
        # Configuration (loaded from files)
        self.config = {
            "NUM_WORKERS": 1,
            "NUM_GPUS_PER_WORKER": 0,
            "AVOID_GPUS": None,
            "INCLUDE_GPUS": None,
            "SUCCESS_CONDITION": "NO_ERROR_AND_VERIFY",
            "OUTPUT_MODE": "LOG_FILE_ONLY",
            "FAILURE_HANDLING": "PRINT_FAILURE_AND_CONTINUE",
            "TASK_FOLDER_LOCATION": "VARIABLE_SUBFOLDER"
        }
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from resource_constraints.py and other files."""
        try:
            # Load resource constraints
            constraints_path = os.path.join(self.threadward_path, "resource_constraints.py")
            if os.path.exists(constraints_path):
                with open(constraints_path, 'r') as f:
                    constraints_code = f.read()
                
                # Execute the constraints code to extract variables
                constraints_globals = {}
                exec(constraints_code, constraints_globals)
                
                # Update config with defined variables
                for key in ["NUM_WORKERS", "NUM_GPUS_PER_WORKER", "AVOID_GPUS", "INCLUDE_GPUS"]:
                    if key in constraints_globals:
                        self.config[key] = constraints_globals[key]
            
            # Load task specification constants
            task_spec_path = os.path.join(self.threadward_path, "task_specification.py")
            if os.path.exists(task_spec_path):
                with open(task_spec_path, 'r') as f:
                    task_spec_code = f.read()
                
                task_spec_globals = {}
                exec(task_spec_code, task_spec_globals)
                
                for key in ["SUCCESS_CONDITION", "OUTPUT_MODE"]:
                    if key in task_spec_globals:
                        self.config[key] = task_spec_globals[key]
            
            # Load variable iteration constants
            var_iter_path = os.path.join(self.threadward_path, "variable_iteration.py")
            if os.path.exists(var_iter_path):
                with open(var_iter_path, 'r') as f:
                    var_iter_code = f.read()
                
                var_iter_globals = {}
                exec(var_iter_code, var_iter_globals)
                
                for key in ["FAILURE_HANDLING", "TASK_FOLDER_LOCATION"]:
                    if key in var_iter_globals:
                        self.config[key] = var_iter_globals[key]
                        
        except Exception as e:
            print(f"Warning: Failed to load configuration: {e}")
    
    def generate_tasks(self) -> bool:
        """Generate all tasks based on variable_iteration.py.
        
        Returns:
            True if tasks generated successfully, False otherwise
        """
        try:
            # Load and execute variable_iteration.py
            var_iter_path = os.path.join(self.threadward_path, "variable_iteration.py")
            if not os.path.exists(var_iter_path):
                print(f"Error: {var_iter_path} not found")
                return False
            
            with open(var_iter_path, 'r') as f:
                var_iter_code = f.read()
            
            # Create variable set and execute the setup function
            variable_set = VariableSet()
            var_iter_globals = {"variable_set": variable_set}
            exec(var_iter_code, var_iter_globals)
            
            # Call setup_variable_set function
            if "setup_variable_set" not in var_iter_globals:
                print("Error: setup_variable_set function not found in variable_iteration.py")
                return False
            
            var_iter_globals["setup_variable_set"](variable_set)
            
            # Generate all combinations
            combinations = variable_set.generate_combinations()
            
            # Create tasks from combinations
            self.tasks = []
            for i, combo in enumerate(combinations):
                task_id = f"task_{i:06d}"
                
                # Extract task folder and create log file path
                task_folder = combo.pop("_task_folder")
                nicknames = combo.pop("_nicknames")
                
                log_file = os.path.join(task_folder, f"{task_id}.log")
                
                task = Task(
                    task_id=task_id,
                    variables=combo,
                    task_folder=task_folder,
                    log_file=log_file
                )
                
                self.tasks.append(task)
            
            # Save tasks to JSON file
            os.makedirs(self.task_queue_path, exist_ok=True)
            all_tasks_path = os.path.join(self.task_queue_path, "all_tasks.json")
            
            with open(all_tasks_path, 'w') as f:
                json.dump([task.to_dict() for task in self.tasks], f, indent=2)
            
            print(f"Generated {len(self.tasks)} tasks")
            return True
            
        except Exception as e:
            print(f"Error generating tasks: {e}")
            return False
    
    def create_workers(self) -> bool:
        """Create worker processes based on configuration.
        
        Returns:
            True if workers created successfully, False otherwise
        """
        try:
            num_workers = self.config["NUM_WORKERS"]
            num_gpus_per_worker = self.config["NUM_GPUS_PER_WORKER"]
            avoid_gpus = self.config["AVOID_GPUS"] or []
            include_gpus = self.config["INCLUDE_GPUS"]
            
            # Get available GPUs
            available_gpus = []
            if GPUtil and num_gpus_per_worker > 0:
                all_gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(all_gpus):
                    if i not in avoid_gpus:
                        if include_gpus is None or i in include_gpus:
                            available_gpus.append(i)
                
                # Check if we have enough GPUs
                total_gpus_needed = num_workers * num_gpus_per_worker
                if len(available_gpus) < total_gpus_needed:
                    print(f"Error: Need {total_gpus_needed} GPUs but only {len(available_gpus)} available")
                    return False
            
            # Create workers
            self.workers = []
            gpu_index = 0
            
            for worker_id in range(num_workers):
                # Assign GPUs to this worker
                worker_gpus = []
                if num_gpus_per_worker > 0:
                    for _ in range(int(num_gpus_per_worker)):
                        if gpu_index < len(available_gpus):
                            worker_gpus.append(available_gpus[gpu_index])
                            gpu_index += 1
                
                # Detect conda environment
                conda_env = os.environ.get("CONDA_DEFAULT_ENV")
                
                worker = Worker(worker_id, worker_gpus, conda_env)
                self.workers.append(worker)
            
            print(f"Created {len(self.workers)} workers")
            return True
            
        except Exception as e:
            print(f"Error creating workers: {e}")
            return False
    
    def start_workers(self) -> bool:
        """Start all worker processes.
        
        Returns:
            True if all workers started successfully, False otherwise
        """
        try:
            # Create worker script
            worker_script_path = self._create_worker_script()
            
            success_count = 0
            for worker in self.workers:
                if worker.start(worker_script_path):
                    success_count += 1
                else:
                    print(f"Failed to start worker {worker.worker_id}")
            
            if success_count == len(self.workers):
                print(f"All {len(self.workers)} workers started successfully")
                return True
            else:
                print(f"Only {success_count}/{len(self.workers)} workers started successfully")
                return False
                
        except Exception as e:
            print(f"Error starting workers: {e}")
            return False
    
    def _create_worker_script(self) -> str:
        """Create the worker script that will be executed by subprocesses."""
        worker_script_content = '''
import sys
import os
import json
import subprocess
import importlib.util

def main():
    worker_id = int(sys.argv[1])
    threadward_path = os.path.join(os.getcwd(), "threadward")
    
    # Load task specification module
    task_spec_path = os.path.join(threadward_path, "task_specification.py")
    spec = importlib.util.spec_from_file_location("task_specification", task_spec_path)
    task_spec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task_spec)
    
    # Load all tasks
    all_tasks_path = os.path.join(threadward_path, "task_queue", "all_tasks.json")
    with open(all_tasks_path, 'r') as f:
        all_tasks_data = json.load(f)
    
    # Call before_each_worker
    if hasattr(task_spec, 'before_each_worker'):
        task_spec.before_each_worker(worker_id)
    
    try:
        # Main worker loop
        while True:
            # Wait for task assignment or shutdown signal
            line = input().strip()
            
            if line == "SHUT_DOWN":
                break
            
            # Find the task
            task_data = None
            for task in all_tasks_data:
                if task["task_id"] == line:
                    task_data = task
                    break
            
            if task_data is None:
                print("FAILURE")
                continue
            
            try:
                # Execute the task
                success = execute_task(task_spec, task_data)
                print("SUCCESS" if success else "FAILURE")
                
            except Exception as e:
                print("FAILURE")
    
    finally:
        # Call after_each_worker
        if hasattr(task_spec, 'after_each_worker'):
            task_spec.after_each_worker(worker_id)

def execute_task(task_spec, task_data):
    """Execute a single task."""
    variables = task_data["variables"]
    task_folder = task_data["task_folder"]
    log_file = task_data["log_file"]
    
    # Create task folder
    os.makedirs(task_folder, exist_ok=True)
    
    # Call before_each_task
    if hasattr(task_spec, 'before_each_task'):
        task_spec.before_each_task(variables, task_folder, log_file)
    
    success = False
    try:
        # Execute the main task
        with open(log_file, 'w') as log:
            # Redirect stdout and stderr to log file
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            try:
                if hasattr(task_spec, 'OUTPUT_MODE') and task_spec.OUTPUT_MODE == "LOG_FILE_ONLY":
                    sys.stdout = log
                    sys.stderr = log
                elif hasattr(task_spec, 'OUTPUT_MODE') and task_spec.OUTPUT_MODE == "LOG_FILE_AND_CONSOLE":
                    sys.stdout = TeeOutput(old_stdout, log)
                    sys.stderr = TeeOutput(old_stderr, log)
                
                # Call the main task method
                task_spec.task_method(variables, task_folder, log_file)
                success = True
                
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        
        # Check success condition
        if hasattr(task_spec, 'SUCCESS_CONDITION'):
            if task_spec.SUCCESS_CONDITION == "NO_ERROR_AND_VERIFY":
                if success and hasattr(task_spec, 'verify_task_success'):
                    success = task_spec.verify_task_success(variables, task_folder, log_file)
            elif task_spec.SUCCESS_CONDITION == "VERIFY_ONLY":
                if hasattr(task_spec, 'verify_task_success'):
                    success = task_spec.verify_task_success(variables, task_folder, log_file)
            elif task_spec.SUCCESS_CONDITION == "ALWAYS_SUCCESS":
                success = True
            # NO_ERROR_ONLY uses the existing success value
    
    except Exception as e:
        success = False
        with open(log_file, 'a') as log:
            log.write(f"\\nError: {str(e)}\\n")
    
    finally:
        # Call after_each_task
        if hasattr(task_spec, 'after_each_task'):
            task_spec.after_each_task(variables, task_folder, log_file)
    
    return success

class TeeOutput:
    """Helper class to write to both console and file."""
    def __init__(self, console, file):
        self.console = console
        self.file = file
    
    def write(self, text):
        self.console.write(text)
        self.file.write(text)
    
    def flush(self):
        self.console.flush()
        self.file.flush()

if __name__ == "__main__":
    main()
'''
        
        worker_script_path = os.path.join(self.threadward_path, "worker_script.py")
        with open(worker_script_path, 'w') as f:
            f.write(worker_script_content)
        
        return worker_script_path
    
    def run(self) -> None:
        """Run the main threadward execution loop."""
        print("Starting Threadward execution...")
        
        # Generate tasks
        if not self.generate_tasks():
            return
        
        # Create and start workers
        if not self.create_workers():
            return
        
        if not self.start_workers():
            return
        
        # Call before_all_tasks
        self._call_hook("before_all_tasks")
        
        # Fill task queue
        for task in self.tasks:
            self.task_queue.put(task)
        
        self.is_running = True
        self.start_time = time.time()
        
        try:
            # Main execution loop
            self._execution_loop()
            
        finally:
            # Cleanup
            self._shutdown_workers()
            
            # Call after_all_tasks
            self._call_hook("after_all_tasks")
            
            self.is_running = False
            print("Threadward execution completed.")
    
    def _execution_loop(self) -> None:
        """Main execution loop for assigning tasks and monitoring workers."""
        while not self.task_queue.empty() and not self.should_stop:
            # Check for completed tasks and reassign workers
            for worker in self.workers:
                if worker.status == "busy":
                    result = worker.check_task_completion()
                    if result is not None:
                        # Task completed
                        task = worker.current_task
                        if result:
                            self.completed_tasks.append(task)
                            self._log_task_result(task, True)
                        else:
                            self.failed_tasks.append(task)
                            self._log_task_result(task, False)
                            self._handle_task_failure(task)
                
                # Assign new task if worker is idle
                if worker.status == "idle" and not self.task_queue.empty():
                    try:
                        next_task = self._get_next_task_for_worker(worker)
                        if next_task and worker.assign_task(next_task):
                            self._call_hook("before_each_task", 
                                          next_task.variables, next_task.task_folder, next_task.log_file)
                    except Exception as e:
                        print(f"Warning: Failed to assign task to worker {worker.worker_id}: {e}")
            
            time.sleep(0.1)  # Small delay to prevent busy waiting
        
        # Wait for remaining tasks to complete
        while any(worker.status == "busy" for worker in self.workers):
            for worker in self.workers:
                if worker.status == "busy":
                    result = worker.check_task_completion()
                    if result is not None:
                        task = worker.current_task
                        if result:
                            self.completed_tasks.append(task)
                            self._log_task_result(task, True)
                        else:
                            self.failed_tasks.append(task)
                            self._log_task_result(task, False)
                            self._handle_task_failure(task)
            
            time.sleep(0.1)
    
    def _get_next_task_for_worker(self, worker: Worker) -> Optional[Task]:
        """Get the next best task for a worker based on hierarchical variable retention."""
        if self.task_queue.empty():
            return None
        
        # For now, just get the next task in queue
        # TODO: Implement hierarchical variable retention logic
        try:
            return self.task_queue.get_nowait()
        except Empty:
            return None
    
    def _log_task_result(self, task: Task, success: bool) -> None:
        """Log task result to appropriate file."""
        result_file = "successful_tasks.txt" if success else "failed_tasks.txt"
        result_path = os.path.join(self.task_queue_path, result_file)
        
        with open(result_path, 'a') as f:
            f.write(f"{task.task_id}\n")
    
    def _handle_task_failure(self, task: Task) -> None:
        """Handle a failed task based on FAILURE_HANDLING setting."""
        failure_handling = self.config["FAILURE_HANDLING"]
        
        if failure_handling == "PRINT_FAILURE_AND_CONTINUE":
            print(f"Task {task.task_id} failed")
        elif failure_handling == "STOP_EXECUTION":
            print(f"Task {task.task_id} failed - stopping execution")
            self.should_stop = True
        # SILENT_CONTINUE does nothing
    
    def _call_hook(self, hook_name: str, *args, **kwargs) -> None:
        """Call a hook function from task_specification.py if it exists."""
        try:
            task_spec_path = os.path.join(self.threadward_path, "task_specification.py")
            if os.path.exists(task_spec_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("task_specification", task_spec_path)
                task_spec = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(task_spec)
                
                if hasattr(task_spec, hook_name):
                    getattr(task_spec, hook_name)(*args, **kwargs)
        except Exception as e:
            print(f"Warning: Failed to call hook {hook_name}: {e}")
    
    def _shutdown_workers(self) -> None:
        """Shutdown all workers gracefully."""
        for worker in self.workers:
            worker.shutdown()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time if self.start_time else 0
        
        total_tasks = len(self.tasks)
        completed_count = len(self.completed_tasks)
        failed_count = len(self.failed_tasks)
        remaining_count = total_tasks - completed_count - failed_count
        
        avg_time_per_task = elapsed_time / max(completed_count + failed_count, 1)
        estimated_remaining_time = avg_time_per_task * remaining_count if remaining_count > 0 else 0
        
        return {
            "elapsed_time": elapsed_time,
            "avg_time_per_task": avg_time_per_task,
            "estimated_remaining_time": estimated_remaining_time,
            "tasks": {
                "total": total_tasks,
                "completed": completed_count,
                "failed": failed_count,
                "remaining": remaining_count
            },
            "workers": [worker.get_stats() for worker in self.workers]
        }