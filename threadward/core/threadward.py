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
from .interactive import InteractiveHandler

try:
    import GPUtil
except ImportError:
    GPUtil = None


class Threadward:
    """Main coordinator class for threadward execution."""
    
    def __init__(self, project_path: str, config_module):
        """Initialize Threadward coordinator.
        
        Args:
            project_path: Path to the project directory
            config_module: The loaded configuration module
        """
        self.project_path = os.path.abspath(project_path)
        self.config_module = config_module
        self.config_file_path = getattr(config_module, '__file__', None)
        
        # Create results directory structure
        self.results_path = os.path.join(project_path, "threadward_results")
        self.task_queue_path = os.path.join(self.results_path, "task_queue")
        
        # Task management
        self.all_tasks: List[Task] = []  # All tasks including skipped ones
        self.tasks: List[Task] = []      # Only tasks to be executed (non-skipped)
        self.task_queue: Queue = Queue()
        self.succeeded_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        self.skipped_tasks: List[Task] = []
        
        # Worker management
        self.workers: List[Worker] = []
        self.worker_assignment_lock = threading.Lock()
        
        # Execution state
        self.start_time: Optional[float] = None
        self.is_running = False
        self.should_stop = False
        
        # Interactive handler
        self.interactive_handler: Optional[InteractiveHandler] = None
        
        # Load configuration from module
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from config module."""
        # Default configuration
        self.config = {
            "NUM_WORKERS": 1,
            "NUM_GPUS_PER_WORKER": 0,
            "AVOID_GPUS": None,
            "INCLUDE_GPUS": None,
            "SUCCESS_CONDITION": "NO_ERROR_AND_VERIFY",
            "OUTPUT_MODE": "LOG_FILE_ONLY",
            "FAILURE_HANDLING": "PRINT_FAILURE_AND_CONTINUE",
            "TASK_FOLDER_LOCATION": "VARIABLE_SUBFOLDER",
            "EXISTING_FOLDER_HANDLING": "SKIP"
        }
        
        # Update with values from config module
        for key in self.config.keys():
            if hasattr(self.config_module, key):
                self.config[key] = getattr(self.config_module, key)
    
    def _handle_existing_folders(self) -> bool:
        """Handle existing task folders based on EXISTING_FOLDER_HANDLING setting.
        
        Returns:
            True if should continue execution, False if should quit
        """
        existing_folder_handling = self.config["EXISTING_FOLDER_HANDLING"]
        existing_folders = []
        
        # Check which task folders already exist
        for task in self.tasks:
            if os.path.exists(task.task_folder):
                existing_folders.append(task.task_folder)
        
        if not existing_folders:
            return True  # No existing folders, continue normally
        
        if existing_folder_handling == "SKIP":
            print(f"Found {len(existing_folders)} existing task folders - skipping these tasks")
            # Separate tasks into skipped and non-skipped
            remaining_tasks = []
            for task in self.tasks:
                if os.path.exists(task.task_folder):
                    self.skipped_tasks.append(task)
                else:
                    remaining_tasks.append(task)
            self.tasks = remaining_tasks
            print(f"Remaining tasks to execute: {len(self.tasks)}")
            return True
            
        elif existing_folder_handling == "OVERWRITE":
            print(f"Found {len(existing_folders)} existing task folders - will overwrite")
            # Delete existing folders
            import shutil
            for folder in existing_folders:
                try:
                    shutil.rmtree(folder)
                    print(f"Removed existing folder: {folder}")
                except Exception as e:
                    print(f"Warning: Failed to remove folder {folder}: {e}")
            return True
            
        elif existing_folder_handling == "QUIT":
            print(f"Found {len(existing_folders)} existing task folders - quitting execution")
            print("Existing folders:")
            for folder in existing_folders[:10]:  # Show first 10
                print(f"  {folder}")
            if len(existing_folders) > 10:
                print(f"  ... and {len(existing_folders) - 10} more")
            return False
            
        else:
            print(f"Warning: Unknown EXISTING_FOLDER_HANDLING value: {existing_folder_handling}")
            print("Using default behavior (SKIP)")
            # Separate tasks into skipped and non-skipped
            remaining_tasks = []
            for task in self.tasks:
                if os.path.exists(task.task_folder):
                    self.skipped_tasks.append(task)
                else:
                    remaining_tasks.append(task)
            self.tasks = remaining_tasks
            return True
    
    def generate_tasks(self) -> bool:
        """Generate all tasks based on config module.
        
        Returns:
            True if tasks generated successfully, False otherwise
        """
        try:
            # Create variable set and call setup function from config module
            variable_set = VariableSet()
            # Set base path for task folders
            variable_set._base_path = self.results_path
            
            if not hasattr(self.config_module, 'setup_variable_set'):
                print("Error: setup_variable_set function not found in configuration")
                return False
            
            self.config_module.setup_variable_set(variable_set)
            
            # Generate all combinations
            combinations = variable_set.generate_combinations()
            
            # Create tasks from combinations
            self.all_tasks = []
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
                
                self.all_tasks.append(task)
            
            # Initially, all tasks are to be executed
            self.tasks = self.all_tasks.copy()
            
            # Create results directory and task_queue folder
            os.makedirs(self.results_path, exist_ok=True)
            os.makedirs(self.task_queue_path, exist_ok=True)
            
            # Create empty task queue files
            queue_files = {
                "all_tasks.json": json.dumps([], indent=2),
                "successful_tasks.txt": "",
                "failed_tasks.txt": ""
            }
            
            for filename, content in queue_files.items():
                file_path = os.path.join(self.task_queue_path, filename)
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        f.write(content)
            
            # Handle existing task folders based on configuration
            if not self._handle_existing_folders():
                return False
            
            # Save tasks to JSON file
            all_tasks_path = os.path.join(self.task_queue_path, "all_tasks.json")
            with open(all_tasks_path, 'w') as f:
                json.dump([task.to_dict() for task in self.tasks], f, indent=2)
            
            print(f"Generated {len(self.all_tasks)} total tasks ({len(self.tasks)} to execute, {len(self.skipped_tasks)} skipped)")
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
            success_count = 0
            for worker in self.workers:
                if worker.start(self.config_file_path, self.results_path):
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
        
        # Start interactive handler
        self.interactive_handler = InteractiveHandler(self)
        self.interactive_handler.start()
        
        try:
            # Main execution loop
            self._execution_loop()
            
        finally:
            # Stop interactive handler
            if self.interactive_handler:
                self.interactive_handler.stop()
            
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
                        if task is not None:
                            if result:
                                self.succeeded_tasks.append(task)
                                self._log_task_result(task, True)
                            else:
                                self.failed_tasks.append(task)
                                self._log_task_result(task, False)
                                self._handle_task_failure(task)
                            # Clear the current task after logging
                            worker.current_task = None
                
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
                        if task is not None:
                            if result:
                                self.succeeded_tasks.append(task)
                                self._log_task_result(task, True)
                            else:
                                self.failed_tasks.append(task)
                                self._log_task_result(task, False)
                                self._handle_task_failure(task)
                            # Clear the current task after logging
                            worker.current_task = None
            
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
            f.flush()  # Ensure it's written to disk immediately
    
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
        """Call a hook function from config module if it exists."""
        try:
            if hasattr(self.config_module, hook_name):
                getattr(self.config_module, hook_name)(*args, **kwargs)
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
        
        # Total tasks includes all tasks (skipped and non-skipped)
        total_all_tasks = len(self.all_tasks)
        non_skipped_total = len(self.tasks) + len(self.succeeded_tasks) + len(self.failed_tasks)
        skipped_count = len(self.skipped_tasks)
        succeeded_count = len(self.succeeded_tasks)
        failed_count = len(self.failed_tasks)
        remaining_count = len(self.tasks) - succeeded_count - failed_count
        
        avg_time_per_task = elapsed_time / max(succeeded_count + failed_count, 1)
        estimated_remaining_time = avg_time_per_task * remaining_count if remaining_count > 0 else 0
        
        return {
            "elapsed_time": elapsed_time,
            "avg_time_per_task": avg_time_per_task,
            "estimated_remaining_time": estimated_remaining_time,
            "tasks": {
                "total": total_all_tasks,
                "non_skipped_total": non_skipped_total,
                "skipped": skipped_count,
                "succeeded": succeeded_count,
                "failed": failed_count,
                "remaining": remaining_count
            },
            "workers": [worker.get_stats() for worker in self.workers]
        }