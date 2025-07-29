"""Worker process logic for threadward."""

import sys
import os
import json
import traceback


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
            log.write("\nError: " + str(e) + "\n")
            log.write(traceback.format_exc())
    
    finally:
        # Call after_each_task
        if hasattr(task_spec, 'after_each_task'):
            task_spec.after_each_task(variables, task_folder, log_file)
    
    return success


def worker_main(worker_id, config_module, results_path):
    """Main worker process loop."""
    # Load all tasks
    all_tasks_path = os.path.join(results_path, "task_queue", "all_tasks.json")
    with open(all_tasks_path, 'r') as f:
        all_tasks_data = json.load(f)
    
    # Call before_each_worker
    if hasattr(config_module, 'before_each_worker'):
        config_module.before_each_worker(worker_id)
    
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
                print("FAILURE", flush=True)
                sys.stdout.flush()
                continue
            
            try:
                # Execute the task
                success = execute_task(config_module, task_data)
                print("SUCCESS" if success else "FAILURE", flush=True)
                sys.stdout.flush()
                
            except Exception as e:
                print("FAILURE", flush=True)
                sys.stdout.flush()
    
    finally:
        # Call after_each_worker
        if hasattr(config_module, 'after_each_worker'):
            config_module.after_each_worker(worker_id)


if __name__ == "__main__":
    # This module should be called with proper imports, not directly
    print("This module should not be run directly", file=sys.stderr)
    sys.exit(1)