"""Worker process logic for threadward."""

import sys 
import os
import json
import time
import traceback
import importlib.util


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


def execute_task(task_spec, task_data, convert_variables_func=None):
    """Execute a single task."""
    variables = task_data["variables"]
    task_folder = task_data["task_folder"]
    log_file = task_data["log_file"]
    nicknames = task_data.get("_nicknames", {})
    
    print(f"DEBUG: execute_task called for task_folder: {task_folder}", flush=True)
    print(f"DEBUG: execute_task log_file: {log_file}", flush=True)
    
    # Convert variables using to_value functions if converter function provided
    if convert_variables_func:
        converted_variables = convert_variables_func(variables, nicknames)
    else:
        converted_variables = variables
    
    # Create task folder
    print(f"DEBUG: Creating task folder: {task_folder}", flush=True)
    try:
        os.makedirs(task_folder, exist_ok=True)
        print(f"DEBUG: Task folder created successfully", flush=True)
    except Exception as e:
        print(f"ERROR: Failed to create task folder {task_folder}: {e}", flush=True)
        return False
    
    # Call before_each_task
    if hasattr(task_spec, 'before_each_task'):
        task_spec.before_each_task(converted_variables, task_folder, log_file)
    
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
                task_spec.task_method(converted_variables, task_folder, log_file)
                success = True
                
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        
        # Check success condition
        if hasattr(task_spec, 'SUCCESS_CONDITION'):
            if task_spec.SUCCESS_CONDITION == "NO_ERROR_AND_VERIFY":
                if success and hasattr(task_spec, 'verify_task_success'):
                    success = task_spec.verify_task_success(converted_variables, task_folder, log_file)
            elif task_spec.SUCCESS_CONDITION == "VERIFY_ONLY":
                if hasattr(task_spec, 'verify_task_success'):
                    success = task_spec.verify_task_success(converted_variables, task_folder, log_file)
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
            task_spec.after_each_task(converted_variables, task_folder, log_file)
    
    return success


def worker_main(worker_id, config_module, results_path):
    """Main worker process loop."""
    print(f"DEBUG: Worker {worker_id} entering main loop", flush=True)
    
    # Load all tasks and converter info
    all_tasks_path = os.path.join(results_path, "task_queue", "all_tasks.json")
    print(f"DEBUG: Worker {worker_id} loading tasks from: {all_tasks_path}", flush=True)
    try:
        with open(all_tasks_path, 'r') as f:
            tasks_json = json.load(f)
        print(f"DEBUG: Worker {worker_id} loaded {len(tasks_json) if isinstance(tasks_json, list) else len(tasks_json.get('tasks', [])) if isinstance(tasks_json, dict) else 'unknown'} tasks", flush=True)
    except Exception as e:
        print(f"ERROR: Worker {worker_id} failed to load tasks: {e}", flush=True)
        return
    
    # Handle both old and new format
    if isinstance(tasks_json, list):
        # Old format - just a list of tasks
        all_tasks_data = tasks_json
        converter_info = {}
    else:
        # New format with converter info
        all_tasks_data = tasks_json.get("tasks", [])
        converter_info = tasks_json.get("converter_info", {})
    
    # Function to convert variables using to_value functions
    def convert_variables(variables, nicknames=None):
        """Convert string variables to objects using to_value functions."""
        converted = {}
        for var_name, string_value in variables.items():
            if var_name in converter_info:
                # This variable has a converter
                converter_func_name = converter_info[var_name]
                if hasattr(config_module, converter_func_name):
                    converter_func = getattr(config_module, converter_func_name)
                    nickname = nicknames.get(var_name, string_value) if nicknames else string_value
                    try:
                        converted[var_name] = converter_func(string_value, nickname)
                    except Exception as e:
                        print(f"Warning: Failed to convert {var_name}: {e}", flush=True)
                        converted[var_name] = string_value
                else:
                    # No converter function found, use string value
                    converted[var_name] = string_value
            else:
                # No converter needed, use string value
                converted[var_name] = string_value
        return converted
    
    # Track hierarchical state
    current_hierarchical_key = ""
    current_hierarchical_values = {}
    current_converted_hierarchical_values = {}
    
    # Call before_each_worker
    print(f"DEBUG: Worker {worker_id} calling before_each_worker", flush=True)
    sys.stdout.flush()
    if hasattr(config_module, 'before_each_worker'):
        try:
            config_module.before_each_worker(worker_id)
            print(f"DEBUG: Worker {worker_id} before_each_worker completed", flush=True)
            sys.stdout.flush()
        except Exception as e:
            print(f"ERROR: Worker {worker_id} before_each_worker failed: {e}", flush=True)
            print(f"DEBUG: Worker {worker_id} before_each_worker traceback: {traceback.format_exc()}", flush=True)
            sys.stdout.flush()
            return
    else:
        print(f"DEBUG: Worker {worker_id} no before_each_worker method found", flush=True)
        sys.stdout.flush()
    
    try:
        # Main worker loop
        print(f"DEBUG: Worker {worker_id} starting main input loop", flush=True)
        
        # Signal that worker is ready to receive tasks
        print("WORKER_READY", flush=True)
        # Force flush stdout to ensure signal reaches parent immediately
        sys.stdout.flush()
        
        # Small delay to ensure parent has time to process the signal
        time.sleep(0.1)
        
        while True:
            try:
                # Wait for task assignment or shutdown signal
                line = input().strip()
            except EOFError:
                # Parent process closed stdin, worker should exit
                print(f"INFO: Worker {worker_id} received EOFError - parent process closed stdin, worker exiting", flush=True)
                print(f"DEBUG: Worker {worker_id} stdin state: closed={sys.stdin.closed if hasattr(sys.stdin, 'closed') else 'unknown'}", flush=True)
                break
            
            if line == "SHUT_DOWN":
                break
            
            # Find the task
            print(f"DEBUG: Worker {worker_id} received task ID: '{line}'", flush=True)
            task_data = None
            for task in all_tasks_data:
                if task["task_id"] == line:
                    task_data = task
                    break
            
            if task_data is None:
                print(f"ERROR: Worker {worker_id} could not find task '{line}' in all_tasks_data", flush=True)
                print(f"DEBUG: Available task IDs: {[t.get('task_id', 'NO_ID') for t in all_tasks_data[:5]]}..." if all_tasks_data else "DEBUG: all_tasks_data is empty", flush=True)
                print("TASK_FAILURE_RESPONSE", flush=True)
                sys.stdout.flush()
                continue
            
            try:
                # Check for hierarchical state change
                hierarchy_info = task_data.get("hierarchy_info", {})
                if hierarchy_info:
                    hierarchical_vars = hierarchy_info.get("hierarchical_variables", [])
                    task_hierarchical_values = {var: task_data["variables"][var] 
                                               for var in hierarchical_vars 
                                               if var in task_data["variables"]}
                    
                    # Compute hierarchical key for this task
                    task_hierarchical_key = "|".join(
                        f"{var}={str(task_hierarchical_values[var])}" 
                        for var in hierarchical_vars if var in task_hierarchical_values
                    )
                    
                    # Check if we need to load new hierarchical values
                    if task_hierarchical_key != current_hierarchical_key:
                        # Unload previous values if any
                        if current_hierarchical_key and hasattr(config_module, 'on_hierarchical_unload'):
                            # Pass converted values to unload
                            config_module.on_hierarchical_unload(current_converted_hierarchical_values, worker_id)
                        
                        # Convert hierarchical values for loading
                        task_nicknames = task_data.get("_nicknames", {})
                        converted_hierarchical_values = convert_variables(task_hierarchical_values, task_nicknames)
                        
                        # Load new values
                        if hasattr(config_module, 'on_hierarchical_load'):
                            # Pass converted values to load
                            config_module.on_hierarchical_load(converted_hierarchical_values, worker_id)
                        
                        current_hierarchical_key = task_hierarchical_key
                        current_hierarchical_values = task_hierarchical_values
                        current_converted_hierarchical_values = converted_hierarchical_values
                
                # Execute the task
                print(f"DEBUG: Worker {worker_id} starting task execution for '{task_data['task_id']}'", flush=True)
                sys.stdout.flush()
                success = execute_task(config_module, task_data, convert_variables)
                print(f"DEBUG: Worker {worker_id} task execution completed, success: {success}", flush=True)
                sys.stdout.flush()
                print("TASK_SUCCESS_RESPONSE" if success else "TASK_FAILURE_RESPONSE", flush=True)
                sys.stdout.flush()
                
            except Exception as e:
                print(f"ERROR: Worker {worker_id} exception during task processing: {e}", flush=True)
                print(f"DEBUG: Worker {worker_id} exception traceback: {traceback.format_exc()}", flush=True)
                print("TASK_FAILURE_RESPONSE", flush=True)
                sys.stdout.flush()
    
    finally:
        # Unload any remaining hierarchical values
        if current_hierarchical_key and hasattr(config_module, 'on_hierarchical_unload'):
            config_module.on_hierarchical_unload(current_converted_hierarchical_values, worker_id)
        
        # Call after_each_worker
        if hasattr(config_module, 'after_each_worker'):
            config_module.after_each_worker(worker_id)


def worker_main_from_file(worker_id, config_file_path, results_path):
    """Main worker process loop that loads config from file."""
    print(f"DEBUG: Worker {worker_id} starting initialization", flush=True)
    print(f"DEBUG: Worker {worker_id} config_file_path: {config_file_path}", flush=True)
    print(f"DEBUG: Worker {worker_id} results_path: {results_path}", flush=True)
    
    # Load the configuration module
    try:
        print(f"DEBUG: Worker {worker_id} loading config module", flush=True)
        spec = importlib.util.spec_from_file_location("config", config_file_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        print(f"DEBUG: Worker {worker_id} config module loaded successfully", flush=True)
    except Exception as e:
        print(f"ERROR: Worker {worker_id} failed to load config: {e}", flush=True)
        print(f"DEBUG: Worker {worker_id} config load traceback: {traceback.format_exc()}", flush=True)
        return
    
    # Check if this is a class-based runner
    print(f"DEBUG: Worker {worker_id} checking for runner class", flush=True)
    sys.stdout.flush()
    runner_instance = None
    for attr_name in dir(config_module):
        attr = getattr(config_module, attr_name)
        if (isinstance(attr, type) and 
            attr.__module__ == config_module.__name__ and
            hasattr(attr, 'task_method')):
            # Found a runner class, instantiate it
            print(f"DEBUG: Worker {worker_id} found runner class: {attr_name}", flush=True)
            sys.stdout.flush()
            runner_instance = attr()
            print(f"DEBUG: Worker {worker_id} instantiated runner class", flush=True)
            sys.stdout.flush()
            break
    
    if runner_instance:
        # Create a wrapper module that delegates to the runner instance
        class ModuleWrapper:
            def __init__(self, runner):
                self.runner = runner
                # Copy constraints as module attributes
                if hasattr(runner, '_constraints'):
                    for key, value in runner._constraints.items():
                        setattr(self, key, value)
            
            def task_method(self, variables, task_folder, log_file):
                return self.runner.task_method(variables, task_folder, log_file)
            
            def verify_task_success(self, variables, task_folder, log_file):
                return self.runner.verify_task_success(variables, task_folder, log_file)
            
            def setup_variable_set(self, variable_set):
                return self.runner.setup_variable_set(variable_set)
            
            def before_all_tasks(self):
                return self.runner.before_all_tasks()
            
            def after_all_tasks(self):
                return self.runner.after_all_tasks()
            
            def before_each_worker(self, worker_id):
                return self.runner.before_each_worker(worker_id)
            
            def after_each_worker(self, worker_id):
                return self.runner.after_each_worker(worker_id)
            
            def before_each_task(self, variables, task_folder, log_file):
                return self.runner.before_each_task(variables, task_folder, log_file)
            
            def after_each_task(self, variables, task_folder, log_file):
                return self.runner.after_each_task(variables, task_folder, log_file)
            
            def on_hierarchical_load(self, hierarchical_values, worker_id):
                if hasattr(self.runner, 'on_hierarchical_load'):
                    return self.runner.on_hierarchical_load(hierarchical_values, worker_id)
            
            def on_hierarchical_unload(self, hierarchical_values, worker_id):
                if hasattr(self.runner, 'on_hierarchical_unload'):
                    return self.runner.on_hierarchical_unload(hierarchical_values, worker_id)
        
        config_module = ModuleWrapper(runner_instance)
        print(f"DEBUG: Worker {worker_id} using ModuleWrapper for class-based runner", flush=True)
    else:
        print(f"DEBUG: Worker {worker_id} using config module directly", flush=True)
    
    # Run the main worker loop
    print(f"DEBUG: Worker {worker_id} calling worker_main", flush=True)
    worker_main(worker_id, config_module, results_path)
    print(f"DEBUG: Worker {worker_id} worker_main returned", flush=True)


if __name__ == "__main__":
    # This module should be called with proper imports, not directly
    print("This module should not be run directly", file=sys.stderr)
    sys.exit(1)