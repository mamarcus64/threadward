"""Worker process logic for threadward."""

import sys 
import os
import json
import time
import traceback
import importlib.util


class VariableNamespace:
    """A namespace class that supports both dot notation and dict-style access."""
    
    def __init__(self, variables_dict, original_values=None, nicknames=None):
        """Initialize with a dictionary of variables.
        
        Args:
            variables_dict: Dict of converted variable values
            original_values: Dict of original string values  
            nicknames: Dict of nicknames for each variable
        """
        self._variables = variables_dict
        self._original_values = original_values or {}
        self._nicknames = nicknames or {}
    
    def __getattr__(self, name):
        """Support dot notation access (variables.model)."""
        if name.startswith('_'):
            # Allow access to private attributes like _variables
            return object.__getattribute__(self, name)
        if name in self._variables:
            return self._variables[name]
        raise AttributeError(f"'VariableNamespace' object has no attribute '{name}'")
    
    def __getitem__(self, key):
        """Support dict-style access (variables['model'])."""
        return self._variables[key]
    
    def __setitem__(self, key, value):
        """Support dict-style assignment."""
        self._variables[key] = value
    
    def __contains__(self, key):
        """Support 'in' operator."""
        return key in self._variables
    
    def get(self, key, default=None):
        """Support dict.get() method."""
        return self._variables.get(key, default)
    
    def get_name(self, value):
        """Get the original string value for a converted variable value.
        
        Args:
            value: The converted value to look up
            
        Returns:
            The original string value, or None if not found
        """
        for var_name, var_value in self._variables.items():
            if var_value is value:
                return self._original_values.get(var_name)
        return None
    
    def get_nickname(self, value):
        """Get the nickname for a converted variable value.
        
        Args:
            value: The converted value to look up
            
        Returns:
            The nickname, or None if not found
        """
        for var_name, var_value in self._variables.items():
            if var_value is value:
                return self._nicknames.get(var_name)
        return None
    
    def keys(self):
        """Support dict.keys() method."""
        return self._variables.keys()
    
    def values(self):
        """Support dict.values() method."""
        return self._variables.values()
    
    def items(self):
        """Support dict.items() method."""
        return self._variables.items()
    
    def __iter__(self):
        """Support iteration over keys."""
        return iter(self._variables)
    
    def __repr__(self):
        """String representation."""
        lines = ["Variables:"]
        for name, value in self._variables.items():
            lines.append(f"  {name}: {value}")
        return "\n".join(lines)


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
    
    # Create task folder first
    try:
        os.makedirs(task_folder, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Failed to create task folder {task_folder}: {e}", flush=True)
        return False
    
    success = False
    try:
        # Execute the main task
        with open(log_file, 'w') as log:
            # Redirect stdout and stderr to log file early
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            try:
                if hasattr(task_spec, 'OUTPUT_MODE') and task_spec.OUTPUT_MODE == "LOG_FILE_ONLY":
                    sys.stdout = log
                    sys.stderr = log
                elif hasattr(task_spec, 'OUTPUT_MODE') and task_spec.OUTPUT_MODE == "LOG_FILE_AND_CONSOLE":
                    sys.stdout = TeeOutput(old_stdout, log)
                    sys.stderr = TeeOutput(old_stderr, log)
                
                # Convert variables using to_value functions if converter function provided
                if convert_variables_func:
                    converted_variables = convert_variables_func(variables, nicknames)
                else:
                    # Create namespace with original values and nicknames even when no conversion needed
                    original_values = dict(variables)  # Variables may be any JSON type
                    final_nicknames = nicknames or {var: str(val) for var, val in variables.items()}
                    converted_variables = VariableNamespace(variables, original_values, final_nicknames)
                
                # Call before_each_task
                if hasattr(task_spec, 'before_each_task'):
                    task_spec.before_each_task(converted_variables, task_folder, log_file)
                
                # Print variables to log
                print(converted_variables)
                print()  # Add blank line for readability
                
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
        global_converter_info = {}
    else:
        # New format with converter info
        all_tasks_data = tasks_json.get("tasks", [])
        global_converter_info = tasks_json.get("converter_info", {})
    
    # Cache for converted values - only keep the most recent value per variable
    conversion_cache = {}  # {var_name: (string_value, converted_value)}
    
    # Function to convert variables using to_value functions
    def convert_variables(variables, nicknames=None):
        """Convert variables to objects using to_value functions if needed."""
        converted = {}
        original_values = {}
        final_nicknames = {}
        
        # Check if _has_converters is in variables (per-task converter info)
        has_converters = variables.get('_has_converters', {})
        
        for var_name, value in variables.items():
            # Skip metadata fields
            if var_name.startswith('_'):
                continue
                
            # Store original value (may be any JSON type)
            original_values[var_name] = value
            
            # Store nickname (use provided nickname or string representation as fallback)
            final_nicknames[var_name] = nicknames.get(var_name, str(value)) if nicknames else str(value)
            
            # Check if this variable has a converter (check per-task info first, then global)
            has_converter = has_converters.get(var_name, False) or (var_name in global_converter_info)
            if has_converter:
                # This variable has a converter - try multiple approaches to find it
                converter_func = None
                
                # Method 1: Check _threadward_converters dict
                converters = getattr(config_module, '_threadward_converters', {})
                if var_name in converters:
                    converter_func = converters[var_name]
                    print(f"Found converter for {var_name} in _threadward_converters", flush=True)
                
                # Method 2: Use function name from global_converter_info (JSON)
                if converter_func is None and var_name in global_converter_info:
                    func_name = global_converter_info[var_name]
                    if hasattr(config_module, func_name):
                        converter_func = getattr(config_module, func_name)
                        print(f"Found converter for {var_name} as {func_name} from JSON", flush=True)
                
                # Method 3: Check {var_name}_converter attribute
                if converter_func is None and hasattr(config_module, f"{var_name}_converter"):
                    converter_func = getattr(config_module, f"{var_name}_converter")
                    print(f"Found converter for {var_name} as {var_name}_converter", flush=True)
                
                # Method 4: Check {var_name}_to_value attribute (backward compatibility)
                if converter_func is None and hasattr(config_module, f"{var_name}_to_value"):
                    converter_func = getattr(config_module, f"{var_name}_to_value")
                    print(f"Found converter for {var_name} as {var_name}_to_value", flush=True)
                
                if converter_func:
                    nickname = final_nicknames[var_name]
                    try:
                        # Convert value to string for the converter function (backward compatibility)
                        string_value = str(value)
                        
                        # Check cache first - only if the cached value matches
                        if var_name in conversion_cache and conversion_cache[var_name][0] == string_value:
                            print(f"Using cached conversion for variable {var_name} with value '{string_value}'", flush=True)
                            converted[var_name] = conversion_cache[var_name][1]
                        else:
                            print(f"Converting variable {var_name} with value '{string_value}' via converter function", flush=True)
                            print(f"WORKER_DEBUG:Starting conversion of {var_name}={string_value}", flush=True)
                            converted_value = converter_func(string_value, nickname)
                            print(f"WORKER_DEBUG:Finished conversion of {var_name}={string_value}", flush=True)
                            # Store in cache, replacing any previous value for this variable
                            conversion_cache[var_name] = (string_value, converted_value)
                            converted[var_name] = converted_value
                    except Exception as e:
                        print(f"Error: Failed to convert {var_name} using converter function: {e}", flush=True)
                        import traceback
                        print(f"Traceback: {traceback.format_exc()}", flush=True)
                        converted[var_name] = value
                else:
                    # Debug info about what was tried
                    methods_tried = ["_threadward_converters"]
                    if var_name in global_converter_info:
                        methods_tried.append(f"JSON:{global_converter_info[var_name]}")
                    methods_tried.extend([f"{var_name}_converter", f"{var_name}_to_value"])
                    
                    print(f"Error: No converter function found for variable '{var_name}' (tried {', '.join(methods_tried)})", flush=True)
                    print(f"Debug: global_converter_info = {global_converter_info}", flush=True)
                    print(f"Debug: _threadward_converters = {getattr(config_module, '_threadward_converters', 'NOT_FOUND')}", flush=True)
                    if var_name in global_converter_info:
                        func_name = global_converter_info[var_name]
                        print(f"Debug: hasattr(config_module, '{func_name}') = {hasattr(config_module, func_name)}", flush=True)
                    # No converter function found, use original value
                    converted[var_name] = value
            else:
                # No converter needed, use original value with preserved type
                converted[var_name] = value
        
        result = VariableNamespace(converted, original_values, final_nicknames)
        return result
    
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
            sys.stdout.flush()
            
            # Send acknowledgment that we received the task
            print("TASK_RECEIVED", flush=True)
            sys.stdout.flush()
            
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
                    print(f"DEBUG: hierarchical_vars: {hierarchical_vars}", flush=True)
                    task_hierarchical_values = {var: task_data["variables"][var] 
                                               for var in hierarchical_vars 
                                               if var in task_data["variables"]}
                    print(f"DEBUG: task_hierarchical_values: {task_hierarchical_values}", flush=True)
                    
                    # Compute hierarchical key for this task
                    task_hierarchical_key = "|".join(
                        f"{var}={str(task_hierarchical_values[var])}" 
                        for var in hierarchical_vars if var in task_hierarchical_values
                    )
                    
                    # Check if we need to load new hierarchical values
                    print(f"DEBUG: Hierarchical key check - current: '{current_hierarchical_key}', task: '{task_hierarchical_key}'", flush=True)
                    if task_hierarchical_key != current_hierarchical_key:
                        print(f"DEBUG: Hierarchical key changed, converting hierarchical variables", flush=True)
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
                    else:
                        print(f"DEBUG: Hierarchical key unchanged, reusing existing hierarchical variables", flush=True)
                
                # Execute the task
                print(f"DEBUG: Worker {worker_id} starting task execution for '{task_data['task_id']}'", flush=True)
                sys.stdout.flush()
                
                # Save original stdout to ensure we can send results back
                original_stdout = sys.stdout
                
                success = execute_task(config_module, task_data, convert_variables)
                
                # Restore stdout and send result to parent process
                sys.stdout = original_stdout
                print(f"DEBUG: Worker {worker_id} task execution completed, success: {success}", flush=True)
                sys.stdout.flush()
                
                # Write result to a dedicated result file instead of stdout to avoid pipe issues
                result_msg = f"{task_data['task_id']}:TASK_SUCCESS_RESPONSE" if success else f"{task_data['task_id']}:TASK_FAILURE_RESPONSE"
                result_file = os.path.join(task_data['task_folder'], f"{task_data['task_id']}_result.txt")
                try:
                    with open(result_file, 'w') as f:
                        f.write(result_msg)
                        f.flush()
                        os.fsync(f.fileno())
                    print(f"DEBUG: Worker {worker_id} wrote result to {result_file}", flush=True)
                    print(f"WORKER_DEBUG:Task {task_data['task_id']} completed, result file created", flush=True)
                except Exception as e:
                    print(f"ERROR: Worker {worker_id} failed to write result file: {e}", flush=True)
                    # Fallback to stdout
                    print(result_msg, flush=True)
                
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
            def __init__(self, runner, original_module):
                self.runner = runner
                # Copy constraints as module attributes
                if hasattr(runner, '_constraints'):
                    for key, value in runner._constraints.items():
                        setattr(self, key, value)
                
                # Copy the _threadward_converters dict if it exists
                if hasattr(original_module, '_threadward_converters'):
                    setattr(self, '_threadward_converters', getattr(original_module, '_threadward_converters'))
                
                # Copy module-level converter functions (all patterns)
                for attr_name in dir(original_module):
                    if (attr_name.endswith('_to_value') or attr_name.endswith('_converter')) and not attr_name.startswith('_'):
                        attr_value = getattr(original_module, attr_name)
                        if callable(attr_value):
                            setattr(self, attr_name, attr_value)
                
                # Also copy all module-level functions that might be converters
                # Check global_converter_info to know which functions we need
                if hasattr(original_module, '__dict__'):
                    for attr_name, attr_value in original_module.__dict__.items():
                        if callable(attr_value) and not attr_name.startswith('_') and not attr_name[0].isupper():
                            # Copy all module-level functions (non-class, non-private)
                            setattr(self, attr_name, attr_value)
            
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
        
        original_config_module = config_module  # Save reference to original module
        config_module = ModuleWrapper(runner_instance, original_config_module)
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