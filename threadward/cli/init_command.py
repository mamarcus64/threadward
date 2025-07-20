"""Init command implementation for threadward CLI."""

import os
import shutil
from pathlib import Path


def init_command(project_path: str = "."):
    """Initialize a new threadward project.
    
    Args:
        project_path: Path to initialize the project
    """
    project_path = os.path.abspath(project_path)
    threadward_path = os.path.join(project_path, "threadward")
    
    # Check if threadward directory already exists
    if os.path.exists(threadward_path):
        response = input(f"threadward directory already exists at {threadward_path}. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Initialization cancelled.")
            return
        
        # Remove existing directory
        shutil.rmtree(threadward_path)
    
    # Create directory structure
    os.makedirs(threadward_path, exist_ok=True)
    task_queue_path = os.path.join(threadward_path, "task_queue")
    os.makedirs(task_queue_path, exist_ok=True)
    
    # Create template files
    _create_task_specification(threadward_path)
    _create_resource_constraints(threadward_path)
    _create_variable_iteration(threadward_path)
    _create_empty_task_files(task_queue_path)
    
    print(f"[SUCCESS] Threadward project initialized at {threadward_path}")
    print()
    print("Next steps:")
    print("1. Edit threadward/task_specification.py - implement your main task")
    print("2. Edit threadward/variable_iteration.py - define variables to iterate over")
    print("3. Edit threadward/resource_constraints.py - configure workers and GPU usage")
    print("4. Run 'threadward run' to start execution")


def _create_task_specification(threadward_path: str):
    """Create task_specification.py template."""
    content = '''"""Task specification for threadward execution.

Implement the required task_method and optional setup/teardown methods.
"""

# Configuration constants
SUCCESS_CONDITION = "NO_ERROR_AND_VERIFY"  # NO_ERROR_AND_VERIFY, NO_ERROR_ONLY, VERIFY_ONLY, ALWAYS_SUCCESS
OUTPUT_MODE = "LOG_FILE_ONLY"  # LOG_FILE_ONLY, CONSOLE_ONLY, LOG_FILE_AND_CONSOLE


def task_method(variables, task_folder, log_file):
    """Main task to be executed for each combination of variables.
    
    Args:
        variables (dict): Dictionary mapping variable names to their values
        task_folder (str): Path to the folder for this specific task
        log_file (str): Path to the log file for this task
    
    This is the core method that you need to implement. It will be called
    for each combination of variables defined in variable_iteration.py.
    
    Example implementation:
        print(f"Processing with variables: {variables}")
        
        # Your task logic here
        import time
        time.sleep(1)  # Simulate work
        
        # Save results to task folder if needed
        with open(f"{task_folder}/result.txt", "w") as f:
            f.write(f"Task completed with variables: {variables}")
    """
    # TODO: Implement your task logic here
    import time
    print(f"Executing task with variables: {variables}")
    time.sleep(1)  # Replace with your actual task
    print("Task completed successfully")


def before_all_tasks():
    """Called once before any tasks begin execution.
    
    Use this for global setup that should happen once for the entire run.
    Examples: Loading models, setting up databases, initializing shared resources.
    """
    print("Setting up for all tasks...")


def after_all_tasks():
    """Called once after all tasks have completed.
    
    Use this for global cleanup or final result processing.
    Examples: Aggregating results, cleaning up resources, generating reports.
    """
    print("Cleaning up after all tasks...")


def before_each_worker(worker_id):
    """Called once for each worker when it starts.
    
    Args:
        worker_id (int): Unique identifier for this worker (0, 1, 2, ...)
    
    Use this for per-worker setup that should persist across tasks.
    Examples: Loading models into GPU memory, setting up worker-specific resources.
    """
    print(f"Setting up worker {worker_id}...")


def after_each_worker(worker_id):
    """Called once for each worker when it shuts down.
    
    Args:
        worker_id (int): Unique identifier for this worker
    
    Use this for per-worker cleanup.
    """
    print(f"Cleaning up worker {worker_id}...")


def before_each_task(variables, task_folder, log_file):
    """Called before each individual task execution.
    
    Args:
        variables (dict): Variables for this task (can be modified)
        task_folder (str): Path to task folder
        log_file (str): Path to log file
    
    Use this for per-task setup. Any modifications to 'variables' will be
    retained for the task execution.
    """
    pass


def after_each_task(variables, task_folder, log_file):
    """Called after each individual task execution.
    
    Args:
        variables (dict): Variables for this task
        task_folder (str): Path to task folder  
        log_file (str): Path to log file
    
    Use this for per-task cleanup or result processing.
    """
    pass


def verify_task_success(variables, task_folder, log_file):
    """Verify if a task completed successfully.
    
    Args:
        variables (dict): Variables for this task
        task_folder (str): Path to task folder
        log_file (str): Path to log file
    
    Returns:
        bool: True if task succeeded, False otherwise
    
    This method is called when SUCCESS_CONDITION includes verification.
    Implement custom logic to determine if the task was successful.
    """
    # TODO: Implement your verification logic
    # Example: Check if expected output files exist
    # return os.path.exists(f"{task_folder}/result.txt")
    return True
'''
    
    file_path = os.path.join(threadward_path, "task_specification.py")
    with open(file_path, 'w') as f:
        f.write(content)


def _create_resource_constraints(threadward_path: str):
    """Create resource_constraints.py template."""
    content = '''"""Resource constraints configuration for threadward execution.

Define how many workers to create and how to allocate GPUs.
"""

# Number of worker processes to create
NUM_WORKERS = 1

# Number of GPUs to assign to each worker
# Set to 0 to disable GPU allocation
# Can be fractional (e.g., 0.5 means 2 workers share 1 GPU)
NUM_GPUS_PER_WORKER = 0

# List of GPU IDs to avoid using (optional)
# Example: AVOID_GPUS = [0, 1] to avoid GPUs 0 and 1
AVOID_GPUS = None

# List of GPU IDs to use exclusively (optional)
# Example: INCLUDE_GPUS = [2, 3] to only use GPUs 2 and 3
# If both AVOID_GPUS and INCLUDE_GPUS are set, INCLUDE_GPUS takes precedence
INCLUDE_GPUS = None

# Examples of common configurations:

# Example 1: Single worker, no GPU
# NUM_WORKERS = 1
# NUM_GPUS_PER_WORKER = 0

# Example 2: 4 workers, each with 1 GPU
# NUM_WORKERS = 4
# NUM_GPUS_PER_WORKER = 1

# Example 3: 8 workers sharing 4 GPUs (2 workers per GPU)
# NUM_WORKERS = 8
# NUM_GPUS_PER_WORKER = 0.5

# Example 4: 2 workers using only GPUs 2 and 3
# NUM_WORKERS = 2
# NUM_GPUS_PER_WORKER = 1
# INCLUDE_GPUS = [2, 3]
'''
    
    file_path = os.path.join(threadward_path, "resource_constraints.py")
    with open(file_path, 'w') as f:
        f.write(content)


def _create_variable_iteration(threadward_path: str):
    """Create variable_iteration.py template."""
    content = '''"""Variable iteration configuration for threadward execution.

Define the variables to iterate over and their combinations.
"""

# Configuration constants
FAILURE_HANDLING = "PRINT_FAILURE_AND_CONTINUE"  # PRINT_FAILURE_AND_CONTINUE, SILENT_CONTINUE, STOP_EXECUTION
TASK_FOLDER_LOCATION = "VARIABLE_SUBFOLDER"  # VARIABLE_SUBFOLDER, VARIABLE_UNDERSCORE


def setup_variable_set(variable_set):
    """Define the variables and their values to iterate over.
    
    Args:
        variable_set: VariableSet object to configure
    
    The order of variables matters! Variables defined first are considered
    "higher level" in the hierarchy. Workers will retain higher-level variable
    values while iterating through lower-level combinations.
    
    Example: If you define variables in order [model, dataset, seed], then
    a worker will load one model and use it for all dataset/seed combinations
    before moving to the next model.
    """
    
    # Example configuration - replace with your own variables
    
    # Add your variables using variable_set.add_variable()
    # The first variable is the highest level in the hierarchy
    variable_set.add_variable(
        name="algorithm",
        values=["gradient_descent", "adam", "rmsprop"],
        nicknames=["GD", "Adam", "RMSprop"]  # Optional: shorter names for folders
    )
    
    # Second level variable
    variable_set.add_variable(
        name="learning_rate", 
        values=[0.001, 0.01, 0.1],
        nicknames=["lr_001", "lr_01", "lr_1"]
    )
    
    # Third level variable with exceptions
    variable_set.add_variable(
        name="batch_size",
        values=[16, 32, 64, 128],
        # Exceptions: limit batch sizes for certain learning rates
        exceptions={
            "0.1": ["16", "32"]  # High learning rate only with small batches (must be strings)
        }
    )
    
    # Lowest level variable
    variable_set.add_variable(
        name="seed",
        values=list(range(5)),  # Seeds 0-4
        nicknames=[f"seed_{i}" for i in range(5)]
    )
    
    # Optional: Add value converters
    # If you need to convert string values to other types, define converter functions
    # The function name should be {variable_name}_to_value
    
    # Example converter functions (uncomment and modify as needed):
    
    # def learning_rate_to_value(string_value, nickname):
    #     """Convert learning rate string to float."""
    #     return float(string_value)
    
    # def seed_to_value(string_value, nickname):
    #     """Convert seed string to int."""
    #     return int(string_value)
    
    # Add converters to the variable set
    # variable_set.add_converter("learning_rate", learning_rate_to_value)
    # variable_set.add_converter("seed", seed_to_value)


# Optional: Define converter functions for your variables
# Function name should be {variable_name}_to_value(string_value, nickname)

def learning_rate_to_value(string_value, nickname):
    """Convert learning rate from string to float."""
    return float(string_value)

def seed_to_value(string_value, nickname):
    """Convert seed from string to int."""
    return int(string_value)

# The setup_variable_set function should call variable_set.add_converter() for each converter
# This is done automatically by threadward when it detects functions with the correct naming pattern
'''
    
    file_path = os.path.join(threadward_path, "variable_iteration.py")
    with open(file_path, 'w') as f:
        f.write(content)


def _create_empty_task_files(task_queue_path: str):
    """Create empty task queue files."""
    files = ["all_tasks.json", "successful_tasks.txt", "failed_tasks.txt"]
    
    for filename in files:
        file_path = os.path.join(task_queue_path, filename)
        with open(file_path, 'w') as f:
            if filename.endswith('.json'):
                f.write('[]')  # Empty JSON array
            else:
                f.write('')  # Empty text file