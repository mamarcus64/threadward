"""Init command implementation for threadward CLI."""

import os
import shutil
from pathlib import Path


def init_command(name: str = None, project_path: str = "."):
    """Initialize a new threadward configuration file.
    
    Args:
        name: Optional name for the threadward configuration (creates threadward_{name}.py if provided, threadward.py otherwise)
        project_path: Path to create the file in
    """
    project_path = os.path.abspath(project_path)
    
    if name:
        config_filename = f"threadward_{name}.py"
    else:
        config_filename = "threadward.py"
    
    config_path = os.path.join(project_path, config_filename)
    
    # Check if file already exists
    if os.path.exists(config_path):
        response = input(f"File {config_filename} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Initialization cancelled.")
            return
    
    # Create the configuration file
    _create_config_file(config_path, name)
    
    print(f"[SUCCESS] Threadward configuration created: {config_filename}")
    print()
    print("Next steps:")
    print(f"1. Edit {config_filename} - implement your task and configure variables")
    print(f"2. Run 'python {config_filename}' to start execution")


def _create_config_file(config_path: str, name: str):
    """Create minimalist threadward configuration file."""
    content = '''import threadward

SUCCESS_CONDITION = "NO_ERROR_AND_VERIFY"
OUTPUT_MODE = "LOG_FILE_ONLY"
NUM_WORKERS = 1
NUM_GPUS_PER_WORKER = 0
AVOID_GPUS = None
INCLUDE_GPUS = None
FAILURE_HANDLING = "PRINT_FAILURE_AND_CONTINUE"
TASK_FOLDER_LOCATION = "VARIABLE_SUBFOLDER"
EXISTING_FOLDER_HANDLING = "SKIP"

def task_method(variables, task_folder, log_file):
    pass

def before_all_tasks():
    pass

def after_all_tasks():
    pass

def before_each_worker(worker_id):
    pass

def after_each_worker(worker_id):
    pass

def before_each_task(variables, task_folder, log_file):
    pass

def after_each_task(variables, task_folder, log_file):
    pass

def verify_task_success(variables, task_folder, log_file):
    return True

def setup_variable_set(variable_set):
    pass

if __name__ == "__main__":
    threadward.run()
'''
    
    with open(config_path, 'w') as f:
        f.write(content)


