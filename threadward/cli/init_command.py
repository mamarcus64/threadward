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
        config_filename = "threadward_run.py"
    
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
    """Create threadward runner class configuration file."""
    
    # Determine class name
    if name:
        class_name = f"{name.capitalize()}Runner"
    else:
        class_name = "Runner"
    
    content = f'''import argparse
import threadward

class {class_name}(threadward.Threadward):
    def __init__(self, debug=False, results_folder="threadward_results"):
        super().__init__(debug=debug, results_folder=results_folder)
        self.set_constraints(
            SUCCESS_CONDITION="NO_ERROR_AND_VERIFY",
            OUTPUT_MODE="LOG_FILE_ONLY",
            NUM_WORKERS=1,
            NUM_GPUS_PER_WORKER=0,
            AVOID_GPUS=None,
            INCLUDE_GPUS=None,
            FAILURE_HANDLING="PRINT_FAILURE_AND_CONTINUE",
            TASK_FOLDER_LOCATION="VARIABLE_SUBFOLDER",
            EXISTING_FOLDER_HANDLING="SKIP",
            TASK_TIMEOUT=30  # Timeout in seconds (-1 for no timeout)
        )
    
    def task_method(self, variables, task_folder, log_file):
        pass
    
    def verify_task_success(self, variables, task_folder, log_file):
        return True
    
    def setup_variable_set(self, variable_set):
        pass

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run threadward experiments')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output for troubleshooting')
    parser.add_argument('--results-folder', default='threadward_results',
                       help='Name of the results folder (default: threadward_results)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    runner = {class_name}(debug=args.debug, results_folder=args.results_folder)
    runner.run()
'''
    
    with open(config_path, 'w') as f:
        f.write(content)


