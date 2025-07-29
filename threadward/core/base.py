"""Abstract base class for Threadward runners."""

from abc import ABC, abstractmethod
from .threadward import Threadward as ThreadwardCore
from .variable_set import VariableSet


class Threadward(ABC):
    """Abstract base class for Threadward experiment runners."""
    
    def __init__(self):
        """Initialize the Threadward runner with default settings."""
        self._constraints = {
            "SUCCESS_CONDITION": "NO_ERROR_AND_VERIFY",
            "OUTPUT_MODE": "LOG_FILE_ONLY", 
            "NUM_WORKERS": 1,
            "NUM_GPUS_PER_WORKER": 0,
            "AVOID_GPUS": None,
            "INCLUDE_GPUS": None,
            "FAILURE_HANDLING": "PRINT_FAILURE_AND_CONTINUE",
            "TASK_FOLDER_LOCATION": "VARIABLE_SUBFOLDER",
            "EXISTING_FOLDER_HANDLING": "SKIP"
        }
    
    def set_constraints(self, **kwargs):
        """Set execution constraints and resource parameters.
        
        Args:
            SUCCESS_CONDITION: How to determine task success
            OUTPUT_MODE: How to handle task output  
            NUM_WORKERS: Number of parallel workers
            NUM_GPUS_PER_WORKER: GPUs per worker
            AVOID_GPUS: List of GPU IDs to avoid
            INCLUDE_GPUS: List of GPU IDs to use exclusively
            FAILURE_HANDLING: How to handle task failures
            TASK_FOLDER_LOCATION: How to organize task folders
            EXISTING_FOLDER_HANDLING: What to do with existing task folders
        """
        self._constraints.update(kwargs)
    
    @abstractmethod
    def task_method(self, variables, task_folder, log_file):
        """Execute the main task logic.
        
        Args:
            variables: Dictionary of variable values for this task
            task_folder: Path to the task's output folder
            log_file: Path to the task's log file
        """
        pass
    
    @abstractmethod
    def verify_task_success(self, variables, task_folder, log_file):
        """Verify if the task completed successfully.
        
        Args:
            variables: Dictionary of variable values for this task
            task_folder: Path to the task's output folder
            log_file: Path to the task's log file
            
        Returns:
            bool: True if task succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    def setup_variable_set(self, variable_set):
        """Configure the variables to iterate over.
        
        Args:
            variable_set: VariableSet instance to configure
        """
        pass
    
    # Extensible hooks (can be overridden but not required)
    def before_all_tasks(self):
        """Called once before all tasks begin."""
        pass
    
    def after_all_tasks(self):
        """Called once after all tasks complete."""
        pass
    
    def before_each_worker(self, worker_id):
        """Called when each worker starts.
        
        Args:
            worker_id: ID of the worker starting
        """
        pass
    
    def after_each_worker(self, worker_id):
        """Called when each worker finishes.
        
        Args:
            worker_id: ID of the worker finishing
        """
        pass
    
    def before_each_task(self, variables, task_folder, log_file):
        """Called before each individual task.
        
        Args:
            variables: Dictionary of variable values for this task
            task_folder: Path to the task's output folder
            log_file: Path to the task's log file
        """
        pass
    
    def after_each_task(self, variables, task_folder, log_file):
        """Called after each individual task.
        
        Args:
            variables: Dictionary of variable values for this task
            task_folder: Path to the task's output folder
            log_file: Path to the task's log file
        """
        pass
    
    def run(self):
        """Execute the threadward experiment."""
        import os
        import inspect
        
        # Get the path of the calling file
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            config_file_path = caller_frame.f_globals.get('__file__')
            if config_file_path is None:
                raise RuntimeError("Could not auto-detect configuration file path")
        finally:
            del frame
        
        config_file_path = os.path.abspath(config_file_path)
        config_dir = os.path.dirname(config_file_path)
        
        # Create a mock config module from this instance
        class MockConfig:
            def __init__(self, runner_instance):
                self.runner = runner_instance
                # Copy constraints as module attributes
                for key, value in runner_instance._constraints.items():
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
        
        mock_config = MockConfig(self)
        # Add the file path to the mock config
        mock_config.__file__ = config_file_path
        
        # Create and run threadward instance
        threadward_core = ThreadwardCore(config_dir, mock_config)
        threadward_core.run()