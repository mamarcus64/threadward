"""
threadward: Parallel Processing for Generalizable AI Experimentation in Python

A lightweight package that enables you to run custom scripts while iterating 
over combinations of script variables with automatic subprocess management and 
GPU allocation.
"""

__version__ = "0.1.0"
__author__ = "threadward"

from .core.threadward import Threadward as ThreadwardCore
from .core.base import Threadward
from .core.task import Task
from .core.worker import Worker
from .core.variable_set import VariableSet

def run(config_file_path=None):
    """Run threadward execution from a configuration file.
    
    Args:
        config_file_path: Path to the threadward configuration file. If None, 
                         auto-detects the calling file.
    """
    import os
    import importlib.util
    import inspect
    
    if config_file_path is None:
        # Auto-detect the calling file
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
    
    # Load the configuration module
    spec = importlib.util.spec_from_file_location("config", config_file_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    # Create and run threadward instance
    threadward = ThreadwardCore(config_dir, config)
    threadward.run()

__all__ = ["Threadward", "Task", "Worker", "VariableSet", "run"]