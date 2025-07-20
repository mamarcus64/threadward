"""
threadward: Parallel Processing for Generalizable AI Experimentation in Python

A lightweight package that enables you to run custom scripts while iterating 
over combinations of script variables with automatic subprocess management and 
GPU allocation.
"""

__version__ = "0.1.0"
__author__ = "threadward"

from .core.threadward import Threadward
from .core.task import Task
from .core.worker import Worker
from .core.variable_set import VariableSet

__all__ = ["Threadward", "Task", "Worker", "VariableSet"]