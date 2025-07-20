"""Task class for threadward package."""

import json
import os
from typing import Dict, Any, Optional


class Task:
    """Represents a single task to be executed by a worker."""
    
    def __init__(self, task_id: str, variables: Dict[str, Any], 
                 task_folder: str, log_file: str):
        """Initialize a task.
        
        Args:
            task_id: Unique identifier for the task
            variables: Dictionary of variable name to value mappings
            task_folder: Directory path for task-specific files
            log_file: Path to log file for task output
        """
        self.task_id = task_id
        self.variables = variables
        self.task_folder = task_folder
        self.log_file = log_file
        self.status = "pending"  # pending, running, completed, failed
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.worker_id: Optional[int] = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "variables": self.variables,
            "task_folder": self.task_folder,
            "log_file": self.log_file,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary."""
        task = cls(
            task_id=data["task_id"],
            variables=data["variables"],
            task_folder=data["task_folder"],
            log_file=data["log_file"]
        )
        task.status = data.get("status", "pending")
        return task
    
    def get_folder_path(self, base_path: str = ".") -> str:
        """Get the full path to the task folder."""
        return os.path.join(base_path, self.task_folder)
    
    def create_folder(self, base_path: str = ".") -> None:
        """Create the task folder if it doesn't exist."""
        folder_path = self.get_folder_path(base_path)
        os.makedirs(folder_path, exist_ok=True)
    
    def get_log_path(self, base_path: str = ".") -> str:
        """Get the full path to the log file."""
        return os.path.join(base_path, self.log_file)
    
    def __str__(self) -> str:
        """String representation of the task."""
        return f"Task({self.task_id}, {self.status})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the task."""
        return f"Task(id={self.task_id}, status={self.status}, variables={self.variables})"