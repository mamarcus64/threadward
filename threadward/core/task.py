"""Task class for threadward package."""

import json
import os
from typing import Dict, Any, Optional


class Task:
    """Represents a single task to be executed by a worker."""
    
    def __init__(self, task_id: str, variables: Dict[str, Any], 
                 task_folder: str, log_file: str,
                 hierarchy_info: Optional[Dict[str, Any]] = None):
        """Initialize a task.
        
        Args:
            task_id: Unique identifier for the task
            variables: Dictionary of variable name to value mappings
            task_folder: Directory path for task-specific files
            log_file: Path to log file for task output
            hierarchy_info: Optional hierarchy information for this task
        """
        self.task_id = task_id
        self.variables = variables
        self.task_folder = task_folder
        self.log_file = log_file
        self.status = "pending"  # pending, running, completed, failed
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.worker_id: Optional[int] = None
        
        # Hierarchy tracking
        self.hierarchy_info = hierarchy_info or {}
        self.hierarchical_key = self._compute_hierarchical_key()
        
    def _compute_hierarchical_key(self) -> str:
        """Compute the hierarchical key for this task based on hierarchical variables."""
        if not self.hierarchy_info or "hierarchical_variables" not in self.hierarchy_info:
            return ""
        
        hierarchical_vars = self.hierarchy_info.get("hierarchical_variables", [])
        key_parts = []
        
        for var_name in hierarchical_vars:
            if var_name in self.variables:
                # Use the string representation for the key
                key_parts.append(f"{var_name}={str(self.variables[var_name])}")
        
        return "|".join(key_parts)
    
    def get_hierarchical_values(self) -> Dict[str, Any]:
        """Get only the hierarchical variable values."""
        if not self.hierarchy_info or "hierarchical_variables" not in self.hierarchy_info:
            return {}
        
        hierarchical_vars = self.hierarchy_info.get("hierarchical_variables", [])
        return {var: self.variables[var] for var in hierarchical_vars if var in self.variables}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "variables": self.variables,
            "task_folder": self.task_folder,
            "log_file": self.log_file,
            "status": self.status,
            "hierarchy_info": self.hierarchy_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary."""
        task = cls(
            task_id=data["task_id"],
            variables=data["variables"],
            task_folder=data["task_folder"],
            log_file=data["log_file"],
            hierarchy_info=data.get("hierarchy_info")
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