"""VariableSet class for threadward package."""

import itertools
from typing import List, Dict, Any, Optional, Callable, Union


def _validate_json_type(value: Any, path: str = "") -> None:
    """Recursively validate that a value is JSON-serializable.
    
    Args:
        value: The value to validate
        path: Current path in the data structure (for error messages)
        
    Raises:
        ValueError: If the value contains non-JSON-serializable types
    """
    # JSON-compatible types: str, int, float, bool, list, dict, None
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _validate_json_type(item, f"{path}[{i}]")
    elif isinstance(value, dict):
        for key, val in value.items():
            if not isinstance(key, str):
                raise ValueError(f"Dictionary keys must be strings. Found {type(key).__name__} at {path}.{key}. "
                               "Only JSON-compatible types are allowed: str, int, float, bool, list, dict, None. "
                               "Use the to_value parameter for other types.")
            _validate_json_type(val, f"{path}.{key}")
    else:
        type_name = type(value).__name__
        raise ValueError(f"Type {type_name} at {path} is not JSON-compatible. "
                        "Only JSON-compatible types are allowed: str, int, float, bool, list, dict, None. "
                        "Use the to_value parameter for other types.")


class VariableSet:
    """Manages hierarchical variable combinations for task generation."""
    
    def __init__(self):
        """Initialize an empty variable set."""
        self.variables = []  # List of variable definitions in order
        self.variable_converters = {}  # Functions to convert string values
        
    def add_variable(self, name: str, values: List[Any], 
                     nicknames: Optional[List[str]] = None,
                     interaction: str = "cartesian",
                     exceptions: Optional[Dict[str, List[str]]] = None,
                     to_value: Optional[Callable[[str, str], Any]] = None) -> None:
        """Add a variable to the set.
        
        Args:
            name: Variable name
            values: List of possible values (must be JSON-compatible types)
            nicknames: Optional list of nicknames for values (for folder naming)
            interaction: How this variable interacts with previous ones ('cartesian')
            exceptions: Dict mapping parent values to restricted child values
            to_value: Optional function to convert string value to object (takes string_value, nickname)
        """
        # Validate that all values are JSON-compatible types
        for i, value in enumerate(values):
            _validate_json_type(value, f"values[{i}]")
        
        # Keep original values (don't convert to strings)
        original_values = values[:]
        
        # Use string representation of values as nicknames if not provided
        if nicknames is None:
            nicknames = [str(v) for v in original_values]
        else:
            nicknames = [str(n) for n in nicknames]
            
        if len(nicknames) != len(original_values):
            raise ValueError(f"Number of nicknames ({len(nicknames)}) must match "
                           f"number of values ({len(original_values)}) for variable '{name}'")
        
        variable_def = {
            "name": name,
            "values": original_values,  # Store original types, not strings
            "nicknames": nicknames,
            "interaction": interaction,
            "exceptions": exceptions or {},
            "to_value": to_value
        }
        
        self.variables.append(variable_def)
        
        # Store converter if provided (for backward compatibility and new to_value parameter)
        if to_value:
            self.variable_converters[name] = to_value
    
    def add_converter(self, variable_name: str, converter_func: Callable[[str, str], Any]) -> None:
        """Add a converter function for a variable (backward compatibility method).
        
        Args:
            variable_name: Name of the variable
            converter_func: Function to convert string value to object (takes string_value, nickname)
        """
        self.variable_converters[variable_name] = converter_func
    
    
    def generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all valid variable combinations using hierarchical retention.
        
        Returns:
            List of dictionaries, each containing variable values and metadata
        """
        if not self.variables:
            return []
        
        combinations = []
        
        # Start with the first variable
        first_var = self.variables[0]
        for i, value in enumerate(first_var["values"]):
            nickname = first_var["nicknames"][i]
            base_combination = {
                first_var["name"]: {
                    "value": value,
                    "nickname": nickname
                }
            }
            
            # Recursively build combinations for remaining variables
            self._build_combinations(base_combination, 1, combinations)
        
        return combinations
    
    def _build_combinations(self, current_combo: Dict[str, Dict[str, str]], 
                           var_index: int, combinations: List[Dict[str, Any]]) -> None:
        """Recursively build variable combinations."""
        if var_index >= len(self.variables):
            # Convert to final format and add to combinations
            final_combo = self._convert_combination(current_combo)
            combinations.append(final_combo)
            return
        
        current_var = self.variables[var_index]
        
        # Check for exceptions based on parent variables
        allowed_values = self._get_allowed_values(current_var, current_combo)
        
        for value in allowed_values:
            # Find the nickname for this value
            value_index = current_var["values"].index(value)
            nickname = current_var["nicknames"][value_index]
            
            # Create new combination with this variable added
            new_combo = current_combo.copy()
            new_combo[current_var["name"]] = {
                "value": value,
                "nickname": nickname
            }
            
            # Continue with next variable
            self._build_combinations(new_combo, var_index + 1, combinations)
    
    def _get_allowed_values(self, variable_def: Dict[str, Any], 
                           current_combo: Dict[str, Dict[str, str]]) -> List[str]:
        """Get allowed values for a variable based on exceptions and parent variables."""
        all_values = variable_def["values"]
        exceptions = variable_def["exceptions"]
        
        if not exceptions:
            return all_values
        
        # Check if any parent variable has exceptions for this variable
        for parent_var_name, parent_data in current_combo.items():
            parent_value = parent_data["value"]
            # Use string representation for exception lookup (backward compatibility)
            parent_value_str = str(parent_value)
            if parent_value_str in exceptions:
                # Return only the allowed values for this parent
                return exceptions[parent_value_str]
        
        return all_values
    
    def _convert_combination(self, combo: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """Convert a combination to final format (preserving original types)."""
        result = {}
        
        for var_name, var_data in combo.items():
            original_value = var_data["value"]
            # Use original value (already JSON-compatible)
            result[var_name] = original_value
        
        # Add metadata
        result["_task_folder"] = self._generate_task_folder(combo, base_path=getattr(self, '_base_path', None))
        result["_nicknames"] = {var: data["nickname"] for var, data in combo.items()}
        
        # Store which variables have to_value functions (for worker conversion)
        result["_has_converters"] = {
            var["name"]: var.get("to_value") is not None 
            for var in self.variables
        }
        
        return result
    
    def _generate_task_folder(self, combo: Dict[str, Dict[str, str]], 
                             mode: str = "VARIABLE_SUBFOLDER", base_path: str = None) -> str:
        """Generate task folder path based on variable nicknames."""
        if mode == "VARIABLE_SUBFOLDER":
            # Create nested folders: var1_nickname/var2_nickname/...
            folder_parts = []
            for var_def in self.variables:
                var_name = var_def["name"]
                if var_name in combo:
                    nickname = combo[var_name]["nickname"]
                    folder_parts.append(nickname)
            folder_path = "/".join(folder_parts)
        
        elif mode == "VARIABLE_UNDERSCORE":
            # Create single folder: var1_nickname_var2_nickname_...
            folder_parts = []
            for var_def in self.variables:
                var_name = var_def["name"]
                if var_name in combo:
                    nickname = combo[var_name]["nickname"]
                    folder_parts.append(nickname)
            folder_path = "_".join(folder_parts)
        
        else:
            raise ValueError(f"Unknown folder mode: {mode}")
        
        # Join with base path if provided
        if base_path:
            import os
            return os.path.join(base_path, folder_path)
        else:
            return folder_path
    
    def get_variable_hierarchy(self) -> List[str]:
        """Get the ordered list of variable names."""
        return [var["name"] for var in self.variables]
    
    def get_total_combinations(self) -> int:
        """Calculate total number of combinations."""
        return len(self.generate_combinations())
    
    def get_hierarchy_info(self, hierarchy_depth: Optional[int] = None) -> Dict[str, Any]:
        """Get information about the variable hierarchy.
        
        Args:
            hierarchy_depth: Number of top-level variables to consider for retention.
                           If None, uses all variables except the last one.
        
        Returns:
            Dict with hierarchy information including depth and variable names
        """
        if hierarchy_depth is None:
            hierarchy_depth = max(len(self.variables) - 1, 0)
        else:
            hierarchy_depth = min(hierarchy_depth, len(self.variables))
        
        hierarchical_vars = [var["name"] for var in self.variables[:hierarchy_depth]]
        
        return {
            "depth": hierarchy_depth,
            "hierarchical_variables": hierarchical_vars,
            "all_variables": self.get_variable_hierarchy()
        }
    
    def get_converter_info(self) -> Dict[str, str]:
        """Get information about which variables have converters.
        
        Returns:
            Dict mapping variable names to their converter function names
        """
        converter_info = {}
        for var in self.variables:
            if var.get("to_value") is not None:
                func = var["to_value"]
                func_name = func.__name__ if hasattr(func, '__name__') else f"{var['name']}_converter"
                # Store the function name so workers can look it up
                converter_info[var["name"]] = func_name
        return converter_info
    
    def export_converters(self, module):
        """Export to_value functions to a module so workers can access them.
        
        Args:
            module: The config module to add converter functions to
        """
        # Store mapping from variable names to their converter function names
        # This allows workers to find the original function by name
        converter_function_names = {}
        converter_functions = {}
        
        for var in self.variables:
            if var.get("to_value") is not None:
                func = var["to_value"]
                func_name = func.__name__ if hasattr(func, '__name__') else f"{var['name']}_converter"
                
                # Store both the function and its name mapping
                converter_functions[var["name"]] = func
                converter_function_names[var["name"]] = func_name
                
                # Also set the function as a module attribute for backward compatibility
                setattr(module, f"{var['name']}_converter", func)
        
        # Set both the function dict and name mapping as module attributes
        setattr(module, '_threadward_converters', converter_functions)
        setattr(module, '_threadward_converter_names', converter_function_names)