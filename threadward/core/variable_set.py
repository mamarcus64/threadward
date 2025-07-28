"""VariableSet class for threadward package."""

import itertools
from typing import List, Dict, Any, Optional, Callable, Union


class VariableSet:
    """Manages hierarchical variable combinations for task generation."""
    
    def __init__(self):
        """Initialize an empty variable set."""
        self.variables = []  # List of variable definitions in order
        self.variable_converters = {}  # Functions to convert string values
        
    def add_variable(self, name: str, values: List[Union[str, int, float]], 
                     nicknames: Optional[List[str]] = None,
                     interaction: str = "cartesian",
                     exceptions: Optional[Dict[str, List[str]]] = None) -> None:
        """Add a variable to the set.
        
        Args:
            name: Variable name
            values: List of possible values (will be converted to strings)
            nicknames: Optional list of nicknames for values (for folder naming)
            interaction: How this variable interacts with previous ones ('cartesian')
            exceptions: Dict mapping parent values to restricted child values
        """
        # Convert all values to strings
        str_values = [str(v) for v in values]
        
        # Use values as nicknames if not provided
        if nicknames is None:
            nicknames = str_values
        else:
            nicknames = [str(n) for n in nicknames]
            
        if len(nicknames) != len(str_values):
            raise ValueError(f"Number of nicknames ({len(nicknames)}) must match "
                           f"number of values ({len(str_values)}) for variable '{name}'")
        
        variable_def = {
            "name": name,
            "values": str_values,
            "nicknames": nicknames,
            "interaction": interaction,
            "exceptions": exceptions or {}
        }
        
        self.variables.append(variable_def)
    
    def add_converter(self, variable_name: str, converter: Callable[[str, str], Any]) -> None:
        """Add a converter function for a variable.
        
        Args:
            variable_name: Name of the variable
            converter: Function that takes (string_value, nickname) and returns converted value
        """
        self.variable_converters[variable_name] = converter
    
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
            if parent_value in exceptions:
                # Return only the allowed values for this parent
                return exceptions[parent_value]
        
        return all_values
    
    def _convert_combination(self, combo: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """Convert a combination to final format with converted values."""
        result = {}
        
        for var_name, var_data in combo.items():
            string_value = var_data["value"]
            nickname = var_data["nickname"]
            
            # Apply converter if available
            if var_name in self.variable_converters:
                converted_value = self.variable_converters[var_name](string_value, nickname)
                result[var_name] = converted_value
            else:
                result[var_name] = string_value
        
        # Add metadata
        result["_task_folder"] = self._generate_task_folder(combo, base_path=getattr(self, '_base_path', None))
        result["_nicknames"] = {var: data["nickname"] for var, data in combo.items()}
        
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