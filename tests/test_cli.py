"""Test CLI functionality of threadward."""

import unittest
import tempfile
import os
import shutil
import json
import sys
from io import StringIO
from unittest.mock import patch, MagicMock

from threadward.cli.init_command import init_command
from threadward.cli.run_command import run_command


class TestInitCommand(unittest.TestCase):
    """Test threadward init command."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_init_command_creates_structure(self):
        """Test that init command creates proper directory structure."""
        init_command(self.test_dir)
        
        # Check that threadward directory was created
        threadward_dir = os.path.join(self.test_dir, "threadward")
        self.assertTrue(os.path.exists(threadward_dir))
        
        # Check that required files were created
        required_files = [
            "task_setup.py"
        ]
        
        for filename in required_files:
            file_path = os.path.join(threadward_dir, filename)
            self.assertTrue(os.path.exists(file_path), f"Missing file: {filename}")
            
            # Check that file is not empty
            with open(file_path, 'r') as f:
                content = f.read().strip()
                self.assertTrue(len(content) > 0, f"Empty file: {filename}")
        
        # Check task_queue directory
        task_queue_dir = os.path.join(threadward_dir, "task_queue")
        self.assertTrue(os.path.exists(task_queue_dir))
        
        # Check task queue files
        queue_files = ["all_tasks.json", "successful_tasks.txt", "failed_tasks.txt"]
        for filename in queue_files:
            file_path = os.path.join(task_queue_dir, filename)
            self.assertTrue(os.path.exists(file_path), f"Missing queue file: {filename}")
    
    def test_init_command_file_contents(self):
        """Test that init command creates files with proper content."""
        init_command(self.test_dir)
        threadward_dir = os.path.join(self.test_dir, "threadward")
        
        # Check task_setup.py contains required functions
        task_setup_path = os.path.join(threadward_dir, "task_setup.py")
        with open(task_setup_path, 'r') as f:
            content = f.read()
        
        required_functions = [
            "def task_method(",
            "def before_all_tasks(",
            "def after_all_tasks(",
            "def verify_task_success("
        ]
        
        for func in required_functions:
            self.assertIn(func, content, f"Missing function in task_setup.py: {func}")
        
        # Check for configuration constants
        self.assertIn("SUCCESS_CONDITION", content)
        self.assertIn("OUTPUT_MODE", content)
        
        # Check task_setup.py contains variable setup and configuration
        self.assertIn("def setup_variable_set(", content)
        self.assertIn("FAILURE_HANDLING", content)
        self.assertIn("TASK_FOLDER_LOCATION", content)
        self.assertIn("NUM_WORKERS", content)
        self.assertIn("NUM_GPUS_PER_WORKER", content)
    
    @patch('builtins.input', return_value='y')
    def test_init_command_overwrite_existing(self, mock_input):
        """Test init command overwrites existing directory when confirmed."""
        # Create initial threadward directory
        threadward_dir = os.path.join(self.test_dir, "threadward")
        os.makedirs(threadward_dir)
        
        # Create a test file
        test_file = os.path.join(threadward_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("old content")
        
        # Run init command (should overwrite)
        init_command(self.test_dir)
        
        # Check that old file is gone and new structure exists
        self.assertFalse(os.path.exists(test_file))
        self.assertTrue(os.path.exists(os.path.join(threadward_dir, "task_setup.py")))
    
    @patch('builtins.input', return_value='n')
    def test_init_command_cancel_overwrite(self, mock_input):
        """Test init command cancels when overwrite is declined."""
        # Create initial threadward directory
        threadward_dir = os.path.join(self.test_dir, "threadward")
        os.makedirs(threadward_dir)
        
        # Create a test file
        test_file = os.path.join(threadward_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("old content")
        
        # Run init command (should cancel)
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            init_command(self.test_dir)
            output = mock_stdout.getvalue()
        
        # Check that operation was cancelled
        self.assertIn("cancelled", output.lower())
        
        # Check that old file still exists
        self.assertTrue(os.path.exists(test_file))
        with open(test_file, 'r') as f:
            self.assertEqual(f.read(), "old content")


class TestRunCommand(unittest.TestCase):
    """Test threadward run command."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.threadward_dir = os.path.join(self.test_dir, "threadward")
        
        # Initialize a test project
        init_command(self.test_dir)
        
        # Modify for quick testing
        self._setup_quick_test()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def _setup_quick_test(self):
        """Set up a quick test configuration."""
        # Modify variable_iteration.py for quick test
        var_iter_content = '''
def setup_variable_set(variable_set):
    variable_set.add_variable("sleep_time", ["0.01"], nicknames=["quick"])
    variable_set.add_variable("test_case", ["A", "B"], nicknames=["case_A", "case_B"])

def sleep_time_to_value(string_value, nickname):
    return float(string_value)
'''
        
        var_iter_path = os.path.join(self.threadward_dir, "variable_iteration.py")
        with open(var_iter_path, 'w') as f:
            f.write(var_iter_content)
        
        # Modify task_specification.py for quick test
        task_spec_content = '''
import time
import os

def task_method(variables, task_folder, log_file):
    sleep_time = variables.get("sleep_time", 0.01)
    test_case = variables.get("test_case", "A")
    
    time.sleep(sleep_time)
    
    # Create success file
    with open(os.path.join(task_folder, "success.txt"), "w") as f:
        f.write(f"Test case {test_case} completed")

def verify_task_success(variables, task_folder, log_file):
    return os.path.exists(os.path.join(task_folder, "success.txt"))

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
'''
        
        task_spec_path = os.path.join(self.threadward_dir, "task_specification.py")
        with open(task_spec_path, 'w') as f:
            f.write(task_spec_content)
        
        # Set single worker for testing
        constraints_content = '''
NUM_WORKERS = 1
NUM_GPUS_PER_WORKER = 0
'''
        
        constraints_path = os.path.join(self.threadward_dir, "resource_constraints.py")
        with open(constraints_path, 'w') as f:
            f.write(constraints_content)
    
    def test_run_command_dry_run(self):
        """Test run command in dry run mode."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            run_command(self.test_dir, dry_run=True)
            output = mock_stdout.getvalue()
        
        # Check that dry run was performed
        self.assertIn("Dry run mode", output)
        self.assertIn("Generated", output)
        self.assertIn("tasks", output)
        
        # Check that all_tasks.json was created
        all_tasks_path = os.path.join(self.threadward_dir, "task_queue", "all_tasks.json")
        self.assertTrue(os.path.exists(all_tasks_path))
        
        # Verify task content
        with open(all_tasks_path, 'r') as f:
            tasks = json.load(f)
        
        self.assertEqual(len(tasks), 2)  # 1 sleep_time * 2 test_cases
        
        # Check task structure
        task = tasks[0]
        self.assertIn("task_id", task)
        self.assertIn("variables", task)
        self.assertIn("sleep_time", task["variables"])
        self.assertIn("test_case", task["variables"])
    
    def test_run_command_missing_project(self):
        """Test run command with missing project."""
        empty_dir = tempfile.mkdtemp()
        
        try:
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                run_command(empty_dir, dry_run=True)
                output = mock_stdout.getvalue()
            
            # Should report error about missing threadward directory
            self.assertIn("Error", output)
            self.assertIn("threadward directory", output)
            
        finally:
            shutil.rmtree(empty_dir)
    
    def test_run_command_missing_files(self):
        """Test run command with missing required files."""
        # Create threadward directory but remove a required file
        incomplete_dir = tempfile.mkdtemp()
        threadward_incomplete = os.path.join(incomplete_dir, "threadward")
        os.makedirs(threadward_incomplete)
        
        # Create only some files
        with open(os.path.join(threadward_incomplete, "task_specification.py"), 'w') as f:
            f.write("# incomplete")
        
        try:
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                run_command(incomplete_dir, dry_run=True)
                output = mock_stdout.getvalue()
            
            # Should report error about missing files
            self.assertIn("Error", output)
            self.assertIn("Required file not found", output)
            
        finally:
            shutil.rmtree(incomplete_dir)


class TestCLIIntegration(unittest.TestCase):
    """Integration tests for CLI commands."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_init_then_run_workflow(self):
        """Test complete workflow: init then run."""
        # Step 1: Initialize project
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            init_command(self.test_dir)
            init_output = mock_stdout.getvalue()
        
        self.assertIn("initialized", init_output.lower())
        
        # Step 2: Verify project structure
        threadward_dir = os.path.join(self.test_dir, "threadward")
        self.assertTrue(os.path.exists(threadward_dir))
        
        # Step 3: Run in dry run mode
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            run_command(self.test_dir, dry_run=True)
            run_output = mock_stdout.getvalue()
        
        # Should successfully generate tasks
        self.assertIn("Generated", run_output)
        self.assertIn("tasks", run_output)
        
        # Check that task files were created
        all_tasks_path = os.path.join(threadward_dir, "task_queue", "all_tasks.json")
        self.assertTrue(os.path.exists(all_tasks_path))
    
    def test_template_functionality(self):
        """Test that generated templates are functional."""
        # Initialize project
        init_command(self.test_dir)
        
        # Try to load and execute the generated files
        threadward_dir = os.path.join(self.test_dir, "threadward")
        
        # Test task_specification.py
        sys.path.insert(0, threadward_dir)
        try:
            import task_specification
            
            # Should be able to call functions without error
            task_specification.before_all_tasks()
            task_specification.after_all_tasks()
            task_specification.before_each_worker(0)
            task_specification.after_each_worker(0)
            
            # Test main task method
            variables = {"test": "value"}
            task_specification.task_method(variables, ".", "test.log")
            
            # Test verification
            result = task_specification.verify_task_success(variables, ".", "test.log")
            self.assertIsInstance(result, bool)
            
        finally:
            sys.path.remove(threadward_dir)
            # Clean up imported module
            if 'task_specification' in sys.modules:
                del sys.modules['task_specification']
        
        # Test variable_iteration.py
        sys.path.insert(0, threadward_dir)
        try:
            import variable_iteration
            from threadward.core.variable_set import VariableSet
            
            # Should be able to call setup function
            vs = VariableSet()
            variable_iteration.setup_variable_set(vs)
            
            # Should generate some combinations
            combinations = vs.generate_combinations()
            self.assertGreater(len(combinations), 0)
            
        finally:
            sys.path.remove(threadward_dir)
            if 'variable_iteration' in sys.modules:
                del sys.modules['variable_iteration']


if __name__ == "__main__":
    # Run CLI tests
    unittest.main(verbosity=2)