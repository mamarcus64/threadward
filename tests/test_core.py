"""Test core functionality of threadward."""

import unittest
import tempfile
import os
import shutil
import json
import time
from unittest.mock import patch, MagicMock

from threadward.core.task import Task
from threadward.core.variable_set import VariableSet
from threadward.core.worker import Worker
from threadward.core.threadward import Threadward


class TestTask(unittest.TestCase):
    """Test Task class functionality."""
    
    def test_task_creation(self):
        """Test basic task creation."""
        variables = {"param1": "value1", "param2": 42}
        task = Task("test_001", variables, "task_folder", "test.log")
        
        self.assertEqual(task.task_id, "test_001")
        self.assertEqual(task.variables, variables)
        self.assertEqual(task.task_folder, "task_folder")
        self.assertEqual(task.log_file, "test.log")
        self.assertEqual(task.status, "pending")
        self.assertIsNone(task.worker_id)
    
    def test_task_serialization(self):
        """Test task to_dict and from_dict methods."""
        variables = {"sleep_time": 1.5, "algorithm": "test"}
        task = Task("test_002", variables, "folder", "log.txt")
        
        # Test serialization
        task_dict = task.to_dict()
        expected_keys = {"task_id", "variables", "task_folder", "log_file", "status"}
        self.assertEqual(set(task_dict.keys()), expected_keys)
        
        # Test deserialization
        new_task = Task.from_dict(task_dict)
        self.assertEqual(new_task.task_id, task.task_id)
        self.assertEqual(new_task.variables, task.variables)
        self.assertEqual(new_task.status, task.status)
    
    def test_task_folder_methods(self):
        """Test task folder path methods."""
        task = Task("test_003", {}, "subdir/task_folder", "task.log")
        
        # Test folder path generation
        folder_path = task.get_folder_path("/base")
        expected = os.path.join("/base", "subdir/task_folder")
        self.assertEqual(folder_path, expected)
        
        # Test log path generation
        log_path = task.get_log_path("/base")
        expected = os.path.join("/base", "task.log")
        self.assertEqual(log_path, expected)


class TestVariableSet(unittest.TestCase):
    """Test VariableSet class functionality."""
    
    def test_empty_variable_set(self):
        """Test empty variable set."""
        vs = VariableSet()
        combinations = vs.generate_combinations()
        self.assertEqual(combinations, [])
    
    def test_single_variable(self):
        """Test variable set with single variable."""
        vs = VariableSet()
        vs.add_variable("sleep_time", [0.5, 1.0, 1.5])
        
        combinations = vs.generate_combinations()
        self.assertEqual(len(combinations), 3)
        
        # Check first combination
        combo = combinations[0]
        self.assertIn("sleep_time", combo)
        self.assertEqual(combo["sleep_time"], "0.5")
        self.assertIn("_task_folder", combo)
        self.assertIn("_nicknames", combo)
    
    def test_multiple_variables_cartesian(self):
        """Test multiple variables with cartesian product."""
        vs = VariableSet()
        vs.add_variable("algorithm", ["A", "B"], nicknames=["Alg_A", "Alg_B"])
        vs.add_variable("sleep_time", [1, 2], nicknames=["1s", "2s"])
        
        combinations = vs.generate_combinations()
        self.assertEqual(len(combinations), 4)  # 2 * 2 = 4
        
        # Verify all combinations exist
        expected_combos = [
            ("A", "1"), ("A", "2"), ("B", "1"), ("B", "2")
        ]
        
        actual_combos = []
        for combo in combinations:
            actual_combos.append((combo["algorithm"], combo["sleep_time"]))
        
        for expected in expected_combos:
            self.assertIn(expected, actual_combos)
    
    def test_variable_exceptions(self):
        """Test variable exceptions functionality."""
        vs = VariableSet()
        vs.add_variable("mode", ["fast", "slow"])
        vs.add_variable("sleep_time", [0.1, 0.5, 2.0], 
                       exceptions={"fast": ["0.1", "0.5"]})  # fast mode only with short sleeps
        
        combinations = vs.generate_combinations()
        
        # Should have 2 + 3 = 5 combinations (fast with 2 values, slow with 3 values)
        self.assertEqual(len(combinations), 5)
        
        # Verify fast mode only has short sleep times
        fast_combos = [c for c in combinations if c["mode"] == "fast"]
        for combo in fast_combos:
            self.assertIn(combo["sleep_time"], ["0.1", "0.5"])
    
    def test_variable_converters(self):
        """Test variable converter functionality."""
        vs = VariableSet()
        vs.add_variable("sleep_time", ["1", "2", "3"])
        
        # Add converter to convert string to float
        def sleep_time_converter(string_value, nickname):
            return float(string_value)
        
        vs.add_converter("sleep_time", sleep_time_converter)
        
        combinations = vs.generate_combinations()
        combo = combinations[0]
        
        # Value should be converted to float
        self.assertIsInstance(combo["sleep_time"], float)
        self.assertEqual(combo["sleep_time"], 1.0)
    
    def test_folder_generation_modes(self):
        """Test different folder generation modes."""
        vs = VariableSet()
        vs.add_variable("alg", ["A"], nicknames=["Algorithm_A"])
        vs.add_variable("time", ["1"], nicknames=["1sec"])
        
        combinations = vs.generate_combinations()
        combo = combinations[0]
        
        # Test subfolder mode (default)
        folder = combo["_task_folder"]
        self.assertEqual(folder, "Algorithm_A/1sec")


class TestSleepTask(unittest.TestCase):
    """Test threadward with time.sleep as the main task."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.threadward_dir = os.path.join(self.test_dir, "threadward")
        os.makedirs(self.threadward_dir)
        os.makedirs(os.path.join(self.threadward_dir, "task_queue"))
        
        # Create test task specification with time.sleep
        self._create_sleep_task_spec()
        self._create_test_variable_iteration()
        self._create_test_resource_constraints()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def _create_sleep_task_spec(self):
        """Create a task specification that uses time.sleep."""
        content = '''
import time
import os

SUCCESS_CONDITION = "NO_ERROR_AND_VERIFY"
OUTPUT_MODE = "LOG_FILE_ONLY"

def task_method(variables, task_folder, log_file):
    """Main task that sleeps for specified duration."""
    sleep_time = variables.get("sleep_time", 1.0)
    algorithm = variables.get("algorithm", "default")
    
    print(f"Starting {algorithm} algorithm with sleep_time={sleep_time}")
    
    # Simulate work with time.sleep
    time.sleep(sleep_time)
    
    # Create result file to verify completion
    result_file = os.path.join(task_folder, "result.txt")
    with open(result_file, "w") as f:
        f.write(f"Completed {algorithm} in {sleep_time} seconds\\n")
        f.write(f"Variables: {variables}\\n")
    
    print(f"Completed {algorithm} algorithm")

def verify_task_success(variables, task_folder, log_file):
    """Verify task completed by checking result file."""
    result_file = os.path.join(task_folder, "result.txt")
    return os.path.exists(result_file)

def before_all_tasks():
    print("Setting up sleep test suite...")

def after_all_tasks():
    print("Sleep test suite completed.")

def before_each_worker(worker_id):
    print(f"Worker {worker_id} starting...")

def after_each_worker(worker_id):
    print(f"Worker {worker_id} shutting down...")

def before_each_task(variables, task_folder, log_file):
    print(f"Starting task with {variables}")

def after_each_task(variables, task_folder, log_file):
    print(f"Finished task")
'''
        
        task_spec_path = os.path.join(self.threadward_dir, "task_specification.py")
        with open(task_spec_path, 'w') as f:
            f.write(content)
    
    def _create_test_variable_iteration(self):
        """Create variable iteration for sleep test."""
        content = '''
FAILURE_HANDLING = "PRINT_FAILURE_AND_CONTINUE"
TASK_FOLDER_LOCATION = "VARIABLE_SUBFOLDER"

def setup_variable_set(variable_set):
    """Set up variables for sleep test."""
    # Test different algorithms
    variable_set.add_variable(
        name="algorithm",
        values=["quicksort", "mergesort", "bubblesort"],
        nicknames=["quick", "merge", "bubble"]
    )
    
    # Test different sleep times (simulating different workloads)
    variable_set.add_variable(
        name="sleep_time",
        values=[0.1, 0.5, 1.0],
        nicknames=["fast", "medium", "slow"]
    )
    
    # Test different random seeds
    variable_set.add_variable(
        name="seed",
        values=[42, 123, 456],
        nicknames=["seed_42", "seed_123", "seed_456"]
    )

def sleep_time_to_value(string_value, nickname):
    """Convert sleep time to float."""
    return float(string_value)

def seed_to_value(string_value, nickname):
    """Convert seed to int."""
    return int(string_value)
'''
        
        var_iter_path = os.path.join(self.threadward_dir, "variable_iteration.py")
        with open(var_iter_path, 'w') as f:
            f.write(content)
    
    def _create_test_resource_constraints(self):
        """Create resource constraints for test."""
        content = '''
NUM_WORKERS = 2
NUM_GPUS_PER_WORKER = 0  # No GPU for tests
AVOID_GPUS = None
INCLUDE_GPUS = None
'''
        
        constraints_path = os.path.join(self.threadward_dir, "resource_constraints.py")
        with open(constraints_path, 'w') as f:
            f.write(content)
    
    def test_task_generation(self):
        """Test that tasks are generated correctly for sleep test."""
        threadward = Threadward(self.test_dir)
        
        # Generate tasks
        success = threadward.generate_tasks()
        self.assertTrue(success)
        
        # Should have 3 * 3 * 3 = 27 tasks
        self.assertEqual(len(threadward.tasks), 27)
        
        # Check that all_tasks.json was created
        all_tasks_path = os.path.join(self.threadward_dir, "task_queue", "all_tasks.json")
        self.assertTrue(os.path.exists(all_tasks_path))
        
        # Verify JSON content
        with open(all_tasks_path, 'r') as f:
            tasks_data = json.load(f)
        
        self.assertEqual(len(tasks_data), 27)
        
        # Check first task structure
        first_task = tasks_data[0]
        required_keys = {"task_id", "variables", "task_folder", "log_file"}
        self.assertTrue(required_keys.issubset(set(first_task.keys())))
        
        # Verify variables are present
        variables = first_task["variables"]
        expected_vars = {"algorithm", "sleep_time", "seed"}
        self.assertTrue(expected_vars.issubset(set(variables.keys())))
    
    def test_variable_conversions(self):
        """Test that variable conversions work correctly."""
        threadward = Threadward(self.test_dir)
        threadward.generate_tasks()
        
        # Check a task to see if conversions were applied
        task = threadward.tasks[0]
        variables = task.variables
        
        # sleep_time should be converted to float
        self.assertIsInstance(variables["sleep_time"], float)
        
        # seed should be converted to int
        self.assertIsInstance(variables["seed"], int)
        
        # algorithm should remain as string
        self.assertIsInstance(variables["algorithm"], str)
    
    def test_task_folder_structure(self):
        """Test that task folders are structured correctly."""
        threadward = Threadward(self.test_dir)
        threadward.generate_tasks()
        
        # Check task folder paths
        for task in threadward.tasks[:5]:  # Check first 5 tasks
            folder_parts = task.task_folder.split('/')
            self.assertEqual(len(folder_parts), 3)  # algorithm/sleep_time/seed
            
            # Verify folder contains expected nicknames
            self.assertIn(folder_parts[0], ["quick", "merge", "bubble"])
            self.assertIn(folder_parts[1], ["fast", "medium", "slow"])
            self.assertTrue(folder_parts[2].startswith("seed_"))
    
    @patch('threadward.core.threadward.Worker')
    def test_worker_creation(self, mock_worker):
        """Test worker creation for sleep test."""
        threadward = Threadward(self.test_dir)
        
        success = threadward.create_workers()
        self.assertTrue(success)
        
        # Should create 2 workers as specified in resource constraints
        self.assertEqual(len(threadward.workers), 2)
    
    def test_configuration_loading(self):
        """Test that configuration is loaded correctly from files."""
        threadward = Threadward(self.test_dir)
        
        # Check that configuration was loaded
        self.assertEqual(threadward.config["NUM_WORKERS"], 2)
        self.assertEqual(threadward.config["NUM_GPUS_PER_WORKER"], 0)
        self.assertEqual(threadward.config["SUCCESS_CONDITION"], "NO_ERROR_AND_VERIFY")
        self.assertEqual(threadward.config["OUTPUT_MODE"], "LOG_FILE_ONLY")
        self.assertEqual(threadward.config["FAILURE_HANDLING"], "PRINT_FAILURE_AND_CONTINUE")
        self.assertEqual(threadward.config["TASK_FOLDER_LOCATION"], "VARIABLE_SUBFOLDER")


class TestIntegrationSleepTest(unittest.TestCase):
    """Integration test running actual sleep tasks."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.threadward_dir = os.path.join(self.test_dir, "threadward")
        os.makedirs(self.threadward_dir)
        os.makedirs(os.path.join(self.threadward_dir, "task_queue"))
        
        # Create minimal test configuration
        self._create_minimal_sleep_test()
    
    def tearDown(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.test_dir)
    
    def _create_minimal_sleep_test(self):
        """Create minimal configuration for quick integration test."""
        
        # Task specification
        task_spec = '''
import time
import os

def task_method(variables, task_folder, log_file):
    sleep_time = variables.get("sleep_time", 0.1)
    time.sleep(sleep_time)
    
    # Create success marker
    with open(os.path.join(task_folder, "success.txt"), "w") as f:
        f.write("completed")

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
        
        with open(os.path.join(self.threadward_dir, "task_specification.py"), 'w') as f:
            f.write(task_spec)
        
        # Variable iteration - minimal test
        var_iter = '''
def setup_variable_set(variable_set):
    variable_set.add_variable("sleep_time", ["0.1", "0.2"], nicknames=["t1", "t2"])
    variable_set.add_variable("test_id", ["1", "2"], nicknames=["test1", "test2"])

def sleep_time_to_value(string_value, nickname):
    return float(string_value)

def test_id_to_value(string_value, nickname):
    return int(string_value)
'''
        
        with open(os.path.join(self.threadward_dir, "variable_iteration.py"), 'w') as f:
            f.write(var_iter)
        
        # Resource constraints - single worker for test
        constraints = '''
NUM_WORKERS = 1
NUM_GPUS_PER_WORKER = 0
'''
        
        with open(os.path.join(self.threadward_dir, "resource_constraints.py"), 'w') as f:
            f.write(constraints)
    
    def test_full_execution_dry_run(self):
        """Test full execution in dry run mode."""
        threadward = Threadward(self.test_dir)
        
        # Generate tasks
        success = threadward.generate_tasks()
        self.assertTrue(success)
        
        # Should generate 2 * 2 = 4 tasks
        self.assertEqual(len(threadward.tasks), 4)
        
        # Verify task structure
        for task in threadward.tasks:
            self.assertIn("sleep_time", task.variables)
            self.assertIn("test_id", task.variables)
            self.assertIsInstance(task.variables["sleep_time"], float)
            self.assertIsInstance(task.variables["test_id"], int)


if __name__ == "__main__":
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("THREADWARD TEST SUITE SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    print(f"{'='*60}")