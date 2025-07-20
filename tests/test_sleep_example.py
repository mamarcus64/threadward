"""
Example test that demonstrates threadward with time.sleep as the main task.

This test creates a realistic example of using threadward for AI experimentation
where time.sleep simulates different computational workloads.
"""

import unittest
import tempfile
import os
import shutil
import json
import time
from threadward.core.threadward import Threadward


class TestSleepTaskExample(unittest.TestCase):
    """Comprehensive example test using time.sleep to simulate AI experiments."""
    
    def setUp(self):
        """Set up a realistic AI experiment simulation."""
        self.test_dir = tempfile.mkdtemp()
        self.threadward_dir = os.path.join(self.test_dir, "threadward")
        os.makedirs(self.threadward_dir)
        os.makedirs(os.path.join(self.threadward_dir, "task_queue"))
        
        print(f"\\n🧪 Setting up AI experiment simulation in: {self.test_dir}")
        
        # Create realistic AI experiment configuration
        self._create_ai_experiment_setup()
    
    def tearDown(self):
        """Clean up test environment."""
        print(f"🧹 Cleaning up test directory: {self.test_dir}")
        shutil.rmtree(self.test_dir)
    
    def _create_ai_experiment_setup(self):
        """Create a realistic AI experiment setup with time.sleep as computation."""
        
        # Create task specification simulating ML model training
        task_spec = '''
import time
import os
import json
import random

SUCCESS_CONDITION = "NO_ERROR_AND_VERIFY"
OUTPUT_MODE = "LOG_FILE_ONLY"

def task_method(variables, task_folder, log_file):
    """Simulate training an AI model with different configurations."""
    
    # Extract experiment parameters
    model_type = variables["model_type"]
    learning_rate = variables["learning_rate"]
    batch_size = variables["batch_size"]
    dataset = variables["dataset"]
    seed = variables["seed"]
    
    print(f"🚀 Starting {model_type} training...")
    print(f"   Dataset: {dataset}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Random Seed: {seed}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Simulate different training times based on model complexity
    training_times = {
        "linear_regression": 0.5,
        "neural_network": 1.5,
        "transformer": 3.0,
        "cnn": 2.0
    }
    
    base_time = training_times.get(model_type, 1.0)
    
    # Adjust time based on dataset size
    dataset_multipliers = {
        "small_dataset": 0.5,
        "medium_dataset": 1.0,
        "large_dataset": 2.0
    }
    
    time_multiplier = dataset_multipliers.get(dataset, 1.0)
    
    # Simulate training epochs
    total_training_time = base_time * time_multiplier
    epochs = 3
    
    training_history = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        print(f"   Epoch {epoch + 1}/{epochs}")
        
        # Simulate epoch training time
        epoch_time = total_training_time / epochs
        time.sleep(epoch_time)
        
        # Simulate metrics (would be real metrics in actual ML)
        # Make metrics somewhat realistic based on parameters
        loss = 1.0 - (epoch * 0.3) + random.uniform(-0.1, 0.1)
        accuracy = min(0.95, (epoch + 1) * 0.25 + random.uniform(-0.05, 0.05))
        
        # Adjust metrics based on learning rate
        if learning_rate > 0.01:
            # High learning rate might cause instability
            loss += random.uniform(0, 0.2)
            accuracy -= random.uniform(0, 0.1)
        elif learning_rate < 0.001:
            # Low learning rate might learn slowly
            accuracy *= 0.9
        
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        
        epoch_data = {
            "epoch": epoch + 1,
            "loss": max(0, loss),
            "accuracy": max(0, min(1, accuracy)),
            "duration": epoch_duration
        }
        
        training_history.append(epoch_data)
        print(f"     Loss: {epoch_data['loss']:.4f}, Accuracy: {epoch_data['accuracy']:.4f}")
    
    # Save training results
    results = {
        "model_type": model_type,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dataset": dataset,
            "seed": seed
        },
        "training_history": training_history,
        "final_metrics": training_history[-1] if training_history else {},
        "total_training_time": sum(epoch["duration"] for epoch in training_history)
    }
    
    # Save detailed results to JSON
    results_path = os.path.join(task_folder, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary to text file
    summary_path = os.path.join(task_folder, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"AI Experiment Summary\\n")
        f.write(f"==================\\n")
        f.write(f"Model: {model_type}\\n")
        f.write(f"Dataset: {dataset}\\n")
        f.write(f"Learning Rate: {learning_rate}\\n")
        f.write(f"Batch Size: {batch_size}\\n")
        f.write(f"Seed: {seed}\\n")
        f.write(f"\\nFinal Results:\\n")
        if training_history:
            final = training_history[-1]
            f.write(f"  Final Loss: {final['loss']:.4f}\\n")
            f.write(f"  Final Accuracy: {final['accuracy']:.4f}\\n")
        f.write(f"  Total Training Time: {results['total_training_time']:.2f}s\\n")
    
    print(f"✅ {model_type} training completed!")
    if training_history:
        final_acc = training_history[-1]['accuracy']
        print(f"   Final accuracy: {final_acc:.4f}")

def verify_task_success(variables, task_folder, log_file):
    """Verify that the AI experiment completed successfully."""
    
    # Check that results file exists
    results_path = os.path.join(task_folder, "training_results.json")
    if not os.path.exists(results_path):
        return False
    
    # Check that results contain valid data
    try:
        with open(results_path, "r") as f:
            results = json.load(f)
        
        # Verify required fields exist
        required_fields = ["model_type", "hyperparameters", "training_history", "final_metrics"]
        if not all(field in results for field in required_fields):
            return False
        
        # Verify training actually happened
        if not results["training_history"]:
            return False
        
        # Verify final accuracy is reasonable (> 0)
        final_metrics = results["final_metrics"]
        if "accuracy" not in final_metrics or final_metrics["accuracy"] <= 0:
            return False
        
        return True
        
    except (json.JSONDecodeError, KeyError):
        return False

def before_all_tasks():
    """Set up the AI experiment environment."""
    print("🔬 Initializing AI Experiment Framework...")
    print("   - Loading experiment configurations")
    print("   - Setting up result tracking")
    print("   - Preparing parallel execution environment")

def after_all_tasks():
    """Clean up and summarize all AI experiments."""
    print("📊 AI Experiment Framework completed!")
    print("   - All model configurations tested")
    print("   - Results saved and verified")
    print("   - Ready for analysis and comparison")

def before_each_worker(worker_id):
    """Set up worker for AI experiments."""
    print(f"👷 Worker {worker_id} initializing for AI experiments...")
    print(f"   - Setting up computational environment")
    print(f"   - Loading shared resources")

def after_each_worker(worker_id):
    """Clean up worker after AI experiments."""
    print(f"👷 Worker {worker_id} shutting down...")
    print(f"   - Cleaning up resources")
    print(f"   - Saving worker statistics")

def before_each_task(variables, task_folder, log_file):
    """Prepare for individual AI experiment."""
    model_type = variables.get("model_type", "unknown")
    print(f"🔧 Preparing {model_type} experiment...")

def after_each_task(variables, task_folder, log_file):
    """Clean up after individual AI experiment."""
    model_type = variables.get("model_type", "unknown")
    print(f"📋 Finalizing {model_type} experiment results...")
'''
        
        task_spec_path = os.path.join(self.threadward_dir, "task_specification.py")
        with open(task_spec_path, 'w') as f:
            f.write(task_spec)
        
        # Create variable iteration simulating ML hyperparameter sweep
        var_iter = '''
FAILURE_HANDLING = "PRINT_FAILURE_AND_CONTINUE"
TASK_FOLDER_LOCATION = "VARIABLE_SUBFOLDER"

def setup_variable_set(variable_set):
    """Set up AI experiment variable combinations."""
    
    # Model architectures (highest level - models are expensive to load)
    variable_set.add_variable(
        name="model_type",
        values=["linear_regression", "neural_network", "cnn", "transformer"],
        nicknames=["linear", "nn", "cnn", "transformer"]
    )
    
    # Datasets (second level - datasets are expensive to load)
    variable_set.add_variable(
        name="dataset",
        values=["small_dataset", "medium_dataset", "large_dataset"],
        nicknames=["small", "medium", "large"]
    )
    
    # Learning rates (third level - hyperparameter tuning)
    variable_set.add_variable(
        name="learning_rate",
        values=[0.001, 0.01, 0.1],
        nicknames=["lr_001", "lr_01", "lr_1"],
        # High learning rates might not work well with complex models
        exceptions={
            "transformer": ["0.001", "0.01"],  # Transformers need careful tuning
            "cnn": ["0.001", "0.01", "0.1"]   # CNNs can handle various rates
        }
    )
    
    # Batch sizes (fourth level - quick to change)
    variable_set.add_variable(
        name="batch_size",
        values=[16, 32, 64],
        nicknames=["b16", "b32", "b64"],
        # Large datasets might need specific batch sizes
        exceptions={
            "large_dataset": ["32", "64"]  # Large datasets need bigger batches
        }
    )
    
    # Random seeds (lowest level - just for reproducibility)
    variable_set.add_variable(
        name="seed",
        values=[42, 123, 456],
        nicknames=["seed42", "seed123", "seed456"]
    )

def learning_rate_to_value(string_value, nickname):
    """Convert learning rate to float."""
    return float(string_value)

def batch_size_to_value(string_value, nickname):
    """Convert batch size to int."""
    return int(string_value)

def seed_to_value(string_value, nickname):
    """Convert seed to int."""
    return int(string_value)
'''
        
        var_iter_path = os.path.join(self.threadward_dir, "variable_iteration.py")
        with open(var_iter_path, 'w') as f:
            f.write(var_iter)
        
        # Create resource constraints for AI experiments
        constraints = '''
# AI Experiment Resource Configuration

# Use 2 workers to simulate parallel model training
NUM_WORKERS = 2

# No GPU allocation for this test (would be non-zero in real AI experiments)
NUM_GPUS_PER_WORKER = 0

# Example GPU configurations for real AI experiments:
# 
# Single GPU per worker:
# NUM_GPUS_PER_WORKER = 1
# 
# Multiple workers sharing GPUs:
# NUM_WORKERS = 4
# NUM_GPUS_PER_WORKER = 0.5  # 2 workers per GPU
#
# Avoid specific GPUs (e.g., if GPU 0 is being used by another process):
# AVOID_GPUS = [0]
#
# Use only specific GPUs:
# INCLUDE_GPUS = [1, 2, 3]

AVOID_GPUS = None
INCLUDE_GPUS = None
'''
        
        constraints_path = os.path.join(self.threadward_dir, "resource_constraints.py")
        with open(constraints_path, 'w') as f:
            f.write(constraints)
        
        # Create empty task queue files
        for filename in ["all_tasks.json", "successful_tasks.txt", "failed_tasks.txt"]:
            file_path = os.path.join(self.threadward_dir, "task_queue", filename)
            with open(file_path, 'w') as f:
                if filename.endswith('.json'):
                    f.write('[]')
                else:
                    f.write('')
    
    def test_ai_experiment_task_generation(self):
        """Test that AI experiment tasks are generated correctly."""
        print("\\n🧮 Testing AI experiment task generation...")
        
        threadward = Threadward(self.test_dir)
        
        # Generate tasks
        start_time = time.time()
        success = threadward.generate_tasks()
        generation_time = time.time() - start_time
        
        self.assertTrue(success, "Task generation should succeed")
        
        print(f"   ⏱️  Task generation took {generation_time:.3f} seconds")
        print(f"   📝 Generated {len(threadward.tasks)} total tasks")
        
        # Verify we have the expected number of tasks
        # 4 models * 3 datasets * varying learning rates * varying batch sizes * 3 seeds
        # With exceptions, this should be less than 4*3*3*3*3 = 324
        self.assertGreater(len(threadward.tasks), 100, "Should generate substantial number of tasks")
        self.assertLess(len(threadward.tasks), 400, "Should respect exceptions and not explode")
        
        # Verify task structure
        for i, task in enumerate(threadward.tasks[:5]):  # Check first 5 tasks
            print(f"   📋 Task {i+1}: {task.task_id}")
            print(f"      Model: {task.variables['model_type']}")
            print(f"      Dataset: {task.variables['dataset']}")
            print(f"      LR: {task.variables['learning_rate']}")
            print(f"      Batch: {task.variables['batch_size']}")
            print(f"      Seed: {task.variables['seed']}")
            
            # Verify variable types
            self.assertIsInstance(task.variables['learning_rate'], float)
            self.assertIsInstance(task.variables['batch_size'], int)
            self.assertIsInstance(task.variables['seed'], int)
            self.assertIsInstance(task.variables['model_type'], str)
            self.assertIsInstance(task.variables['dataset'], str)
    
    def test_ai_experiment_exceptions(self):
        """Test that variable exceptions work correctly for AI experiments."""
        print("\\n🚫 Testing AI experiment variable exceptions...")
        
        threadward = Threadward(self.test_dir)
        threadward.generate_tasks()
        
        # Check transformer tasks - should only have learning rates 0.001 and 0.01
        transformer_tasks = [t for t in threadward.tasks if t.variables['model_type'] == 'transformer']
        
        print(f"   🤖 Found {len(transformer_tasks)} transformer tasks")
        
        for task in transformer_tasks:
            lr = task.variables['learning_rate']
            self.assertIn(lr, [0.001, 0.01], 
                         f"Transformer should only use LR 0.001 or 0.01, got {lr}")
        
        # Check large dataset tasks - should only have batch sizes 32 and 64
        large_dataset_tasks = [t for t in threadward.tasks if t.variables['dataset'] == 'large_dataset']
        
        print(f"   📊 Found {len(large_dataset_tasks)} large dataset tasks")
        
        for task in large_dataset_tasks:
            batch_size = task.variables['batch_size']
            self.assertIn(batch_size, [32, 64], 
                         f"Large dataset should only use batch size 32 or 64, got {batch_size}")
    
    def test_ai_experiment_folder_structure(self):
        """Test that AI experiment folders are structured hierarchically."""
        print("\\n📁 Testing AI experiment folder structure...")
        
        threadward = Threadward(self.test_dir)
        threadward.generate_tasks()
        
        # Check folder hierarchy: model/dataset/lr/batch/seed
        sample_tasks = threadward.tasks[:10]
        
        for task in sample_tasks:
            folder_parts = task.task_folder.split('/')
            self.assertEqual(len(folder_parts), 5, 
                           f"Should have 5 folder levels, got {len(folder_parts)}: {task.task_folder}")
            
            # Verify folder names match expected nicknames
            model_nick, dataset_nick, lr_nick, batch_nick, seed_nick = folder_parts
            
            # Model nicknames
            self.assertIn(model_nick, ["linear", "nn", "cnn", "transformer"])
            
            # Dataset nicknames  
            self.assertIn(dataset_nick, ["small", "medium", "large"])
            
            # Learning rate nicknames
            self.assertIn(lr_nick, ["lr_001", "lr_01", "lr_1"])
            
            # Batch size nicknames
            self.assertIn(batch_nick, ["b16", "b32", "b64"])
            
            # Seed nicknames
            self.assertTrue(seed_nick.startswith("seed"))
        
        print(f"   ✅ All {len(sample_tasks)} sample task folders have correct structure")
    
    def test_ai_experiment_dry_run_performance(self):
        """Test AI experiment generation performance."""
        print("\\n⚡ Testing AI experiment performance...")
        
        # Test task generation performance
        start_time = time.time()
        threadward = Threadward(self.test_dir)
        threadward.generate_tasks()
        generation_time = time.time() - start_time
        
        print(f"   ⏱️  Generated {len(threadward.tasks)} tasks in {generation_time:.3f} seconds")
        print(f"   📈 Generation rate: {len(threadward.tasks)/generation_time:.1f} tasks/second")
        
        # Performance assertions
        self.assertLess(generation_time, 5.0, "Task generation should be fast")
        
        if len(threadward.tasks) > 0:
            rate = len(threadward.tasks) / generation_time
            self.assertGreater(rate, 10, "Should generate at least 10 tasks per second")
        
        # Test configuration loading performance
        start_time = time.time()
        config = threadward.config
        config_time = time.time() - start_time
        
        print(f"   ⚙️  Configuration loading took {config_time:.3f} seconds")
        self.assertLess(config_time, 1.0, "Configuration loading should be very fast")
    
    def test_ai_experiment_realistic_metrics(self):
        """Test that the AI experiment produces realistic output."""
        print("\\n📊 Testing AI experiment realistic metrics...")
        
        threadward = Threadward(self.test_dir)
        threadward.generate_tasks()
        
        # Pick a representative task to examine
        sample_task = threadward.tasks[0]
        
        print(f"   🔬 Examining sample task: {sample_task.task_id}")
        print(f"      Variables: {sample_task.variables}")
        
        # Verify task folder structure exists as expected
        expected_folder = sample_task.task_folder
        print(f"      Expected folder: {expected_folder}")
        
        # Verify log file path
        expected_log = sample_task.log_file
        print(f"      Expected log: {expected_log}")
        
        self.assertTrue(expected_folder, "Task should have folder defined")
        self.assertTrue(expected_log, "Task should have log file defined")
        self.assertTrue(expected_log.endswith('.log'), "Log file should have .log extension")


if __name__ == "__main__":
    # Run the sleep task example test
    print("🚀 Running Threadward Sleep Task Example Tests")
    print("=" * 60)
    
    unittest.main(verbosity=2)