# `threadward`: Parallel Processing for Generalizable AI Experimentation in Python

`threadward` is a lightweight, cross-platform package that enables you to run custom scripts while iterating over combinations of script variables. Just define your task, declare the variables you want to iterate over, set your GPU and CPU constraints, and `threadward` will handle the rest -- automatically spinning up Python subprocess workers, creating task queues, and allocating jobs to workers.

## Platform Support

`threadward` works on:
- **Linux** (fully supported, including GPU allocation)
- **macOS** (fully supported, including GPU allocation) 
- **Windows** (fully supported, including GPU allocation)

**GPU Support**: Optional GPU allocation works on any system with CUDA-compatible GPUs. Set `NUM_GPUS_PER_WORKER = 0` to run CPU-only on any platform.

## Table of Contents
- [Platform Support](#platform-support)
- [Installing `threadward`](#installing-threadward)
- [Quick Start](#quick-start)
- [Local Package Imports](#local-package-imports)
- [Configuration Options](#configuration-options)
- [Variable Specifications](#variable-specifications)
- [Implementation Details](#implementation-details)
- [Troubleshooting](#troubleshooting)

## Installing `threadward`

Install `threadward` directly from GitHub:
```bash
pip install git+https://github.com/mamarcus64/threadward.git
```

## Quick Start

### 1. Initialize a Configuration File

Create a new threadward configuration file:
```bash
threadward init my_experiment
```

This creates `threadward_my_experiment.py` in your current directory.

Or create a default configuration file:
```bash
threadward init
```

This creates `threadward_run.py` in your current directory.

### 2. Edit Your Configuration

Open `threadward_my_experiment.py` (or `threadward_run.py`) and implement your Runner class:

```python
import argparse
import threadward

class My_experimentRunner(threadward.Threadward):
    def __init__(self, debug=False, results_folder="threadward_results"):
        super().__init__(debug=debug, results_folder=results_folder)
        self.set_constraints(
            NUM_WORKERS=1,
            NUM_GPUS_PER_WORKER=0,
            TASK_TIMEOUT=30  # Timeout in seconds (-1 for no timeout)
        )
    
    def task_method(self, variables, task_folder, log_file):
        # Your task logic here
        print(f"Running with variables: {variables}")
    
    def verify_task_success(self, variables, task_folder, log_file):
        return True
        
    def setup_variable_set(self, variable_set):
        variable_set.add_variable(
            name="learning_rate",
            values=[0.001, 0.01, 0.1],
            nicknames=["lr_001", "lr_01", "lr_1"]
        )
        variable_set.add_variable(
            name="batch_size",
            values=[16, 32, 64]
        )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run threadward experiments')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output for troubleshooting')
    parser.add_argument('--results-folder', default='threadward_results',
                       help='Name of the results folder (default: threadward_results)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    runner = My_experimentRunner(debug=args.debug, results_folder=args.results_folder)
    runner.run()
```

### 3. Run Your Experiment

```bash
python threadward_my_experiment.py
```

Or if you used the default name:
```bash
python threadward_run.py
```

#### CLI Options

You can customize execution with command line arguments:

```bash
# Enable debug output for troubleshooting
python threadward_my_experiment.py --debug

# Use a custom results folder
python threadward_my_experiment.py --results-folder my_custom_results

# Combine options
python threadward_my_experiment.py --debug --results-folder experiment_2024
```

**Available CLI Arguments:**
- `--debug`: Enable detailed debug output to help troubleshoot worker communication issues
- `--results-folder`: Specify the name of the results folder (default: "threadward_results")

That's it! `threadward` will create task folders, manage workers, and execute your tasks across all variable combinations.

### Interactive Commands During Execution

While your experiment is running, you can use interactive commands to monitor progress and control execution:

- **`show` or `s`** - Display current execution statistics including:
  - Elapsed time and estimated remaining time
  - Task progress (total, non-skipped, skipped, succeeded, failed, remaining)
  - Worker status and resource usage
  - CPU, memory, and GPU utilization

- **`help` or `h`** - Show available commands

- **`quit`, `q`, or `exit`** - Gracefully stop execution
  - Workers will complete their current tasks before shutting down
  - Progress is saved (use `EXISTING_FOLDER_HANDLING = "SKIP"` to resume)

Example interaction:
```
> show

============================================================
THREADWARD EXECUTION STATUS
============================================================
Elapsed Time: 00:15:42
Estimated Remaining: 01:23:18
Average Time per Task: 12.35s

Tasks:
  Total:              128
  Skipped:             32
  Non-Skipped Total:   96
  Succeeded:           60 (62.5%)
  Failed:               2 (2.1%)
  Remaining:           34 (35.4%)

Workers (4 total):
  Worker 0: [BUSY] task_000077
    CPU: 87.2% (max: 92.1%)
    Memory: 2341MB (max: 2456MB)
    GPU Memory: 4096MB (max: 4096MB)
    Succeeded: 19, Failed: 0
  Worker 1: [IDLE]
    Succeeded: 18, Failed: 1
  ...
============================================================

> quit
Stopping execution gracefully...
Workers will finish their current tasks before shutting down.
```

## Local Package Imports

One of the key advantages of the new structure is seamless local package imports. If you have a project structure like:

```
YOUR_PROJECT/
├── local_package/
│   ├── __init__.py
│   ├── models.py
│   └── utils.py
├── data/
└── threadward_my_experiment.py
```

You can directly import from your local packages in the configuration file:

```python
import threadward
from local_package.models import MyModel
from local_package.utils import process_data

class Runner(threadward.Threadward):
    def __init__(self, debug=False, results_folder="threadward_results"):
        super().__init__(debug=debug, results_folder=results_folder)
        
    def task_method(self, variables, task_folder, log_file):
        model = MyModel(variables['model_type'])
        data = process_data(variables['dataset'])
        # ... rest of your task
    
    def verify_task_success(self, variables, task_folder, log_file):
        return True
        
    def setup_variable_set(self, variable_set):
        pass

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run threadward experiments')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output for troubleshooting')
    parser.add_argument('--results-folder', default='threadward_results',
                       help='Name of the results folder (default: threadward_results)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    runner = Runner(debug=args.debug, results_folder=args.results_folder)
    runner.run()
```

This works because:
1. The configuration file runs from your project directory
2. Python automatically includes the current directory in the import path
3. All local packages are available for import without any special setup

**Key Point**: If `YOUR_DIRECTORY/local_package` exists, then `import local_package.local_file` will work seamlessly in your generated configuration file.

## Configuration Options

Your configuration file supports the following options:

### Task Control
- `SUCCESS_CONDITION`: How to determine task success
  - `"NO_ERROR_AND_VERIFY"` (default): No errors AND `verify_task_success` returns True
  - `"NO_ERROR_ONLY"`: Only check for no errors
  - `"VERIFY_ONLY"`: Only use `verify_task_success` 
  - `"ALWAYS_SUCCESS"`: Always consider successful

- `OUTPUT_MODE`: How to handle task output
  - `"LOG_FILE_ONLY"` (default): Only log to file
  - `"CONSOLE_ONLY"`: Only print to console
  - `"LOG_FILE_AND_CONSOLE"`: Both file and console

### Resource Management
- `NUM_WORKERS`: Number of parallel workers (default: 1)
- `NUM_GPUS_PER_WORKER`: GPUs per worker (default: 0)
- `AVOID_GPUS`: List of GPU IDs to avoid (default: None)
- `INCLUDE_GPUS`: List of GPU IDs to use exclusively (default: None)

### Task Organization
- `TASK_FOLDER_LOCATION`: How to organize task folders
  - `"VARIABLE_SUBFOLDER"` (default): Nested folders by variable
  - `"VARIABLE_UNDERSCORE"`: Single folder with underscore-separated names

- `EXISTING_FOLDER_HANDLING`: What to do with existing task folders
  - `"SKIP"` (default): Skip tasks with existing folders
  - `"OVERWRITE"`: Delete existing folders and rerun
  - `"QUIT"`: Stop execution if any folders exist

- `FAILURE_HANDLING`: How to handle task failures
  - `"PRINT_FAILURE_AND_CONTINUE"` (default): Print failure and continue
  - `"SILENT_CONTINUE"`: Continue silently
  - `"STOP_EXECUTION"`: Stop on first failure

### Execution Control
- `TASK_TIMEOUT`: Timeout in seconds for task completion (default: 30)
  - Set to `-1` for no timeout
  - Tasks that exceed this timeout will be marked as failed

## Variable Specifications

### Hierarchical Variables

`threadward` uses **hierarchical variable retention** - variables defined first are considered "higher level" and workers retain these values while iterating through lower-level combinations.

```python
def setup_variable_set(variable_set):
    # First variable (highest level)
    variable_set.add_variable(
        name="model",
        values=["gpt2", "bert-base", "llama-7b"],
        nicknames=["GPT2", "BERT", "Llama"]
    )
    
    # Second level - will iterate for each model
    variable_set.add_variable(
        name="dataset", 
        values=["dataset1", "dataset2"]
    )
    
    # Lowest level - will iterate for each model/dataset combo
    variable_set.add_variable(
        name="seed",
        values=[42, 123, 456]
    )
```

With this setup, a worker will load one model and use it for all dataset/seed combinations before moving to the next model.

### Variable Exceptions

You can specify exceptions to exclude certain combinations:

```python
variable_set.add_variable(
    name="batch_size",
    values=[16, 32, 64, 128],
    exceptions={
        "0.1": ["16", "32"]  # High learning rate only with small batches
    }
)
```

### Value Converters

Convert string values to objects by defining converter functions:

```python
def model_to_value(string_value, nickname):
    if nickname == 'BERT':
        return AutoModel.from_pretrained(string_value)
    else:
        return AutoModelForCausalLM.from_pretrained(string_value)
```

## Implementation Details

### File Structure

After running your configuration file, `threadward` creates:

```
YOUR_PROJECT/
├── threadward_my_experiment.py  # Your configuration file
├── threadward_results/          # All results stored here (configurable via --results-folder)
│   ├── task_queue/              # Created during execution
│   │   ├── all_tasks.json
│   │   ├── successful_tasks.txt
│   │   └── failed_tasks.txt
│   └── [task_folders]/          # Individual task results
│       ├── GD/lr_001/16/seed_0/
│       ├── GD/lr_001/16/seed_1/
│       └── ...
```

### Worker Management

- Workers are Python subprocesses that communicate via file-based IPC for reliability
- Each worker can be assigned specific GPUs via `CUDA_VISIBLE_DEVICES`
- Workers persist across multiple tasks to retain loaded models/data
- Uses the same Python executable as the parent process (inherits conda/virtualenv environment)
- Automatic cleanup and resource management
- Robust error handling and process monitoring

### Task Scheduling

1. Generate all task combinations from your variable specifications
2. Create task queue and worker processes
3. Assign tasks to workers based on hierarchical variable retention
4. Monitor execution and handle failures according to your configuration
5. Clean up resources when complete

The system is designed to be robust, resumable (via `EXISTING_FOLDER_HANDLING = "SKIP"`), and efficient for iterative experimentation workflows.

## Troubleshooting

### Debug Mode

If you encounter issues with worker processes or task execution, enable debug mode:

```bash
python threadward_my_experiment.py --debug
```

Debug mode provides detailed output including:
- Worker initialization and startup messages
- Task assignment and acknowledgment tracking
- Inter-process communication details
- Task completion status and timing
- Error details and process state information

### Common Issues

**Workers fail to start:**
- Check that your Python environment has all required dependencies
- Verify GPU availability if using `NUM_GPUS_PER_WORKER > 0`
- Enable debug mode to see detailed startup messages

**Tasks marked as failed despite completing:**
- Increase `TASK_TIMEOUT` if tasks need more time to complete
- Check your `verify_task_success` implementation
- Review task log files in the individual task folders

**Environment issues:**
- `threadward` uses the same Python executable as the parent process
- This automatically inherits your current conda/virtualenv environment
- Custom environment variables (like `CUDA_VISIBLE_DEVICES`) are properly set per worker