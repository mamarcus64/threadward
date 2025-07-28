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

This creates `tw_my_experiment.py` in your current directory.

### 2. Edit Your Configuration

Open `tw_my_experiment.py` and implement your task:

```python
import threadward

def task_method(variables, task_folder, log_file):
    # Your task logic here
    print(f"Running with variables: {variables}")
    
def setup_variable_set(variable_set):
    variable_set.add_variable(
        name="learning_rate",
        values=[0.001, 0.01, 0.1],
        nicknames=["lr_001", "lr_01", "lr_1"]
    )
    variable_set.add_variable(
        name="batch_size",
        values=[16, 32, 64]
    )

if __name__ == "__main__":
    threadward.run()
```

### 3. Run Your Experiment

```bash
python tw_my_experiment.py
```

That's it! `threadward` will create task folders, manage workers, and execute your tasks across all variable combinations.

## Local Package Imports

One of the key advantages of the new structure is seamless local package imports. If you have a project structure like:

```
YOUR_PROJECT/
├── local_package/
│   ├── __init__.py
│   ├── models.py
│   └── utils.py
├── data/
└── tw_my_experiment.py
```

You can directly import from your local packages in the configuration file:

```python
import threadward
from local_package.models import MyModel
from local_package.utils import process_data

def task_method(variables, task_folder, log_file):
    model = MyModel(variables['model_type'])
    data = process_data(variables['dataset'])
    # ... rest of your task

if __name__ == "__main__":
    threadward.run()
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
├── tw_my_experiment.py          # Your configuration file
├── task_queue/                  # Created during execution
│   ├── all_tasks.json
│   ├── successful_tasks.txt
│   └── failed_tasks.txt
├── worker_script.py             # Temporary worker script
└── [task_folders]/              # Individual task results
    ├── GD/lr_001/16/seed_0/
    ├── GD/lr_001/16/seed_1/
    └── ...
```

### Worker Management

- Workers are Python subprocesses that communicate via stdin/stdout
- Each worker can be assigned specific GPUs via `CUDA_VISIBLE_DEVICES`
- Workers persist across multiple tasks to retain loaded models/data
- Automatic cleanup and resource management

### Task Scheduling

1. Generate all task combinations from your variable specifications
2. Create task queue and worker processes
3. Assign tasks to workers based on hierarchical variable retention
4. Monitor execution and handle failures according to your configuration
5. Clean up resources when complete

The system is designed to be robust, resumable (via `EXISTING_FOLDER_HANDLING = "SKIP"`), and efficient for iterative experimentation workflows.