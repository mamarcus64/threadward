# `threadward`: Parallel Processing for Generalizable AI Experimentation in Python
`threadward` is a lightweight package that enables you to run custom scripts while iterating over combinations of script variables. Just define the script, declare the variables you want to iterate over, set your GPU and CPU constraints, and `threadward` will do the rest -- automatically spinning up different Python subprocess workers, creating a `Task` queue and allocating jobs to workers, and retaining important variables that persist between each `Task`.

From a high-level, users first initialize a `threadward` project via `threadward init`, which creates a template for the user to implement that defines a single Python `Task` and a set of variables to run this `Task` over. Then, `threadward run` starts the project, where Python subprocess workers are created via the `subprocess` library; the main `threadward` script handles assigning `Task`s to these workers and monitoring their status.

## Table of Contents
- [Installing `threadward`](#installing-threadward)
- [Usage](#usage)
- [`Task` specifications](#task-specifications)
- [`VariableSet` specifications](#variableset-specifications)
- [Constraint specifications](#constraint-specifications)
- [Implementation Details](#implementation-details)

## Installing `threadward`

`threadward` is meant to be executed on a Linux operating system that uses NVIDIA GPUs and will likely crash in other execution environments. Additionally, `threadward` is designed to be run in a conda environment. To install `threadward` to your conda environment, run:
```bash
pip install git+https://github.com/mamarcus64/threadward.git
```

## Usage

To begin using `threadward`, you must first initialize it:
```bash
cd YOUR_DIRECTORY
threadward init
```
This creates the folder `YOUR_DIRECTORY/threadward` with this structure:
```
YOUR_DIRECTORY/
├── EXISTING_CODE
└── threadward/
  ├── task_specification.py
  ├── resource_constraints.py
  ├── variable_iteration.py
  └── task_queue/ 
    ├── all_tasks.json (empty until threadward run)
    ├── successful_tasks.txt (empty until threadward run)
    └── failed_tasks.txt (empty until threadward run)
```

Fill in the following Python files:
- `task_specification.py`
    - **required:** implement the main task in `task_method`
    - **optional:** implement setup/teardown
        - `before/after_all_tasks`
        - `before/after_each_worker`
        - `before/after_each_task`
    - **optional:** implement task completion verification
        - `verify_task_success`
- `resource_constraints.py`
    - **optional:** define CPU and subprocess constraints
    - **optional:** define GPU constraints
- `variable_iteration.py`
    - **required:** define the sets of variables that you will iterate over
    - **optional:** log and handle `Task` failures
    - **optional:** save successful `Task` outputs

After filling out these files, simply run:
```bash
cd YOUR_DIRECTORY
threadward run
```
During execution, you can interact with `threadward` in the terminal with these commands:
- `show` (`s`)
    - Displays high-level information:
       - Time since start
       - Average time per task (and max time)
       - Estimated time remaining
       - Number of tasks completed (failed and successful)
       - Number of tasks remaining
    - For each subprocess, displays:
       - Worker ID (defined by `threadward`, incrementing from 0)
       - PID (defined by the Linux OS)
       - CPU Utilization (max and current)
       - RAM Usage (max and current)
       - GPU Memory Usage (max and current)
- `quit` (`q`)
    - Stops all new tasks **but does not interrupt any currently running tasks.** Calls all teardown methods normally.
- `add` (`a`)
    - Adds one more subprocess worker based on the original constraints.
- `remove [Worker ID]` (`r [Worker ID]`)
    -  Shuts down the specified subprocess **after the current worker's task finishes**. Calls relevant teardown methods normally.

### `threadward run` Execution Loop

1. Generate all tasks into `threadward/task_queue/all_tasks.json` based on `variable_iteration.py` and create a `Task` list. This JSON contains a list of dictionaries, with each entry containing a unique `Task` ID as well as the key-value pairs of each variable to both its string value and its nickname.
2. Call `before_all_tasks()`
3. Create workers based on `resource_contraints.py`
4. For each worker, call `before_each_worker(worker_id)`
5. While the `Task` queue is not empty:
    - Wait until the next free worker
    - Assign a `Task` to the freed worker that retains the most amount of hierarchical variables from the previous task
        - Call `before_each_task(variables, task_folder, log_file)`
        - The worker executes the `Task`
        - Call `after_each_task(variables, task_folder, log_file)`
        - Write to `threadward/task_queue/successful_tasks.txt` or `threadward/task_queue/failed_tasks.txt` based on completed `Task` status
6. For each worker, call `after_each_worker(worker_id)`
7. Call `after_all_tasks()`

## `Task` specifications

The `task_specification.py` file will contain the following method headers:
```
task_method(variables, task_folder, log_file)
before_all_tasks()
after_all_tasks()
before_each_worker(worker_id)
after_each_worker(worker_id)
before_each_task(variables, task_folder, log_file)
after_each_task(variables, task_folder, log_file)
verify_task_success(variables, task_folder, log_file)
```
- `variables`: a dictionary mapping from variable name to value as defined in `variable_iteration.py`.
    - Any modification to `variables` during `before_each_task` will be retained 
- `worker_id`: integer corresponding to `threadward` worker ID, incrementing from 0
- `task_folder`: the corresponding folder associated with the current `Task`
- `log_file`: location of the log file for the output generated during the `Task`.
   - NOTE: by default, **ALL TASK OUTPUT** (including errors) will be redirected to the log file instead of being printed to the terminal.

In addition, the following default settings can be changed. These are listed as constant variables in upper-case listed at the top of the `task_specification.py` file.
- `SUCCESS_CONDITION`: how to check if a `Task` successfully finished.
    - `NO_ERROR_AND_VERIFY` (default): The `Task` both finishes with no error and passes the `verify_task_success` method (which returns `True` by default).
    - `NO_ERROR_ONLY`: The `Task` finishes with no error. `verify_task_sucesss` is not called after `Task` completion.
    - `VERIFY_ONLY`: `verify_task_success` returns `True` after the `Task` is finished, regardless of if the `Task` threw an error.
    - `ALWAYS_SUCCESS`: Success is a mindset. If you never think you will fail, you won't.
- `OUTPUT_MODE`: how the `Task` logs output generated inside a `Task`.
    - `LOG_FILE_ONLY` (default): All output is logged to the log file and nothing is printed to the console.
    - `CONSOLE_ONLY`: All output is printed to the console only. This is NOT recommended for multiple workers.
    - `LOG_FILE_AND_CONSOLE`: All output is both logged to the log file and printed to the console.
    
## `VariableSet` specifications

`threadward` operates under the framework of **hierarchical variable retention**. When iterating over multiple variables, all variable values higher up on the hierarchy are retained when iterating over the variables underneath them. **This means that the order of the defined variables matters a lot.**

For example, imagine that I were running an NLP sentiment analysis pipeline with four variables, `model`, `dataset`, `test_set`, and `seed`, in that exact order. This means that once a worker subprocess loads a specific model, it will only receive tasks that use that model until all tasks using that model are finished -- i.e. every combination of `(dataset, test_set, seed)` defined in the `VariableSet`. Similarly, once a specific `model` loads a certain `dataset`, it will iterate over every `(test_set, seed)` combination before loading a new dataset.

In the `variable_iteration.py` file, there is one main function, `setup_variable_set`, where the implementation for the example above would be:

```python
def setup_variable_set(variable_set):
   # ORDER MATTERS! Start by inserting the highest-level variables.
   variable_set.add_variable(name="model", values=["gpt2", "google-bert/bert-base-uncased", "meta-llama/Llama-3.1-8B"], nicknames=["GPT2", "BERT", "Llama3.1"])

   # interaction defines how the combination of variables interact with the variable above it. Default is Cartesian product
   variable_set.add_variable(name="dataset", values=["HappyGoLucky", "FinalEmotions"], interaction="cartesian")

   # Use the exceptions keyword to pass in dictionaries for specific values
   variable_set.add_variable(name="test_set", values=["dev", "test"], exceptions={"HappyGoLucky": ["test"]})

   # Use the nicknames variable to set easier names for logging and folder creation.
   variable_set.add_variable(name="seed", values=list(range(0, 9999, 1111)), nicknames=list(range(0, 9, 1)))
```

By default, task folders are created based on the variable nicknames, in the order they are defined. If a nickname is not explicitly defined, the true value will be created. For our example, each `Task` folder would be located in `{model}/{dataset}/{test_set}/{seed}`.

As you might have noticed, if we want to retain the values of the `model` variable, right now that only corresponds to the string of the Hugging Face model ID. **The VariableSet class only accepts string (or string-castable) values as input. If you want to set a value of a variable, you must create and implement a method named `{variable_name}_to_value(string_value, nickname)`.** For our example, we would need to define:
```python
def model_to_value(string_value, nickname):
    if nickname == 'BERT':
        return AutoModel.from_pretrained(string_value)
    else:
        return AutoModelForCausalLM.from_pretrained(string_value)

def dataset_to_value(string_value, nickname):
    # assuming we load this function from our main source code
    return load_dataset(string_value)
```

If this function exists, all relevant variables will be passed through this function and the return value will be what is passed to the `Task`.

In addition, the following default settings can be changed. These are listed as constant variables in upper-case listed at the top of the `variable_iteration.py` file.
- `FAILURE_HANDLING`: what to do on a failed `Task`.
    - `PRINT_FAILURE_AND_CONTINUE` (default): prints the failed `Task` to the console and continues.
    - `SILENT_CONTINUE`: does not print anything and continues.
    - `STOP_EXECUTION`: stops execution after the first failed `Task` (but lets each worker finish its current `Task`).
- `TASK_FOLDER_LOCATION`: how to determine the `Task` folder location.
    - `VARIABLE_SUBFOLDER` (default): creates nested subfolders for each variable.
    - `VARIABLE_UNDERSCORE`: creates one folder per task where variables are concatenated via underscore.

## Constraint specifications

The `resource_constraints.py` file contains several important variables that determine how many workers are created and how progress is monitored. These include:

- `NUM_WORKERS`: The total number of workers (default: 1).
- `NUM_GPUS_PER_WORKER`: The total number of workers per GPU (default: 0).
    - This value can be less than one (i.e., 1/2), which means more than one worker will be assigned to each GPU.
        - NOTE: Python will not "play nice" if you assign multiple workers to the same GPU -- if multiple workers on the same GPU will cause an out of memory error, this will not be flagged before execution. Be careful if you decide to do this.
    - This value must have the `NUM_GPUS` value defined.
- `AVOID_GPUS`: list of GPUs (defined as their GPU number; i.e., '5', not 'cuda:5') to not allocate (default: None).
    - If there are less than `NUM_WORKERS * NUM_GPUS_PER_WORKER` GPUs available, throw an error.
- `INCLUDE_GPUS`: list of GPUs to use from. Only GPUs from this list will be used (default: None).
    - If `NUM_WORKERS * NUM_GPUS_PER_WORKER < INCLUDE_GPUS`, throw an error.
    - If there are overlapping GPUs in `AVOID_GPUS` and `INCLUDE_GPUS`, throw an error.

## Implementation Details

### Class Structures

There are several classes included in `threadward`, including `Worker`, `Task`, `VariableSet`, and `Threadward`, which is the main entry point. Most of the files created for the user template extend a `threadward` class as the user is implementing its functionality.

### Creating Subprocesses

`threadward` uses `subprocess` over `multiprocessing` because of GPU allocations. Because we are assigning different GPUs for each worker, spinning up a subprocess worker will involve the following steps:
1. Identify the conda environment used by the main `threadward` script
2. Activate this conda environment
3. Set active GPUs via the `CUDA_VISIBLE_DEVICES` environment variable
    - We set visible GPUs on an individual-worker level; for each worker, only `NUM_GPUS_PER_WORKER` devices should be visible

### Task Scheduling

Worker subprocesses and the main `threadward` script communicate via `STDIN` and `STDOUT`: as `STDIN` input, workers recieve the `Task` ID, where they can then lookup the specific variable values in `task_queue.json`. The worker itself writes the `Task` success or failure in the appropriate log `.txt` file and returns either a `0` (for success) or `1` (for failure). Receiving the `STDIN` of `SHUT_DOWN` is the signal to the worker to shut down the process.

### Monitoring GPU and CPU Usage

RAM, CPU Usage, and GPU Memory will be monitored using the `psutil` and `GPUtil` packages. They are included in the `pip` installation of `threadward`. When using `GPUtil`, the specific devices for each worker must be remembered. It is important to keep track of each worker's PID in order to track this information.