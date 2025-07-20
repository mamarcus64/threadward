"""Run command implementation for threadward CLI."""

import os
import sys
import threading
import time
from ..core.threadward import Threadward


def run_command(project_path: str = ".", dry_run: bool = False):
    """Run threadward execution.
    
    Args:
        project_path: Path to the project directory
        dry_run: If True, only generate tasks and show statistics without executing
    """
    project_path = os.path.abspath(project_path)
    threadward_path = os.path.join(project_path, "threadward")
    
    # Validate project structure
    if not os.path.exists(threadward_path):
        print(f"Error: No threadward directory found at {threadward_path}")
        print("Run 'threadward init' first to initialize a project.")
        return
    
    required_files = [
        "task_specification.py",
        "resource_constraints.py", 
        "variable_iteration.py"
    ]
    
    for filename in required_files:
        file_path = os.path.join(threadward_path, filename)
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            return
    
    # Create Threadward instance
    threadward = Threadward(project_path)
    
    if dry_run:
        # Dry run mode - just generate tasks and show stats
        print("Dry run mode: Generating tasks without execution...")
        if threadward.generate_tasks():
            stats = {
                "total_tasks": len(threadward.tasks),
                "sample_tasks": threadward.tasks[:3] if threadward.tasks else []
            }
            print(f"[SUCCESS] Generated {stats['total_tasks']} tasks")
            
            if stats['sample_tasks']:
                print("\nSample tasks:")
                for i, task in enumerate(stats['sample_tasks']):
                    print(f"  {i+1}. {task.task_id}: {task.variables}")
                if len(threadward.tasks) > 3:
                    print(f"  ... and {len(threadward.tasks) - 3} more tasks")
        return
    
    # Full execution mode
    print("Starting threadward execution...")
    
    # Start execution in a separate thread so we can handle interactive commands
    execution_thread = threading.Thread(target=threadward.run)
    execution_thread.daemon = True
    execution_thread.start()
    
    # Interactive command loop
    try:
        _interactive_loop(threadward)
    except KeyboardInterrupt:
        print("\nShutting down...")
        threadward.should_stop = True
    
    # Wait for execution to complete
    execution_thread.join()


def _interactive_loop(threadward: Threadward):
    """Run the interactive command loop while threadward is executing."""
    print("\nInteractive commands:")
    print("  show (s) - Show execution status")
    print("  quit (q) - Stop execution gracefully")
    print("  add (a) - Add one more worker")
    print("  remove <worker_id> (r <worker_id>) - Remove specified worker")
    print("  help (h) - Show this help message")
    print()
    
    while threadward.is_running or not threadward.start_time:
        try:
            # Wait for user input with a timeout so we can check if execution finished
            cmd = _get_input_with_timeout("threadward> ", 1.0)
            if cmd is None:
                continue
                
            cmd = cmd.strip().lower()
            
            if cmd in ['show', 's']:
                _show_status(threadward)
            elif cmd in ['quit', 'q']:
                print("Stopping execution...")
                threadward.should_stop = True
                break
            elif cmd in ['add', 'a']:
                _add_worker(threadward)
            elif cmd.startswith('remove') or cmd.startswith('r'):
                parts = cmd.split()
                if len(parts) >= 2:
                    try:
                        worker_id = int(parts[1])
                        _remove_worker(threadward, worker_id)
                    except ValueError:
                        print("Error: Worker ID must be an integer")
                else:
                    print("Error: Please specify worker ID (e.g., 'remove 0')")
            elif cmd in ['help', 'h']:
                print("Available commands:")
                print("  show (s) - Show execution status")
                print("  quit (q) - Stop execution gracefully") 
                print("  add (a) - Add one more worker")
                print("  remove <worker_id> (r <worker_id>) - Remove specified worker")
                print("  help (h) - Show this help message")
            elif cmd:
                print(f"Unknown command: {cmd}. Type 'help' for available commands.")
                
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nUse 'quit' to stop execution gracefully.")


def _get_input_with_timeout(prompt: str, timeout: float):
    """Get user input with a timeout. Returns None if timeout occurs."""
    import select
    import sys
    
    print(prompt, end='', flush=True)
    
    # Check if input is available
    if select.select([sys.stdin], [], [], timeout)[0]:
        return input()
    else:
        print('\r', end='', flush=True)  # Clear the prompt line
        return None


def _show_status(threadward: Threadward):
    """Show current execution status."""
    stats = threadward.get_stats()
    
    print("\n" + "="*60)
    print("THREADWARD EXECUTION STATUS")
    print("="*60)
    
    # Time information
    elapsed_hours = stats['elapsed_time'] // 3600
    elapsed_minutes = (stats['elapsed_time'] % 3600) // 60
    elapsed_seconds = stats['elapsed_time'] % 60
    
    print(f"Time since start: {int(elapsed_hours):02d}:{int(elapsed_minutes):02d}:{int(elapsed_seconds):02d}")
    print(f"Average time per task: {stats['avg_time_per_task']:.2f} seconds")
    
    if stats['tasks']['remaining'] > 0:
        remaining_hours = stats['estimated_remaining_time'] // 3600
        remaining_minutes = (stats['estimated_remaining_time'] % 3600) // 60
        print(f"Estimated time remaining: {int(remaining_hours):02d}:{int(remaining_minutes):02d}")
    
    # Task information
    print(f"\nTasks completed: {stats['tasks']['completed']}")
    print(f"Tasks failed: {stats['tasks']['failed']}")
    print(f"Tasks remaining: {stats['tasks']['remaining']}")
    print(f"Total tasks: {stats['tasks']['total']}")
    
    # Worker information
    print(f"\nWorkers ({len(stats['workers'])}):")
    print("-" * 60)
    for worker_stats in stats['workers']:
        worker_id = worker_stats['worker_id']
        pid = worker_stats['pid'] or 'N/A'
        status = worker_stats['status']
        current_task = worker_stats['current_task'] or 'None'
        
        print(f"Worker {worker_id} (PID: {pid})")
        print(f"  Status: {status}")
        print(f"  Current task: {current_task}")
        print(f"  Completed: {worker_stats['total_tasks_completed']}, Failed: {worker_stats['total_tasks_failed']}")
        
        # Resource usage
        cpu_current = worker_stats['cpu_percent']['current']
        cpu_max = worker_stats['cpu_percent']['max']
        mem_current = worker_stats['memory_mb']['current']
        mem_max = worker_stats['memory_mb']['max']
        
        print(f"  CPU: {cpu_current:.1f}% (max: {cpu_max:.1f}%)")
        print(f"  RAM: {mem_current:.1f}MB (max: {mem_max:.1f}MB)")
        
        if worker_stats['gpu_ids']:
            gpu_current = worker_stats['gpu_memory_mb']['current']
            gpu_max = worker_stats['gpu_memory_mb']['max']
            print(f"  GPU Memory: {gpu_current:.1f}MB (max: {gpu_max:.1f}MB)")
            print(f"  Assigned GPUs: {worker_stats['gpu_ids']}")
        
        print()
    
    print("="*60)


def _add_worker(threadward: Threadward):
    """Add a new worker to the execution."""
    print("Adding worker functionality not yet implemented.")
    # TODO: Implement dynamic worker addition


def _remove_worker(threadward: Threadward, worker_id: int):
    """Remove a worker from execution."""
    if worker_id < 0 or worker_id >= len(threadward.workers):
        print(f"Error: Invalid worker ID {worker_id}. Valid range: 0-{len(threadward.workers)-1}")
        return
    
    worker = threadward.workers[worker_id]
    if worker.status == "stopped":
        print(f"Worker {worker_id} is already stopped.")
        return
    
    print(f"Stopping worker {worker_id}...")
    worker.shutdown()
    print(f"Worker {worker_id} stopped.")