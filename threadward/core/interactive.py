"""Interactive CLI handler for threadward execution."""

import threading
import sys
import time
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .threadward import Threadward


class InteractiveHandler:
    """Handles interactive commands during threadward execution."""
    
    def __init__(self, threadward_instance: 'Threadward'):
        """Initialize the interactive handler.
        
        Args:
            threadward_instance: The main Threadward instance to interact with
        """
        self.threadward = threadward_instance
        self.commands = {
            'show': self.show_stats,
            's': self.show_stats,
            'help': self.show_help,
            'h': self.show_help,
            'quit': self.quit_execution,
            'q': self.quit_execution,
            'exit': self.quit_execution,
        }
        self.running = False
        self.input_thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the interactive command handler in a separate thread."""
        self.running = True
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        print("\nInteractive mode enabled. Type 'help' for available commands.\n")
    
    def stop(self):
        """Stop the interactive command handler."""
        self.running = False
    
    def _input_loop(self):
        """Main input loop for handling user commands."""
        while self.running and self.threadward.is_running:
            try:
                # Use a non-blocking approach to check for input
                command = input("> ").strip()
                
                # Parse command and arguments
                parts = command.split()
                if not parts:
                    continue
                    
                cmd = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                if cmd in self.commands:
                    if cmd in ['show', 's']:
                        self.show_stats(args)
                    else:
                        self.commands[cmd]()
                elif cmd:
                    print(f"Unknown command: '{cmd}'. Type 'help' for available commands.")
                
            except EOFError:
                # Handle Ctrl+D
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C
                print("\nUse 'quit' command to exit gracefully.")
            except Exception:
                # Silently ignore other exceptions in the input thread
                pass
    
    def show_stats(self, args=None):
        """Display current execution statistics.
        
        Args:
            args: List of worker IDs to display specifically, or None for default display
        """
        stats = self.threadward.get_stats()
        
        # Get current time for display
        current_time_str = datetime.fromtimestamp(stats['current_time']).strftime("%m/%d/%y at %I:%M %p")
        
        print("\n" + "="*60)
        print(f"THREADWARD EXECUTION STATUS - {current_time_str}")
        print("="*60)
        
        # Time information
        elapsed = stats['elapsed_time']
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Format start time
        start_time_str = datetime.fromtimestamp(stats['start_time']).strftime("%m/%d/%y at %I:%M %p")
        print(f"Elapsed Time: {hours:02d}:{minutes:02d}:{seconds:02d} (Started {start_time_str})")
        
        if stats['estimated_remaining_time'] > 0:
            remaining = stats['estimated_remaining_time']
            hours, remainder = divmod(int(remaining), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # Calculate estimated completion time
            est_completion_time = stats['current_time'] + stats['estimated_remaining_time']
            est_completion_str = datetime.fromtimestamp(est_completion_time).strftime("%m/%d/%y at %I:%M %p")
            print(f"Estimated Remaining: {hours:02d}:{minutes:02d}:{seconds:02d} (Est. {est_completion_str})")
        
        print(f"Average Time per Task: {stats['avg_time_per_task']:.2f}s")
        
        # Task information
        print(f"\nTasks:")
        print(f"  Total:              {stats['tasks']['total']:>6}")
        print(f"  Skipped:            {stats['tasks']['skipped']:>6}")
        print(f"  Non-Skipped Total:  {stats['tasks']['non_skipped_total']:>6}")
        print(f"  Succeeded:          {stats['tasks']['succeeded']:>6} ({stats['tasks']['succeeded']/max(stats['tasks']['non_skipped_total'], 1)*100:.1f}%)")
        print(f"  Failed:             {stats['tasks']['failed']:>6} ({stats['tasks']['failed']/max(stats['tasks']['non_skipped_total'], 1)*100:.1f}%)")
        print(f"  Remaining:          {stats['tasks']['remaining']:>6} ({stats['tasks']['remaining']/max(stats['tasks']['non_skipped_total'], 1)*100:.1f}%)")
        
        # Worker information
        workers_to_show = self._get_workers_to_display(stats['workers'], args)
        
        if args:
            print(f"\nWorkers (showing {len(workers_to_show)} of {len(stats['workers'])} total):")
        else:
            print(f"\nWorkers ({len(stats['workers'])} total):")
            
        for i, worker_stats in enumerate(workers_to_show):
            # Add ellipsis separator if needed
            if isinstance(worker_stats, str) and worker_stats == "...":
                print("  ...")
                continue
                
            status = worker_stats['status']
            worker_id = worker_stats['worker_id']
            
            # Format status with color/symbol
            if status == 'busy':
                if worker_stats.get('current_task_info'):
                    task_info = worker_stats['current_task_info']
                    task_id = task_info['task_id']
                    runtime = task_info['runtime_seconds']
                    
                    # Format runtime as MM:SS or H:MM:SS
                    if runtime < 3600:  # Less than 1 hour
                        minutes, seconds = divmod(int(runtime), 60)
                        runtime_str = f"{minutes}:{seconds:02d}"
                    else:  # 1 hour or more
                        hours, remainder = divmod(int(runtime), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        runtime_str = f"{hours}:{minutes:02d}:{seconds:02d}"
                    
                    status_str = f"[BUSY] Task({task_id}, running for {runtime_str})"
                else:
                    status_str = f"[BUSY] {worker_stats['current_task']}"
            elif status == 'idle':
                status_str = "[IDLE]"
            else:
                status_str = f"[{status.upper()}]"
            
            print(f"  Worker {worker_id}: {status_str}")
            
            # Show resource usage if available
            if worker_stats['cpu_percent']['current'] > 0:
                print(f"    CPU: {worker_stats['cpu_percent']['current']:.1f}% (max: {worker_stats['cpu_percent']['max']:.1f}%)")
            if worker_stats['memory_mb']['current'] > 0:
                print(f"    Memory: {worker_stats['memory_mb']['current']:.0f}MB (max: {worker_stats['memory_mb']['max']:.0f}MB)")
            if worker_stats['gpu_memory_mb']['current'] > 0:
                print(f"    GPU Memory: {worker_stats['gpu_memory_mb']['current']:.0f}MB (max: {worker_stats['gpu_memory_mb']['max']:.0f}MB)")
            
            print(f"    Succeeded: {worker_stats['total_tasks_succeeded']}, Failed: {worker_stats['total_tasks_failed']}")
        
        print("="*60 + "\n")
    
    def _get_workers_to_display(self, all_workers, args):
        """Get the list of workers to display based on arguments.
        
        Args:
            all_workers: List of all worker stats
            args: List of arguments from command line
            
        Returns:
            List of worker stats to display (may include "..." string for ellipsis)
        """
        if args:
            # Parse specific worker IDs
            worker_ids = []
            try:
                for arg in args:
                    # Handle comma-separated values: "5,6,7"
                    if ',' in arg:
                        worker_ids.extend([int(x.strip()) for x in arg.split(',') if x.strip()])
                    else:
                        worker_ids.append(int(arg))
            except ValueError as e:
                print(f"Error: Invalid worker ID format. Use numbers only (e.g., 'show 5' or 'show 5,6,7')")
                return all_workers[:4] if len(all_workers) > 4 else all_workers
            
            # Filter workers by requested IDs
            workers_by_id = {w['worker_id']: w for w in all_workers}
            result = []
            for worker_id in worker_ids:
                if worker_id in workers_by_id:
                    result.append(workers_by_id[worker_id])
                else:
                    print(f"Warning: Worker {worker_id} not found")
            return result
        
        else:
            # Default behavior: show first 2, last 2 if more than 4 workers
            if len(all_workers) <= 4:
                return all_workers
            else:
                # Show first 2, ellipsis, last 2
                result = []
                result.extend(all_workers[:2])  # First 2
                result.append("...")  # Ellipsis marker
                result.extend(all_workers[-2:])  # Last 2
                return result
    
    def show_help(self):
        """Display available commands."""
        print("\nAvailable commands:")
        print("  show, s      - Display current execution statistics")
        print("  show N       - Display statistics for worker N only")
        print("  show N M...  - Display statistics for workers N, M, etc.")
        print("  show N,M,P   - Display statistics for workers N, M, P (comma-separated)")
        print("  help, h      - Show this help message")
        print("  quit, q      - Gracefully stop execution and exit")
        print("  exit         - Same as quit")
        print("\nNote: By default, 'show' displays first 2 and last 2 workers if more than 4 exist\n")
    
    def quit_execution(self):
        """Gracefully stop the execution."""
        print("\nStopping execution gracefully...")
        print("Workers will finish their current tasks before shutting down.")
        self.threadward.should_stop = True
        self.running = False