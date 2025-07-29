"""Interactive CLI handler for threadward execution."""

import threading
import sys
import time
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
                command = input("> ").strip().lower()
                
                if command in self.commands:
                    self.commands[command]()
                elif command:
                    print(f"Unknown command: '{command}'. Type 'help' for available commands.")
                
            except EOFError:
                # Handle Ctrl+D
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C
                print("\nUse 'quit' command to exit gracefully.")
            except Exception:
                # Silently ignore other exceptions in the input thread
                pass
    
    def show_stats(self):
        """Display current execution statistics."""
        stats = self.threadward.get_stats()
        
        print("\n" + "="*60)
        print("THREADWARD EXECUTION STATUS")
        print("="*60)
        
        # Time information
        elapsed = stats['elapsed_time']
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Elapsed Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        if stats['estimated_remaining_time'] > 0:
            remaining = stats['estimated_remaining_time']
            hours, remainder = divmod(int(remaining), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Estimated Remaining: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        print(f"Average Time per Task: {stats['avg_time_per_task']:.2f}s")
        
        # Task information
        print(f"\nTasks:")
        print(f"  Total:              {stats['tasks']['total']:>6}")
        print(f"  Non-Skipped Total:  {stats['tasks']['non_skipped_total']:>6}")
        print(f"  Skipped:            {stats['tasks']['skipped']:>6}")
        print(f"  Succeeded:          {stats['tasks']['succeeded']:>6} ({stats['tasks']['succeeded']/max(stats['tasks']['non_skipped_total'], 1)*100:.1f}%)")
        print(f"  Failed:             {stats['tasks']['failed']:>6} ({stats['tasks']['failed']/max(stats['tasks']['non_skipped_total'], 1)*100:.1f}%)")
        print(f"  Remaining:          {stats['tasks']['remaining']:>6} ({stats['tasks']['remaining']/max(stats['tasks']['non_skipped_total'], 1)*100:.1f}%)")
        
        # Worker information
        print(f"\nWorkers ({len(stats['workers'])} total):")
        for worker_stats in stats['workers']:
            status = worker_stats['status']
            worker_id = worker_stats['worker_id']
            
            # Format status with color/symbol
            if status == 'busy':
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
    
    def show_help(self):
        """Display available commands."""
        print("\nAvailable commands:")
        print("  show, s  - Display current execution statistics")
        print("  help, h  - Show this help message")
        print("  quit, q  - Gracefully stop execution and exit")
        print("  exit     - Same as quit\n")
    
    def quit_execution(self):
        """Gracefully stop the execution."""
        print("\nStopping execution gracefully...")
        print("Workers will finish their current tasks before shutting down.")
        self.threadward.should_stop = True
        self.running = False