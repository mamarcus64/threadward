#!/usr/bin/env python3
import argparse
import threadward
import time

def slow_converter(string_value, nickname):
    """Slow converter that takes 5 seconds."""
    print(f"Starting slow conversion of {string_value}...")
    time.sleep(5)  # Simulate slow conversion
    print(f"Finished slow conversion of {string_value}")
    return f"CONVERTED:{string_value}"

class TimingTestRunner(threadward.Threadward):
    def __init__(self, debug=False, results_folder="timing_results"):
        super().__init__(debug=debug, results_folder=results_folder)
        self.set_constraints(
            NUM_WORKERS=2,  # Use 2 workers to test parallel assignment
            NUM_GPUS_PER_WORKER=0,
            TASK_TIMEOUT=30
        )
    
    def task_method(self, variables, task_folder, log_file):
        print(f"Task executing with model: {variables.model}")
        time.sleep(1)  # Quick task
        print(f"Task completed with model: {variables.model}")
    
    def verify_task_success(self, variables, task_folder, log_file):
        return True
    
    def setup_variable_set(self, variable_set):
        # Add a variable with slow converter
        variable_set.add_variable(
            name="model",
            values=["model1", "model2", "model3", "model4"],
            to_value=slow_converter  # This takes 5 seconds per conversion
        )

def parse_args():
    parser = argparse.ArgumentParser(description='Test task assignment timing')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output for troubleshooting')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    runner = TimingTestRunner(debug=args.debug)
    runner.run()