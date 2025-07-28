"""Main CLI entry point for threadward."""

import argparse
import sys
import os
from .init_command import init_command


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="threadward: Parallel Processing for Generalizable AI Experimentation in Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  threadward init                 Create threadward_run.py with Runner class
  threadward init experiment_1    Create threadward_experiment_1.py with Experiment_1Runner class
  threadward init loop_2 --path /path/to/project    Create runner in specific directory
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init subcommand
    init_parser = subparsers.add_parser("init", help="Initialize a new threadward configuration")
    init_parser.add_argument(
        "name",
        nargs="?",
        help="Optional name for the threadward runner (creates threadward_{name}.py with {name}Runner class, or threadward_run.py with Runner class if no name)"
    )
    init_parser.add_argument(
        "--path",
        default=".",
        help="Path to create configuration file (default: current directory)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        if args.command == "init":
            init_command(args.name, args.path)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()