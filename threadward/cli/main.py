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
  threadward init experiment_1    Create tw_experiment_1.py configuration file
  threadward init loop_2 --path /path/to/project    Create configuration in specific directory
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init subcommand
    init_parser = subparsers.add_parser("init", help="Initialize a new threadward configuration")
    init_parser.add_argument(
        "name",
        help="Name for the threadward configuration (creates tw_{name}.py)"
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