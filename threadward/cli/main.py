"""Main CLI entry point for threadward."""

import argparse
import sys
import os
from .init_command import init_command
from .run_command import run_command


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="threadward: Parallel Processing for Generalizable AI Experimentation in Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  threadward init          Initialize a new threadward project in current directory
  threadward run           Run threadward in current directory
  threadward run --help    Show detailed run options
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    init_parser = subparsers.add_parser(
        "init", 
        help="Initialize a new threadward project"
    )
    init_parser.add_argument(
        "--path",
        default=".",
        help="Path to initialize project (default: current directory)"
    )
    
    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run threadward execution"
    )
    run_parser.add_argument(
        "--path",
        default=".",
        help="Path to project directory (default: current directory)"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate tasks and show statistics without executing"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "init":
            init_command(args.path)
        elif args.command == "run":
            run_command(args.path, dry_run=args.dry_run)
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()