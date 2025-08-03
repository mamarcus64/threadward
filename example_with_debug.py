#!/usr/bin/env python3
"""
Example of how to add --debug flag to your threadward runner script.

Add this code to the top of your threadward_run.py or similar file.
"""

import argparse
import sys
import os

# Add the current directory to path so we can import threadward
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from threadward.core.threadward import Threadward

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run threadward experiments')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output for troubleshooting')
    parser.add_argument('--timezone', default='US/Pacific',
                       help='Timezone for display timestamps (default: US/Pacific). Examples: US/Eastern, US/Central, Europe/London, UTC')
    return parser.parse_args()

def main():
    """Main function - replace this with your existing threadward setup."""
    args = parse_args()
    
    # Your existing threadward setup code goes here
    # For example:
    
    # Import your config module
    # import your_config_module
    
    # Create threadward instance with debug flag and timezone
    # threadward = Threadward(".", your_config_module, debug=args.debug, timezone=args.timezone)
    # threadward.run()
    
    print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    print(f"Timezone: {args.timezone}")
    print("Replace this main() function with your actual threadward setup")

if __name__ == "__main__":
    main()