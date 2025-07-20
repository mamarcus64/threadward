#!/usr/bin/env python3
"""
Comprehensive test runner for threadward package.

This script runs all tests and provides detailed output and debugging information.
It includes tests with time.sleep as the main task to demonstrate threadward functionality.
"""

import unittest
import sys
import os
import time
import traceback
from io import StringIO

# Add the parent directory to the Python path to import threadward
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """Run all threadward tests with verbose output and debugging."""
    
    print("=" * 80)
    print("THREADWARD COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print(f"Test directory: {os.path.dirname(__file__)}")
    print(f"Package directory: {os.path.dirname(os.path.dirname(__file__))}")
    print("-" * 80)
    
    start_time = time.time()
    
    # Discover and load all tests
    loader = unittest.TestLoader()
    test_dir = os.path.dirname(__file__)
    
    # Load tests from all test modules
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # Create a test runner with maximum verbosity
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    # Run the tests
    print("Starting test execution...\n")
    result = runner.run(suite)
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print detailed summary
    print("\n" + "=" * 80)
    print("TEST EXECUTION SUMMARY")
    print("=" * 80)
    
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    # Print detailed failure information
    if result.failures:
        print(f"\n{'='*60}")
        print("DETAILED FAILURE ANALYSIS")
        print(f"{'='*60}")
        
        for i, (test, traceback_str) in enumerate(result.failures, 1):
            print(f"\nFailure {i}: {test}")
            print("-" * 60)
            print("Traceback:")
            print(traceback_str)
            
            # Try to extract the assertion error message
            lines = traceback_str.split('\n')
            for line in lines:
                if 'AssertionError' in line:
                    print(f"Assertion: {line.strip()}")
                    break
    
    # Print detailed error information
    if result.errors:
        print(f"\n{'='*60}")
        print("DETAILED ERROR ANALYSIS")
        print(f"{'='*60}")
        
        for i, (test, traceback_str) in enumerate(result.errors, 1):
            print(f"\nError {i}: {test}")
            print("-" * 60)
            print("Traceback:")
            print(traceback_str)
            
            # Try to extract the error message
            lines = traceback_str.split('\n')
            for line in lines:
                if any(exc in line for exc in ['Error:', 'Exception:', 'ImportError:']):
                    print(f"Error: {line.strip()}")
                    break
    
    # Print test coverage information
    print(f"\n{'='*60}")
    print("TEST COVERAGE ANALYSIS")
    print(f"{'='*60}")
    
    test_categories = {
        'Core functionality': 0,
        'CLI commands': 0,
        'Integration tests': 0,
        'Sleep task tests': 0
    }
    
    # Count tests by category (simple heuristic based on test names)
    for test_case in result.testsRun:
        test_name = str(test_case) if hasattr(test_case, '__str__') else ''
        if 'test_core' in test_name.lower():
            test_categories['Core functionality'] += 1
        elif 'test_cli' in test_name.lower():
            test_categories['CLI commands'] += 1
        elif 'integration' in test_name.lower():
            test_categories['Integration tests'] += 1
        elif 'sleep' in test_name.lower():
            test_categories['Sleep task tests'] += 1
    
    for category, count in test_categories.items():
        if count > 0:
            print(f"{category}: {count} tests")
    
    # Print recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if result.failures or result.errors:
        print("❌ Some tests failed. Please review the detailed analysis above.")
        if result.failures:
            print("   - Check assertion failures for logic errors")
        if result.errors:
            print("   - Check error traces for implementation issues")
    else:
        print("✅ All tests passed successfully!")
        print("   - threadward package appears to be working correctly")
        print("   - All core functionality is operational")
        print("   - CLI commands are functional")
        print("   - Sleep task integration tests passed")
    
    # Performance analysis
    if execution_time > 30:
        print(f"⚠️  Tests took {execution_time:.1f} seconds - consider optimizing for faster feedback")
    elif execution_time < 5:
        print(f"⚡ Tests completed quickly in {execution_time:.1f} seconds - good for rapid development")
    else:
        print(f"✅ Test execution time ({execution_time:.1f}s) is reasonable")
    
    print(f"\n{'='*80}")
    
    # Return success/failure for CI/CD
    return len(result.failures) == 0 and len(result.errors) == 0


def run_specific_test_category(category):
    """Run tests from a specific category."""
    
    print(f"Running {category} tests only...\n")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if category == "core":
        import test_core
        suite.addTests(loader.loadTestsFromModule(test_core))
    elif category == "cli":
        import test_cli
        suite.addTests(loader.loadTestsFromModule(test_cli))
    else:
        print(f"Unknown category: {category}")
        return False
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0


def run_performance_tests():
    """Run performance-focused tests to check threadward efficiency."""
    
    print("Running performance tests...\n")
    
    # Import performance test classes
    from test_core import TestSleepTask, TestIntegrationSleepTest
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add specific performance-related tests
    suite.addTest(TestSleepTask('test_task_generation'))
    suite.addTest(TestSleepTask('test_variable_conversions'))
    suite.addTest(TestIntegrationSleepTest('test_full_execution_dry_run'))
    
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    end_time = time.time()
    
    print(f"\nPerformance test execution time: {end_time - start_time:.2f} seconds")
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    """Main test runner entry point."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Threadward Test Runner")
    parser.add_argument(
        "--category", 
        choices=["all", "core", "cli", "performance"],
        default="all",
        help="Run specific test category (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable extra verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Verbose mode enabled")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Run appropriate tests based on category
    if args.category == "all":
        success = run_all_tests()
    elif args.category == "performance":
        success = run_performance_tests()
    else:
        success = run_specific_test_category(args.category)
    
    # Exit with appropriate code for CI/CD
    sys.exit(0 if success else 1)