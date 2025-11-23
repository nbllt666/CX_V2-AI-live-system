#!/usr/bin/env python3
"""
Test runner script for the CX-V2 AI Live System.
This script runs all tests and generates coverage reports.
"""

import subprocess
import sys
import os

def run_tests():
    """Run all tests with coverage."""
    print("Running tests...")
    
    # Run tests with coverage
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v", 
        "--cov=.", 
        "--cov-report=html:htmlcov", 
        "--cov-report=term-missing"
    ])
    
    return result.returncode

if __name__ == "__main__":
    # Change to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Run tests
    exit_code = run_tests()
    
    # Exit with the same code as the tests
    sys.exit(exit_code)