#!/usr/bin/env python3
import unittest
import sys

def main():
    """
    Discovers and runs all tests in the slm_emergent_ai/tests directory.
    Exits with status code 0 for success, 1 for failure.
    """
    # Create a TestLoader instance
    loader = unittest.TestLoader()

    # Discover tests in the specified directory
    # The start_dir should be relative to the repository root where this script is located.
    # Assuming 'slm_emergent_ai' is a subdirectory in the root.
    test_dir = 'slm_emergent_ai/tests'
    suite = loader.discover(start_dir=test_dir, pattern='test_*.py')

    # Create a TextTestRunner instance
    # Verbosity 2 provides more detailed output
    runner = unittest.TextTestRunner(verbosity=2)

    # Run the tests
    result = runner.run(suite)

    # Exit with appropriate status code
    if result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
```
