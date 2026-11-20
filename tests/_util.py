"""Lets module-level test_* functions be collected by `unittest discover`."""
import unittest


def function_tests(namespace: dict):
    """Return a load_tests hook that runs every test_* function of the module."""
    functions = [value for name, value in namespace.items()
                 if name.startswith("test_") and callable(value)]

    def load_tests(loader, tests, pattern):
        suite = unittest.TestSuite()
        suite.addTests(unittest.FunctionTestCase(function) for function in functions)
        return suite

    return load_tests
