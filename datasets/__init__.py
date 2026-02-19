"""
Import all Dataset modules to auto-register them.
"""

import pkgutil
import importlib
import inspect

__all__ = []

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    module = importlib.import_module(f"{__name__}.{module_name}")

    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Only include classes defined in this module
        if obj.__module__ == module.__name__:
            globals()[name] = obj
            __all__.append(name)
