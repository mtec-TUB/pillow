"""
Registry system for datasets.
"""

import importlib
from typing import Dict, Type, Optional
from datasets.base import BaseDataset


class DatasetRegistry:
    """Registry for datasets"""
    
    _datasets: Dict[str, Type[BaseDataset]] = {}
    # Maps dataset name -> module name (e.g. "bdsp"), discovered without importing
    # the module, so a dataset's script (and its imports) is only loaded on demand.
    _modules: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, dataset_class: Type[BaseDataset]):
        """Register a dataset"""
        cls._datasets[name] = dataset_class

    @classmethod
    def register_module(cls, name: str, module_name: str):
        """Record which module defines a dataset, without importing it"""
        cls._modules[name] = module_name

    @classmethod
    def get_dataset(cls, name: str):
        """Get a dataset class by name, importing its module on demand"""
        if name not in cls._datasets and name in cls._modules:
            importlib.import_module(f"datasets.{cls._modules[name]}")

        if name not in cls._datasets:
            raise ValueError(f"Dataset '{name}' is not registered. Please choose from: "
                             f"{', '.join(cls.list_datasets())}")
        return cls._datasets[name]
    
    @classmethod
    def list_datasets(cls):
        """List all registered datasets"""
        return sorted(set(cls._datasets) | set(cls._modules))


def register_dataset(name: str):
    """Decorator to register a dataset"""
    def decorator(dataset: Type[BaseDataset]):
        DatasetRegistry.register(name, dataset)
        return dataset
    return decorator


def get_dataset(name: str):
    """Get a dataset class by name"""
    return DatasetRegistry.get_dataset(name)
