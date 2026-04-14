"""
Registry system for datasets.
"""

from typing import Dict, Type, Optional
from datasets.base import BaseDataset


class DatasetRegistry:
    """Registry for datasets"""
    
    _datasets: Dict[str, Type[BaseDataset]] = {}
    
    @classmethod
    def register(cls, name: str, dataset_class: Type[BaseDataset]):
        """Register a dataset"""
        cls._datasets[name] = dataset_class
    
    @classmethod
    def get_dataset(cls, name: str):
        """Get a dataset class by name"""
        if name not in cls._datasets:
            raise ValueError(f"Dataset '{name}' is not registered. Please choose from: "
                             f"{', '.join(cls.list_datasets())}")
        return cls._datasets[name]
    
    @classmethod
    def list_datasets(cls):
        """List all registered datasets"""
        return list(cls._datasets.keys())


def register_dataset(name: str):
    """Decorator to register a dataset"""
    def decorator(dataset: Type[BaseDataset]):
        DatasetRegistry.register(name, dataset)
        return dataset
    return decorator


def get_dataset(name: str):
    """Get a dataset class by name"""
    return DatasetRegistry.get_dataset(name)
