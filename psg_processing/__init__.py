"""
PSG Processing Package

A comprehensive toolkit for processing polysomnography (PSG) datasets.
Supports multiple file formats (EDF, H5, CSV, WFDB) with signal processing,
filtering, resampling, and automated sleep stage analysis.
"""

from .file_handlers import (
    FileHandler, EDFHandler, H5Handler, CSVHandler, 
    WFDBHandler, FileHandlerFactory
)
from .core import Dataset_Explorer, DatasetProcessor, SignalProcessor
from .utils import LoggingManager

__version__ = "1.0.0"
__author__ = "Linda Beland"

__all__ = [    
    # File Handlers
    'FileHandler',
    'EDFHandler',
    'H5Handler', 
    'CSVHandler',
    'WFDBHandler',
    'FileHandlerFactory',
    
    # Core Processing
    'Dataset_Explorer',
    'DatasetProcessor',
    'SignalProcessor',
    
    # Utils
    'LoggingManager'
]
