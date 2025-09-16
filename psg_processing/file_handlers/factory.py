"""
Factory for creating file handlers based on file type.
"""

from .edf_handler import EDFHandler
from .h5_handler import H5Handler
from .wfdb_handler import WFDBHandler
from .dreamt_csv_handler import DreamtCSVHandler


class FileHandlerFactory:
    """Factory class to get the appropriate file handler."""

    def __init__(self, dataset_name=None):
        self.dataset_name = dataset_name
        self.handlers = {
            ".edf": EDFHandler,
            ".h5": H5Handler,
            ".hea": WFDBHandler,
        }
        
        # Dataset-specific CSV handlers
        self.csv_handlers = {
            "DREAMT - Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology": DreamtCSVHandler,
            # Future: "OTHER_DATASET": OtherCSVHandler,
        }

    def get_handler(self, logger, filepath):
        """
        Get the appropriate handler for a file based on its extension and dataset context.
        
        Args:
            logger: Logger instance to pass to the handler
            filepath: Path to the file to determine handler for
            
        Returns:
            FileHandler instance or None if no handler supports the file
        """
        filepath_lower = filepath.lower()
        
        # Handle CSV files with dataset-specific logic
        if ".csv" in filepath_lower:
            if self.dataset_name and self.dataset_name in self.csv_handlers:
                handler_class = self.csv_handlers[self.dataset_name]
                handler = handler_class(logger)
                # Verify the handler actually supports this file
                if handler.supports_format(filepath):
                    return handler
            return None  # No appropriate CSV handler found
        
        # Handle other file types
        for ext, handler_class in self.handlers.items():
            if ext in filepath_lower:
                return handler_class(logger)
        return None
