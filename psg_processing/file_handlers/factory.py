"""
Factory for creating file handlers based on file type.
"""

from .edf_handler import EDFHandler
from .h5_handler import H5Handler
from .csv_handler import CSVHandler
from .wfdb_handler import WFDBHandler


class FileHandlerFactory:
    """Factory class to get the appropriate file handler."""

    def __init__(self):
        self.handlers = {
            ".edf": EDFHandler,
            ".h5": H5Handler,
            ".csv": CSVHandler,
            ".hea": WFDBHandler,
        }

    def get_handler(self, logger, filepath):
        """Get the appropriate handler for a file."""
        for ext, handler in self.handlers.items():
            if ext in filepath.lower():
                return handler(logger)
        return None

    def get_file_type(self, filepath):
        """Get the file type identifier."""
        handler = self.get_handler(filepath)
        if handler:
            return handler.file_extension[1:]  # Remove the dot
        return None
