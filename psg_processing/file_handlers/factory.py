"""
Factory for creating file handlers based on file type.
"""

from .mat_handler import MATHandler
from .edf_handler import EDFHandler
from .h5_handler import H5Handler
from .wfdb_handler import WFDBHandler
from .dreamt_csv_handler import DreamtCSVHandler

    
# Handler based on file extension
handlers = {
        "mat": MATHandler,
        "rec": EDFHandler,
        "edf": EDFHandler,
        "h5": H5Handler,
        "hea": WFDBHandler,
    }
# Dataset-specific CSV handlers
csv_handlers = {
        "DREAMT": DreamtCSVHandler,
        # Future: "OTHER_DATASET": OtherCSVHandler,
    }


def get_handler(dataset_name, psg_ext):
    """
    Get the appropriate handler for a file based on its extension and dataset context.
    
    Args:
        filepath: Path to the file to determine handler for
        
    Returns:
        FileHandler instance or None if no handler supports the file
    """
    psg_ext_lower = psg_ext.lower().split('.')[-1]
    
    # Handle CSV files with dataset-specific logic
    if psg_ext_lower == "csv":
        if dataset_name and dataset_name in csv_handlers:
            handler_class = csv_handlers[dataset_name]
            return handler_class
        raise ValueError(f"No CSV handler found for dataset: {dataset_name}")
    
    # Handle other file types
    if psg_ext_lower in handlers:
        return handlers[psg_ext_lower]
    raise ValueError(f"No handler found for file extension: {psg_ext_lower}")
