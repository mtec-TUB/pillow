"""
Utilities for PSG data processing.
"""

import os
import logging
import glob

# alignment options
class Alignment(Enum):
    """Options for aligning signal and annotation lengths at front and/or end."""
    MATCH_SHORTER = "match shorter"     # no padding, but cropping if necessary
    MATCH_LONGER = "match longer"   # no cropping, but padding with custom values
    MATCH_SIGNAL = "match signal length"    # pad/crop to signal length
    MATCH_ANNOT = "match annot length"  # pad/crop to annotation length

@dataclass
class ProcessorConfig:
    dataset: str
    base_data_dir: str
    data_dir: Path
    ann_dir: Path
    output_dir: Path
    action: str
    resample: int
    filter: bool
    channels: list[str]
    epoch_duration: int
    overwrite: bool
    allow_missing: bool
    num_jobs: int
    rm_move: bool
    rm_unk: bool
    n_wake_epochs: int
    alignment: Alignment
    pad_values: list
    min_sleep_epochs: int


class LoggingManager:
    """
    A class to manage all logging operations for PSG dataset processing.

    This class centralizes logging setup, file handler management, and
    provides clean interfaces for different logging scenarios.
    """

    def __init__(self, level=logging.INFO, format=None, date_format=None):
        self.level = level
        self.format = format or "%(asctime)s - %(levelname)s - %(message)s"
        self.date_format = date_format or "%Y-%m-%d %H:%M:%S"
        self.log_filename = "processing.log"


    def cleanup_file_handlers(self,logger):
        """
        Remove all file handlers from logger while keeping console handlers.
        This prevents file handle leaks while preserving console output.
        """
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()  # Properly close the file handle
                logger.removeHandler(handler)

    def setup_channel_file_logging(self,logger, output_dir):
        """
        Set up or update the file handler for channel-specific logging.
        Keeps the console handler unchanged.

        Args:
            logger: The logger instance
            output_dir: Directory where the log file should be created

        Returns:
            Path to the created log file
        """
        log_file_path = os.path.join(output_dir, self.log_filename)

        # Remove any existing file handlers but keep console handlers
        self.cleanup_file_handlers(logger)

        # Add new file handler for this channel
        file_handler = logging.FileHandler(log_file_path)

        formatter = logging.Formatter(
            self.format, datefmt=self.date_format
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return log_file_path

    def setup_logger(self, dir_name=None, overwrite=False):
        """
        Create a logger with both console and optional file output.

        Args:
            dir: Optional path to folder to delete old logging files

        Returns:
            Configured logger instance
        """

        # delete all exisiting log_files in output folder if overwrite True
        if overwrite and dir_name:
            log_files = glob.glob(os.path.join(dir_name, '**',self.log_filename), recursive=True
            )
            for f in log_files:
                os.remove(f)
            
        logger = logging.getLogger()

        # Avoid adding handlers multiple times
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(self.level)

        # Create detailed formatter
        formatter = logging.Formatter(
            fmt=self.format, datefmt=self.date_format
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger
