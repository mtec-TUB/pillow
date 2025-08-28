"""
Logging management utilities for PSG data processing.
"""

import os
import logging


class LoggingManager:
    """
    A class to manage all logging operations for PSG dataset processing.
    
    This class centralizes logging setup, file handler management, and
    provides clean interfaces for different logging scenarios.
    """
    
    @staticmethod
    def cleanup_file_handlers(logger):
        """
        Remove all file handlers from logger while keeping console handlers.
        This prevents file handle leaks while preserving console output.
        """
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()  # Properly close the file handle
                logger.removeHandler(handler)

    @staticmethod
    def setup_channel_file_logging(logger, output_dir, log_filename):
        """
        Set up or update the file handler for channel-specific logging.
        Keeps the console handler unchanged.

        Args:
            logger: The logger instance
            output_dir: Directory where the log file should be created
            log_filename: Name of the log file

        Returns:
            Path to the created log file
        """
        log_file_path = os.path.join(output_dir, log_filename)

        # Remove any existing file handlers but keep console handlers
        LoggingManager.cleanup_file_handlers(logger)

        # Add new file handler for this channel
        file_handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return log_file_path

    @staticmethod
    def setup_logger(level=logging.INFO):
        """
        Create a logger with both console and optional file output.
        
        Args:
            name: Logger name (usually module name or dataset name)
            log_file: Optional path to log file. If None, only console logging
            level: Logging level (default: INFO)
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger()
        
        # Avoid adding handlers multiple times
        if logger.hasHandlers():
            logger.handlers.clear()
        
        logger.setLevel(level)
        
        # Create detailed formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
