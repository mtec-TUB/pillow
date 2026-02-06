"""
Logging management for PSG processing.
"""

import os
import logging
import glob


class BufferedHandler(logging.Handler):
    """
    A logging handler that buffers log records in memory.
    Records are stored and can be flushed to a file later.
    """
    
    def __init__(self):
        super().__init__()
        self.buffer = []
        self.current_channel = None
    
    def emit(self, record):
        """Store the log record in the buffer with channel information."""
        # Tag the record with the current channel
        record.channel = self.current_channel
        self.buffer.append(record)
    
    def set_current_channel(self, channel):
        """Set the channel for subsequent log records."""
        self.current_channel = channel
    
    def get_records(self, channel=None):
        """Return buffered records, optionally filtered by channel."""
        if channel is None:
            return self.buffer
        return [record for record in self.buffer if getattr(record, 'channel', None) == channel]
    
    def clear(self, channel=None):
        """Clear the buffer, optionally only for a specific channel."""
        if channel is None:
            self.buffer = []
        else:
            self.buffer = [record for record in self.buffer if getattr(record, 'channel', None) != channel]
    
    def flush_to_file(self, file_path, formatter, channel=None):
        """Write buffered records to a file, optionally filtered by channel."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        records = self.get_records(channel)
        with open(file_path, 'a') as f:
            for record in records:
                f.write(formatter.format(record) + '\n')


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
        self.buffered_handler = None

    def cleanup_file_handlers(self, logger):
        """
        Remove all file handlers from logger while keeping console handlers.
        This prevents file handle leaks while preserving console output.
        """
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()  # Properly close the file handle
                logger.removeHandler(handler)

    def setup_file_logging(self, logger, output_dir, log_filename):
        """
        Set up or update the file handler for logging.
        Keeps the console handler unchanged.

        Args:
            logger: The logger instance
            output_dir: Directory where the log file should be created

        Returns:
            Path to the created log file
        """
        log_file_path = os.path.join(output_dir, log_filename)

        # Remove any existing file handlers but keep console handlers
        self.cleanup_file_handlers(logger)

        # Add new file handler
        file_handler = logging.FileHandler(log_file_path)

        formatter = logging.Formatter(self.format, datefmt=self.date_format)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return log_file_path

    def setup_logger(self, dir_name=None, overwrite=False):
        """
        Create a logger with console output and remove all log files if overwriting is desired.

        Args:
            dir: Optional path to folder to delete old logging files

        Returns:
            Configured logger instance
        """

        # delete all exisiting log_files in output folder if overwrite True
        if overwrite and dir_name:
            log_files = glob.glob(os.path.join(dir_name, "**", "*.log"), recursive=True)
            for f in log_files:
                os.remove(f)

        logger = logging.getLogger()

        # Avoid adding handlers multiple times
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(self.level)

        # Create detailed formatter
        formatter = logging.Formatter(fmt=self.format, datefmt=self.date_format)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger
    
    def start_buffering(self, logger):
        """Start buffering log messages instead of writing them immediately."""
        # Remove any existing buffered handler
        self.stop_buffering(logger)
        
        # Create and add new buffered handler
        self.buffered_handler = BufferedHandler()
        self.buffered_handler.setLevel(self.level)
        formatter = logging.Formatter(fmt=self.format, datefmt=self.date_format)
        self.buffered_handler.setFormatter(formatter)
        logger.addHandler(self.buffered_handler)
    
    def set_current_channel(self, channel):
        """Set the current channel being processed for log tagging."""
        if self.buffered_handler is not None:
            self.buffered_handler.set_current_channel(channel)
    
    def stop_buffering(self, logger):
        """Stop buffering and remove the buffered handler."""
        if self.buffered_handler is not None:
            logger.removeHandler(self.buffered_handler)
            self.buffered_handler = None
    
    def flush_buffer_to_file(self, file_path, channel=None):
        """Write buffered log messages to a file, optionally filtered by channel."""
        if self.buffered_handler is not None:
            formatter = logging.Formatter(fmt=self.format, datefmt=self.date_format)
            self.buffered_handler.flush_to_file(file_path, formatter, channel)
    
    def clear_buffer(self, channel=None):
        """Clear the buffer without writing to file, optionally only for a specific channel."""
        if self.buffered_handler is not None:
            self.buffered_handler.clear(channel)
