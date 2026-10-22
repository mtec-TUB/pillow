"""
Logging utilities: console logger for the pipeline and buffered per-file loggers.
"""

import logging
import os
import sys

class BufferedHandler(logging.Handler):
    """
    Buffer log records in memory and allow channel-based filtering.
    Flush them to console and file on demand, with optional channel filtering.
    """

    def __init__(self, formatter, console_level):
        super().__init__()
        self.buffer = []
        self.console_level = console_level
        self.current_channel = None
        self.setFormatter(formatter)

    def emit(self, record):
        record.channel = self.current_channel
        self.buffer.append(record)

    def set_channel(self, channel):
        """Tag all following log records with this channel."""
        self.current_channel = channel

    def flush_to_console_and_file(self, log_path, channel=None):
        """Write buffered records to log file and console, optionally only for one channel."""
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Write all logs to file
        with open(log_path, "a", encoding="utf-8") as f:
            for record in self.buffer:
                if channel is None or record.channel is None or record.channel == channel:
                    f.write(self.format(record) + "\n")

        # Write to console with specified log level
        console_levelno = logging.getLevelNamesMapping().get(self.console_level, logging.INFO)
        for record in self.buffer:
            if (channel is None or record.channel == channel) \
            and record.levelno >= console_levelno:
                sys.stdout.write(self.format(record) + "\n")

        sys.stdout.flush()

    def flush_to_console(self):
        """Write buffered records to console only."""
        console_levelno = logging.getLevelNamesMapping().get(self.console_level, logging.INFO)
        for record in self.buffer:
            if record.levelno >= console_levelno:
                sys.stdout.write(self.format(record) + "\n")
        sys.stdout.flush()

    def clear(self, channel=None):
        """Remove buffered records, either all or only those of one channel."""
        if channel is None:
            self.buffer.clear()
        else:
            self.buffer = [
                r for r in self.buffer if r.channel != channel
            ]

class LoggingManager:
    """Create the pipeline logger and the buffered per-file loggers with a shared format."""

    def __init__(self, console_level=logging.INFO,
                 fmt="%(asctime)s - %(levelname)s - %(message)s",
                 datefmt="%Y-%m-%d %H:%M:%S"):
        self.console_level = console_level
        self.format = fmt
        self.date_format = datefmt

    def create_pipeline_logger(self, name="psg.pipeline"):
        """Create a logger that writes directly to the console."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)   # Always show pipeline logs in console
        logger.propagate = False
        logger.handlers.clear()

        formatter = logging.Formatter(self.format, self.date_format)
        console = logging.StreamHandler()
        console.setFormatter(formatter)

        logger.addHandler(console)
        return logger

    def create_file_logger(self, file_identifier):
        """Create a logger for one PSG file that buffers its records until flushed.

        Returns:
            tuple: (logger, buffer_handler)
        """
        logger_name = f"psg.file.{file_identifier}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()

        formatter = logging.Formatter(
            f"%(asctime)s - %(levelname)s - [{file_identifier}] %(message)s",
            self.date_format,
        )

        buffer_handler = BufferedHandler(formatter, self.console_level)
        logger.addHandler(buffer_handler)

        return logger, buffer_handler
