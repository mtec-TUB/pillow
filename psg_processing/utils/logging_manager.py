import logging
import os
import sys

class BufferedHandler(logging.Handler):
    """
    Buffer log records in memory and allow channel-based filtering.
    Flush them to console and file on demand, with optional channel filtering.
    """

    def __init__(self, formatter):
        super().__init__()
        self.buffer = []
        self.current_channel = None
        self.setFormatter(formatter)

    def emit(self, record):
        record.channel = self.current_channel
        self.buffer.append(record)

    def set_channel(self, channel):
        self.current_channel = channel

    def flush_to_console_and_file(self, log_path, channel=None):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Write all logs to file, optionally filtering by channel
        with open(log_path, "a") as f:
            for record in self.buffer:
                if channel is None or record.channel == channel:
                    f.write(self.format(record) + "\n")

        # Write warnings and above to console, optionally filtering by channel
        for record in self.buffer:
            if (channel is None or record.channel == channel) \
            and record.levelno >= logging.WARNING:
                sys.stdout.write(self.format(record) + "\n")

        sys.stdout.flush()

    def clear(self, channel=None):
        if channel is None:
            self.buffer.clear()
        else:
            self.buffer = [
                r for r in self.buffer if r.channel != channel
            ]

class LoggingManager:

    def __init__(self, level=logging.INFO,
                 fmt="%(asctime)s - %(levelname)s - %(message)s",
                 datefmt="%Y-%m-%d %H:%M:%S"):
        self.level = level
        self.format = fmt
        self.date_format = datefmt

    def create_pipeline_logger(self, name="psg.pipeline"):
        logger = logging.getLogger(name)
        logger.setLevel(self.level)
        logger.propagate = False
        logger.handlers.clear()

        formatter = logging.Formatter(self.format, self.date_format)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(self.level)

        logger.addHandler(console)
        return logger

    def create_file_logger(self, file_identifier):
        logger_name = f"psg.file.{file_identifier}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.level)
        logger.propagate = False
        logger.handlers.clear()

        formatter = logging.Formatter(self.format, self.date_format)

        buffer_handler = BufferedHandler(formatter)
        buffer_handler.setLevel(self.level)

        logger.addHandler(buffer_handler)

        return logger, buffer_handler