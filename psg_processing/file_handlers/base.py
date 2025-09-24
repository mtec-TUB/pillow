"""
Base file handler for PSG data processing.
"""


class FileHandler:
    """Base class for handling different PSG file formats."""

    def __init__(self, logger=None):
        self.logger = logger

    def get_channels(self, filepath):
        """Extract channel information from file."""
        raise NotImplementedError

    def read_signal(self, filepath, channel):
        """Read signal data for a specific channel."""
        raise NotImplementedError

    def get_signal_data(self, filepath, epoch_duration, channel):
        """Get complete signal information for processing."""
        raise NotImplementedError
