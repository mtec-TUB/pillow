"""
Base file handler for PSG data processing.
"""


class FileHandler:
    """Base class for handling different PSG file formats."""

    def __init__(self, logger=None):
        self.logger = logger
        self.file_extension = None
        # Call the child's initialization method to set file_extension
        self._initialize()

    def _initialize(self):
        """Initialize handler-specific attributes. Must be implemented by subclasses."""
        raise NotImplementedError

    def get_channels(self, filepath):
        """Extract channel information from file."""
        raise NotImplementedError

    def read_signal(self, filepath, channel):
        """Read signal data for a specific channel."""
        raise NotImplementedError

    def get_signal_data(self, filepath, epoch_duration, channel):
        """Get complete signal information for processing."""
        raise NotImplementedError

    def supports_format(self, filepath):
        """Check if this handler supports the given file format."""
        return self.file_extension in filepath.lower()
