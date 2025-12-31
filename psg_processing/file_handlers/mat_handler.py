"""
mat file handler for PSG data processing.
"""

from scipy.io import loadmat
from .base import FileHandler


class MATHandler(FileHandler):
    """Handler for mat files."""

    def get_channels(self, logger, filepath):
        """Extract channel names and frequencies from mat files."""
        try:
            psg_f = loadmat(filepath)['Data']
            return psg_f.dtype.names
        except Exception as e:
            logger.error(f"Error reading mat file {filepath}: {e}")
            return []

    def read_signal(self, logger, filepath, channel):
        """Read signal from Mat file for specific channel."""
        try:
            psg_f = loadmat(filepath)['Data']
            if channel in psg_f.dtype.names:
                return psg_f[0,0][channel][:, 0]
        except Exception as e:
            logger.error(f"Error reading mat signal from {filepath}: {e}")
        return None

    def get_signal_data(self, logger, filepath, channel):
        """Get complete EDF signal information for processing."""
        try:
            psg_f = loadmat(filepath)['Data']

            if channel not in psg_f.dtype.names:
                self.logger.info(f"Channel {channel} not found")
                return None

            logger.info(f"Channel selected: {channel}")

            signal = psg_f[0,0][channel][:, 0]
            samples = psg_f[0,0]['num_Labels'][0,0]
            assert len(signal) == samples
            logger.info(f"Select channel samples: {samples}")

            sampling_rate = psg_f[0,0]['fs'][0,0]
            file_duration = samples / sampling_rate

            return {
                "signal": signal,
                "sampling_rate": sampling_rate,
                "start_datetime": None,
                "file_duration": file_duration,
            }
        except Exception as e:
            logger.error(f"Error processing mat file {filepath}: {e}")
            raise
