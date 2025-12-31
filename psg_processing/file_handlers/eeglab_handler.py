"""
mat file handler for PSG data processing.
"""

from mne.io import read_raw_eeglab
from .base import FileHandler


class EEGLABHandler(FileHandler):
    """Handler for mat files."""

    def get_channels(self, logger, filepath):
        """Extract channel names and frequencies from .set files."""
        try:
            raw_data = read_raw_eeglab(filepath, verbose=False, preload=True)
            return raw_data.ch_names
        except Exception as e:
            logger.error(f"Error reading .set file {filepath}: {e}")
            return []

    def read_signal(self, logger, filepath, channel):
        """Read signal from .set file for specific channel."""
        try:
            raw_data = read_raw_eeglab(filepath, verbose=False, preload=True)
            if channel in raw_data.ch_names:
                return raw_data.get_data(picks=channel)[0]
        except Exception as e:
            logger.error(f"Error reading signal from {filepath}: {e}")
        return None

    def get_signal_data(self, logger, filepath, channel):
        """Get complete EDF signal information for processing."""
        try:
            raw_data = read_raw_eeglab(filepath, verbose=False, preload=True)

            if channel not in raw_data.ch_names:
                self.logger.info(f"Channel {channel} not found")
                return None

            logger.info(f"Channel selected: {channel}")

            signal = raw_data.get_data(picks=channel)[0]
            samples = raw_data.n_times
            assert samples == len(signal)
            logger.info(f"Select channel samples: {samples}")

            sampling_rate = raw_data.info['sfreq']
            file_duration = samples / sampling_rate
            assert file_duration == raw_data.duration

            start_datetime = raw_data.info['meas_date']
            return {
                "signal": signal,
                "sampling_rate": sampling_rate,
                "start_datetime": start_datetime,
                "file_duration": file_duration,
            }
        except Exception as e:
            logger.error(f"Error processing mat file {filepath}: {e}")
            raise
