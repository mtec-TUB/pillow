"""
H5 file handler for PSG data processing.
"""

import h5py
from .base import FileHandler


class H5Handler(FileHandler):
    """Handler for H5 files."""

    def collect_h5_dataset(self, name, obj, dataset):
        """
        Helper function to collect H5 datasets during file iteration.

        Args:
            name: Dataset name
            obj: H5 object
            dataset: List to append dataset information to
        """
        if isinstance(obj, h5py.Dataset):
            dataset.append((name, obj[:]))

    def get_channels(self, logger, filepath):
        """Extract channel names from H5 files."""
        try:
            with h5py.File(filepath, "r") as f:
                dataset = []
                f.visititems(
                    lambda name, obj: self.collect_h5_dataset(name, obj, dataset)
                )
                return [data[0] for data in dataset]
        except Exception as e:
            logger.error(f"Error reading H5 file {filepath}: {e}")
            return []

    def read_signal(self, logger, filepath, channel):
        """Read signal from H5 file for specific channel."""
        try:
            with h5py.File(filepath, "r") as f:
                dataset = []
                f.visititems(
                    lambda name, obj: self.collect_h5_dataset(name, obj, dataset)
                )
                signal_labels = [data[0] for data in dataset]
                if channel in signal_labels:
                    select_ch_idx = signal_labels.index(channel)
                    return dataset[select_ch_idx][1]
        except Exception as e:
            logger.error(f"Error reading H5 signal from {filepath}: {e}")
        return None

    def get_signal_data(self, logger, filepath, channel):
        """Get complete H5 signal information for processing."""
        # DOD-O and DOD-H -specific sampling rate
        sampling_rate = 250
        try:
            with h5py.File(filepath, "r") as f:
                dataset = []
                f.visititems(
                    lambda name, obj: self.collect_h5_dataset(name, obj, dataset)
                )

                ch_names = [data[0] for data in dataset]

                if channel not in ch_names:
                    self.logger.info(f"Channel {channel} not found")
                    return None

                select_ch_idx = ch_names.index(channel)
                logger.info(f"Channel selected: {channel}")

                ch_samples = [data[1].shape[0] for data in dataset]
                logger.info(f"Select channel samples: {ch_samples[select_ch_idx]}")
                file_duration = ch_samples[select_ch_idx] / sampling_rate

                signal = dataset[select_ch_idx][1]

                return {
                    "signal": signal,
                    "sampling_rate": sampling_rate,
                    "start_datetime": None,
                    "file_duration": file_duration,
                }
        except Exception as e:
            logger.error(f"Error processing H5 file {filepath}: {e}")
            raise
