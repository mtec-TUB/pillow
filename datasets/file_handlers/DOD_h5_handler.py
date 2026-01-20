"""
H5 file handler for PSG data processing.
"""

import h5py


class DOD_H5Handler:
    """Handler for H5 files."""

    def get_channels(self, logger, filepath):
        """Extract channel names (datasets) from H5 files."""
        try:
            channel_names = []
            with h5py.File(filepath, "r") as f:

                def visitor(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        channel_names.append(name)

                f.visititems(visitor)
            return channel_names
        except Exception as e:
            logger.error(f"Error reading H5 file {filepath}: {e}")
            return []

    def read_signal(self, logger, filepath, channel):
        """Read signal from H5 file for specific channel."""
        try:
            with h5py.File(filepath, "r") as f:
                if channel not in f:
                    return None

                signal = f[channel][:]
                return signal
        except Exception as e:
            logger.error(f"Error reading H5 signal from {filepath}: {e}")

    def get_signal_data(self, logger, filepath, channel):
        """Get complete H5 signal information for processing."""
        try:
            with h5py.File(filepath, "r") as f:
                dataset = f[channel]
                signal = dataset[:]
                logger.info(f"Channel selected: {channel}")
                logger.info(f"Select channel samples: {len(signal)}")

                # works for DOD-H and DOD-O
                unit = dataset.parent.attrs.get("unit")
                sampling_rate = dataset.parent.attrs.get("fs")

                file_duration = len(signal) / sampling_rate

                return {
                    "signal": signal,
                    "sampling_rate": sampling_rate,
                    "unit": unit,
                    "start_datetime": None,
                    "file_duration": file_duration,
                }
        except Exception as e:
            logger.error(f"Error processing H5 file {filepath}: {e}")
            raise
