import h5py
from datetime import datetime

class DOD_H5Handler:
    """Handler for DOD dataset files
    Works for DOD-O and DOD-H dataset
    """

    def get_channels(self, logger, filepath):
        """Extract channel names from file."""
        try:
            channel_names = []
            with h5py.File(filepath, "r") as f:

                def visitor(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        channel_names.append(name)

                f["signals"].visititems(visitor)    # only get channels in signals subfolder
            return channel_names
        except Exception as e:
            logger.error(f"Error during channel extraction from {filepath}: {e}")
            raise

    def read_signal(self, logger, filepath, channel):
        """Read signal from file for specific channel."""
        try:
            with h5py.File(filepath, "r") as f:
                if channel not in f["signals"]:
                    return None

                signal = f["signals"][channel][:]
                return signal
        except Exception as e:
            logger.error(f"Error during signal extraction from {filepath}: {e}")
            raise

    def get_signal_data(self, logger, filepath, channel):
        """Get complete signal information for specific channel."""
        try:
            with h5py.File(filepath, "r") as f:
                dataset = f["signals"][channel]
                signal = dataset[:]

                unit = dataset.parent.attrs.get("unit")

                sampling_rate = dataset.parent.attrs.get("fs")
                file_duration = len(signal) / sampling_rate


                start_datetime = datetime.fromtimestamp(f.attrs.get("start_time"))

                return {
                    "signal": signal,
                    "sampling_rate": sampling_rate,
                    "unit": unit,
                    "start_datetime": start_datetime,
                    "file_duration": file_duration,
                }
        except Exception as e:
            logger.error(f"Error during data retrieval {filepath}: {e}")
            raise
