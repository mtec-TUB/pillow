"""
CSV file handler for PSG data processing.
"""

import pandas as pd
from .base import FileHandler


class CSVHandler(FileHandler):
    """Handler for CSV files."""

    def __init__(self):
        super().__init__()
        self.file_extension = ".csv"

    def get_channels(self, filepath):
        """Extract column names from CSV files."""
        try:
            dataset = pd.read_csv(filepath, sep=",", header=0, nrows=1)
            return list(dataset.columns)
        except Exception as e:
            print(f"Error reading CSV file {filepath}: {e}")
            return []

    def read_signal(self, filepath, channel):
        """Read signal from CSV file for specific channel."""
        try:
            dataset = pd.read_csv(filepath, sep=",", header=0)
            if channel in dataset.columns:
                return dataset[channel].to_numpy()
        except Exception as e:
            print(f"Error reading CSV signal from {filepath}: {e}")
        return None

    def get_signal_data(self, logger, filepath, epoch_duration, channel, ann_parse):
        """Get complete CSV signal information for processing."""
        try:
            sampling_rate = 64
            dataset = pd.read_csv(filepath, sep=",", header=0)

            if channel not in dataset.columns:
                logger.info(f"Channel {channel} not found")
                return None

            # Prepare dataset to get labels:
            # - starting after Preparation Stage 'P'
            # - filter out only full epochs (30 seconds) from 64Hz data
            dataset = dataset[dataset["Sleep_Stage"] != "P"].reset_index()
            signal = dataset[channel].to_numpy()
            dataset = dataset.iloc[
                ((dataset.index == 0) | (dataset.index + 1) % (sampling_rate * 30) == 0)
            ]

            logger.info(f"Channel selected: {channel}")
            logger.info(f"Select channel samples: {len(signal)}")

            n_epoch_samples = sampling_rate * epoch_duration
            file_duration = len(signal)

            return {
                "signal": signal,
                "sampling_rate": sampling_rate,
                "n_epoch_samples": n_epoch_samples,
                "start_datetime": None,
                "file_duration": file_duration,
            }
        except Exception as e:
            logger.error(f"Error processing CSV file {filepath}: {e}")
            return None
