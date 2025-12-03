"""
DREAMT-specific CSV file handler for PSG data processing.
"""

import pandas as pd
from .base import FileHandler


class DreamtCSVHandler(FileHandler):
    """Handler for DREAMT dataset CSV files."""

    def get_channels(self, logger, filepath):
        """Extract column names from DREAMT CSV files."""
        try:
            dataset = pd.read_csv(filepath, sep=",", header=0, nrows=1)
            # Exclude non-signal columns specific to DREAMT
            signal_columns = [col for col in dataset.columns 
                            if col not in ['TIMESTAMP', 'Sleep_Stage']]
            return signal_columns
        except Exception as e:
            logger.error(f"Error reading DREAMT CSV file {filepath}: {e}")
            return []

    def read_signal(self, logger, filepath, channel):
        """Read signal from DREAMT CSV file for specific channel."""
        try:
            dataset = pd.read_csv(filepath, sep=",", header=0)
            if channel in dataset.columns:
                # DREAMT-specific: Remove preparation stage data
                dataset = dataset[dataset["Sleep_Stage"] != "P"].reset_index()
                return dataset[channel].to_numpy()
        except Exception as e:
            logger.error(f"Error reading DREAMT CSV signal from {filepath}: {e}")
        return None

    def get_signal_data(self, logger, filepath, epoch_duration, channel):
        """Get complete DREAMT CSV signal information for processing."""
        try:
            # DREAMT-specific sampling rate
            sampling_rate = 64
            dataset = pd.read_csv(filepath, sep=",", header=0)

            if channel not in dataset.columns:
                self.logger.info(f"Channel {channel} not found")
                return None

            # DREAMT-specific preprocessing:
            # - Remove preparation stage 'P'
            dataset = dataset[dataset["Sleep_Stage"] != "P"].reset_index()
            signal = dataset[channel].to_numpy()

            logger.info(f"Channel selected: {channel}")
            logger.info(f"Select channel samples: {len(signal)}")

            n_epoch_samples = sampling_rate * epoch_duration
            file_duration = len(signal) / sampling_rate

            return {
                "signal": signal,
                "sampling_rate": sampling_rate,
                "n_epoch_samples": n_epoch_samples,
                "start_datetime": None,
                "file_duration": file_duration,
            }
        except Exception as e:
            logger.error(f"Error processing DREAMT CSV file {filepath}: {e}")
            raise