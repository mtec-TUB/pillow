"""
WFDB file handler for PSG data processing.
"""

import os
import numpy as np
import wfdb
from .base import FileHandler


class WFDBHandler(FileHandler):
    """Handler for WFDB files."""

    def _initialize(self):
        self.file_extension = ".hea"

    def get_channels(self, filepath):
        """Extract channel names from WFDB files."""
        try:
            psg_fname_no_ext, _ = os.path.splitext(filepath)
            record = wfdb.rdheader(psg_fname_no_ext)
            return record.sig_name
        except Exception as e:
            self.logger.error(f"Error reading WFDB file {filepath}: {e}")
            return []

    def read_signal(self, filepath, channel):
        """Read signal from WFDB file for specific channel."""
        try:
            psg_fname_no_ext, _ = os.path.splitext(filepath)
            record = wfdb.rdrecord(psg_fname_no_ext)
            signal = np.transpose(record.p_signal)

            if channel in record.sig_name:
                select_ch_idx = record.sig_name.index(channel)
                return signal[select_ch_idx]
        except Exception as e:
            self.logger.error(f"Error reading WFDB signal from {filepath}: {e}")
        return None

    def get_signal_data(self, filepath, epoch_duration, channel):
        """Get complete WFDB signal information for processing."""
        try:
            psg_fname, ext = os.path.splitext(filepath)
            record = wfdb.rdheader(psg_fname)
            signal_labels = record.sig_name

            if channel not in signal_labels:
                self.logger.info(f"Channel {channel} not found")
                return None

            self.logger.info(f"Channel selected: {channel}")

            records, fields = wfdb.rdsamp(psg_fname, channel_names=[channel])
            select_ch_idx = fields["sig_name"].index(channel)

            sampling_rate = fields["fs"]
            signal = records[:, select_ch_idx]

            self.logger.info(f"Select channel samples: {len(signal)}")

            n_epoch_samples = int(epoch_duration * sampling_rate)
            file_duration = fields["sig_len"] / sampling_rate

            return {
                "signal": signal,
                "sampling_rate": sampling_rate,
                "n_epoch_samples": n_epoch_samples,
                "start_datetime": None,
                "file_duration": file_duration,
            }
        except Exception as e:
            self.logger.error(f"Error processing WFDB file {filepath}: {e}")
            raise
