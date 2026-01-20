"""
WFDB file handler for PSG data processing.
"""

import os
import numpy as np
import wfdb
from datetime import datetime, date
from .base import FileHandler


class WFDBHandler(FileHandler):
    """Handler for WFDB files."""

    def get_channels(self, logger, filepath):
        """Extract channel names from WFDB files."""
        try:
            psg_fname_no_ext, _ = os.path.splitext(filepath)
            record = wfdb.rdheader(psg_fname_no_ext)
            # freqs = []
            # for ch_name in record.sig_name:
            #     _, fields = wfdb.rdsamp(psg_fname_no_ext, channel_names=[ch_name])
            #     freqs.append(fields["fs"])
            return record.sig_name #, freqs
        except Exception as e:
            logger.error(f"Error reading WFDB file {filepath}: {e}")
            return []

    def read_signal(self, logger, filepath, channel):
        """Read signal from WFDB file for specific channel."""
        try:
            psg_fname_no_ext, _ = os.path.splitext(filepath)
            record = wfdb.rdrecord(psg_fname_no_ext)
            signal = np.transpose(record.p_signal)

            if channel in record.sig_name:
                select_ch_idx = record.sig_name.index(channel)
                return signal[select_ch_idx]
        except Exception as e:
            logger.error(f"Error reading WFDB signal from {filepath}: {e}")
        return None

    def get_signal_data(self, logger, filepath, channel):
        """Get complete WFDB signal information for processing."""
        try:
            psg_fname, _ = os.path.splitext(filepath)
            record = wfdb.rdheader(psg_fname)
            signal_labels = record.sig_name

            if channel not in signal_labels:
                logger.info(f"Channel {channel} not found")
                return None
            
            if record.base_datetime:
                start_datetime = record.base_datetime
            elif record.base_date and record.base_time:
                start_datetime = datetime.combine(record.base_date,record.base_time)
            elif record.base_time:
                start_datetime = datetime.combine(date(1985,1,1),record.base_time)
            else:
                start_datetime = None

            logger.info(f"Channel selected: {channel}")

            records, fields = wfdb.rdsamp(psg_fname, channel_names=[channel])
            select_ch_idx = fields["sig_name"].index(channel)

            sampling_rate = fields["fs"]
            signal = records[:, select_ch_idx]

            logger.info(f"Select channel samples: {len(signal)}")

            file_duration = fields["sig_len"] / sampling_rate

            return {
                "signal": signal,
                "sampling_rate": sampling_rate,
                "start_datetime": start_datetime,
                "file_duration": file_duration,
            }
        except Exception as e:
            logger.error(f"Error processing WFDB file {filepath}: {e}")
            raise
