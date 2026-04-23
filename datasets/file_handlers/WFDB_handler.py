import os
import numpy as np
import wfdb
from datetime import datetime, date, time

class WFDBHandler:
    """Handler for WFDB files (.hea + .dat)."""

    def get_channels(self, logger, filepath):
        """Extract channel names from file."""
        try:
            psg_fname_no_ext, _ = os.path.splitext(filepath)
            record = wfdb.rdheader(psg_fname_no_ext)
            # freqs = []
            # for ch_name in record.sig_name:
            #     _, fields = wfdb.rdsamp(psg_fname_no_ext, channel_names=[ch_name])
            #     freqs.append(fields["fs"])
            return record.sig_name  # , freqs
        except Exception as e:
            logger.error(f"Error reading WFDB file {filepath}: {e}")
            raise

    def read_signal(self, logger, filepath, channel):
        """Read signal from file for specific channel."""
        try:
            psg_fname_no_ext, _ = os.path.splitext(filepath)
            record = wfdb.rdrecord(psg_fname_no_ext, channel_names=[channel])

            if record.n_sig != 0:
                return record.p_signal[:, 0]
            else:
                return None # channel not found in this file
        except Exception as e:
            logger.error(f"Error reading signal from {filepath}: {e}")
            raise

    def get_start_datetime(self, logger, filepath):
        """Get start datetime of file."""
        try:
            psg_fname_no_ext, _ = os.path.splitext(filepath)
            record = wfdb.rdheader(psg_fname_no_ext)
            
            psg_date = record.base_date
            if psg_date == None:
                psg_date = date(1985, 1, 1)
            psg_time = record.base_time
            if psg_time == None:
                start_datetime = None
            else:
                start_datetime = datetime.combine(psg_date, psg_time)

            return start_datetime
        except Exception as e:
            logger.error(f"Error processing WFDB file {filepath}: {e}")
            raise

    def get_signal_data(self, logger, filepath, channel):
        """Get complete signal information for processing."""
        try:
            psg_fname_no_ext, _ = os.path.splitext(filepath)
            record = wfdb.rdrecord(psg_fname_no_ext, channel_names=[channel])
            
            sampling_rate = record.fs
            signal = record.p_signal[:,0]

            file_duration = record.sig_len / sampling_rate

            unit = record.units[0]

            return {
                "signal": signal,
                "sampling_rate": sampling_rate,
                "unit": unit,
                "file_duration": file_duration,
            }
        except Exception as e:
            logger.error(f"Error processing WFDB file {filepath}: {e}")
            raise
