"""File handler for WFDB records (.hea header + .dat signal files)."""

import os
from datetime import datetime, date
import logging

import wfdb

class WFDBHandler:
    """Handler for WFDB files (.hea + .dat)."""

    def get_channels(self, logger: logging.Logger, filepath):
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
            logger.error("Error during channel extraction: %s", e)
            raise

    def read_signal(self, logger: logging.Logger, filepath, channel):
        """Read signal from file for specific channel."""
        try:
            psg_fname_no_ext, _ = os.path.splitext(filepath)
            record = wfdb.rdrecord(psg_fname_no_ext, channel_names=[channel])

            if record.n_sig != 0:
                return record.p_signal[:, 0]

            return None # channel not found in this file
        except Exception as e:
            logger.error("Error reading signal: %s", e)
            raise

    def get_file_info(self, logger: logging.Logger, filepath):
        """Get start datetime and file duration."""
        try:
            psg_fname_no_ext, _ = os.path.splitext(filepath)
            record = wfdb.rdheader(psg_fname_no_ext)

            psg_date = record.base_date
            if psg_date is None:
                psg_date = date(1985, 1, 1)
            psg_time = record.base_time
            if psg_time is None:
                start_datetime = None
            else:
                start_datetime = datetime.combine(psg_date, psg_time)

            fs = record.fs[0] if isinstance(record.fs, list) else record.fs
            sig_len = record.sig_len[0] if isinstance(record.sig_len, list) else record.sig_len

            file_duration = sig_len / fs

            return {"start_datetime": start_datetime, "file_duration": file_duration}
        except Exception as e:
            logger.error("Error during file info retrieval: %s", e)
            raise

    def get_signal_data(self, logger: logging.Logger, filepath, channel):
        """Get complete signal information for processing."""
        try:
            psg_fname_no_ext, _ = os.path.splitext(filepath)
            record = wfdb.rdrecord(psg_fname_no_ext, channel_names=[channel])

            sampling_rate = record.fs
            signal = record.p_signal[:,0]

            # file_duration = record.sig_len / sampling_rate

            unit = record.units[0]

            return {
                "signal": signal,
                "sampling_rate": sampling_rate,
                "unit": unit
            }
        except Exception as e:
            logger.error("Runtime error during data retrieval: %s", e)
            raise
