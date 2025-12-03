"""
EDF file handler for PSG data processing.
"""

import pyedflib
from .base import FileHandler


class EDFHandler(FileHandler):
    """Handler for EDF files."""

    def get_channels(self, logger, filepath):
        """Extract channel names and frequencies from EDF files."""
        try:
            with pyedflib.EdfReader(filepath) as psg_f:
                labels = psg_f.getSignalLabels()
                # freqs = psg_f.getSampleFrequencies()
                # return labels, freqs 
                return labels
        except Exception as e:
            logger.error(f"Error processing EDF file: {e}")
            logger.error("Maybe the repair_edfs.py script or EDF Browser header repairer can help.")
            raise

    def read_signal(self, logger, filepath, channel):
        """Read signal from EDF file for specific channel."""
        try:
            with pyedflib.EdfReader(filepath) as psg_f:
                ch_names_file = psg_f.getSignalLabels()
                if channel in ch_names_file:
                    ch_idx = ch_names_file.index(channel)
                    return psg_f.readSignal(ch_idx)
        except Exception as e:
            logger.error(f"Error reading EDF signal from: {e}")
            logger.error("Maybe the repair_edfs.py script can help.")
        return None

    def get_signal_data(self, logger, filepath, epoch_duration, channel):
        """Get complete EDF signal information for processing."""
        try:
            psg_f = pyedflib.EdfReader(filepath)

            ch_names = psg_f.getSignalLabels()
            if channel not in ch_names:
                self.logger.info(f"Channel {channel} not found")
                psg_f.close()
                return None

            select_ch_idx = ch_names.index(channel)
            logger.info(f"Channel selected: {channel}")

            start_datetime = psg_f.getStartdatetime()
            file_duration = psg_f.getFileDuration()

            ch_samples = psg_f.getNSamples()
            logger.info(f"Select channel samples: {ch_samples[select_ch_idx]}")
            ch_freq = psg_f.getSampleFrequencies()

            sampling_rate = ch_freq[select_ch_idx]
            n_epoch_samples = epoch_duration * sampling_rate
            signal = psg_f.readSignal(select_ch_idx)
            
            unit = psg_f.getPhysicalDimension(select_ch_idx)

            psg_f.close()

            return {
                "signal": signal,
                "sampling_rate": sampling_rate,
                "unit": unit,
                "n_epoch_samples": n_epoch_samples,
                "start_datetime": start_datetime,
                "file_duration": file_duration,
            }
        except Exception as e:
            logger.error(f"Error processing EDF file: {e}")
            logger.error("Maybe the repair_edfs.py script or EDF Browser header repairer can help.")
            raise
