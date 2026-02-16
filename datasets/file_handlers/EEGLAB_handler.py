"""
mat file handler for PSG data processing.
"""

import mne
from pymatreader import read_mat


class EEGLABHandler:
    """Handler for eeglab .set files."""

    def get_channels(self, logger, filepath):
        """Extract channel names and frequencies from .set files."""
        try:
            # Fast path: read only metadata from the .set MAT file.
            mat = read_mat(filepath, variable_names=['chanlocs'])

            chanlocs = mat.get("chanlocs")
            labels = chanlocs.get('labels',[]) 

            return labels
        except Exception as e:
            logger.warning(
                f"Fast channel read failed for {filepath}: {e}. Falling back to MNE."
            )
            try:
                raw_data = mne.io.read_raw_eeglab(filepath, verbose="WARNING")
                return raw_data.ch_names
            except Exception as mne_error:
                logger.error(f"Error reading .set file {filepath}: {mne_error}")
                raise

    def read_signal(self, logger, filepath, channel):
        """Read signal from .set file for specific channel."""
        try:
            raw_data = mne.io.read_raw_eeglab(filepath, verbose='ERROR')
            if channel in raw_data.ch_names:
                return raw_data.get_data(picks=channel)[0]
        except Exception as e:
            logger.error(f"Error reading signal from {filepath}: {e}")
        return None

    def get_signal_data(self, logger, filepath, channel):
        """Get complete EDF signal information for processing."""
        try:
            raw_data = mne.io.read_raw_eeglab(filepath, verbose='WARNING', preload=True)

            signal = raw_data.get_data(picks=channel)[0]
            samples = raw_data.n_times
            assert samples == len(signal)
            logger.info(f"Select channel samples: {samples}")

            sampling_rate = raw_data.info["sfreq"]
            file_duration = samples / sampling_rate
            assert file_duration == raw_data.duration

            start_datetime = raw_data.info["meas_date"]

            unit = raw_data.info['chs'][raw_data.info['ch_names'].index(channel)]['unit']
            unit = mne._fiff.meas_info._unit2human[unit]
            return {
                "signal": signal,
                "sampling_rate": sampling_rate,
                "start_datetime": start_datetime,
                "file_duration": file_duration,
            }
        except Exception as e:
            logger.error(f"Error processing mat file {filepath}: {e}")
            raise
