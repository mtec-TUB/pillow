import mne
from pymatreader import read_mat

class EEGLABHandler:
    """Handler for EEGLAB .set files."""

    def get_channels(self, logger, filepath):
        """Extract channel names file."""
        try:
            # Fast path: read only metadata from the .set file.
            mat = read_mat(filepath, variable_names=['chanlocs'])
            chanlocs = mat.get("chanlocs")
            labels = chanlocs.get('labels',[]) 
            return labels
        
        except:
            # Fall back to slower mne that may preload the data
            try:
                raw_data = mne.io.read_raw_eeglab(filepath, verbose="WARNING")
                return raw_data.ch_names
            except Exception as e:
                logger.error(f"Error during channel extraction from {filepath}: {e}")
                raise

    def read_signal(self, logger, filepath, channel):
        """Read signal from file for specific channel."""
        try:
            raw_data = mne.io.read_raw_eeglab(filepath, verbose='ERROR')
            if channel in raw_data.ch_names:
                return raw_data.get_data(picks=channel)[0]
            else:
                return None # channel not found in this file
        except Exception as e:
            logger.error(f"Error reading signal from {filepath}: {e}")
        

    def get_signal_data(self, logger, filepath, channel):
        """Get complete signal information for specific channel."""
        try:
            raw_data = mne.io.read_raw_eeglab(filepath, verbose='WARNING', preload=True)

            signal = raw_data.get_data(picks=channel)[0]
            samples = raw_data.n_times
            assert samples == len(signal)

            sampling_rate = raw_data.info["sfreq"]
            file_duration = samples / sampling_rate
            assert file_duration == raw_data.duration

            start_datetime = raw_data.info["meas_date"]

            unit = raw_data.info['chs'][raw_data.info['ch_names'].index(channel)]['unit']
            unit = mne._fiff.meas_info._unit2human[unit]
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
