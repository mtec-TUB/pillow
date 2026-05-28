from mne.io import read_raw_eeglab, read_epochs_eeglab
from mne import _fiff
import numpy as np
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
                raw_data = read_raw_eeglab(filepath, verbose="WARNING",preload=False)
                return raw_data.ch_names
            except TypeError as e:
                raw_epoched_data = read_epochs_eeglab(filepath, verbose="WARNING")
                return raw_epoched_data.ch_names
            except OSError as e:
                logger.error(f"Skipping corrupt/unreadable file: {e}")
                return []
            except Exception as e:
                logger.error(f"Error during channel extraction from: {e}")
                raise

    def read_signal(self, logger, filepath, channel):
        """Read signal from file for specific channel."""
        try:
            raw_data = read_raw_eeglab(filepath, verbose='ERROR')
            if channel in raw_data.ch_names:
                return raw_data.get_data(picks=channel)[0]
            else:
                return None # channel not found in this file
        except TypeError as e:
            # Probably because its an epoched file -> reading as epochs and concatenating
            # First load with mat to get channel names and n_epochs (needed to construct events for mne)
            eeg = read_mat(filepath)
            eeg = eeg.get("EEG", eeg)   # handle nested structure
            chanlocs = eeg.get("chanlocs")
            labels = chanlocs.get('labels',[]) 
            if channel in labels:
                n_epochs = eeg.get("trials")
                events = np.vstack((np.arange(n_epochs,dtype=int), np.zeros(n_epochs,dtype=int), np.ones(n_epochs, dtype=int))).T
                event_id=dict(unknown=1)
                raw_epoched_data = read_epochs_eeglab(filepath, verbose="WARNING", events=events, event_id=event_id)
            
                epoched_data = raw_epoched_data.get_data(picks=channel)[:,0,:]  # shape (n_epochs, n_times)
                continuous = epoched_data.flatten()
                return continuous
            else:
                return None
        except Exception as e:
            logger.error(f"Error reading signal: {e}")
            return None
        
    def get_file_info(self, logger, filepath):
        """Get start datetime and file duration."""
        try:
            raw_data = read_raw_eeglab(filepath, verbose='WARNING', preload=True)
            info = raw_data.info

            file_duration = raw_data.duration

        except TypeError as e:
            raw_data = read_epochs_eeglab(filepath, verbose="WARNING")
            info = raw_data.info
            file_duration = raw_data.duration

        except RuntimeError as e:
            try:
                logger.warning(f"File is truncated, reconstructing it from .fdt file.")
                raw_data = read_raw_eeglab(filepath, verbose='WARNING', preload=False)
                info = raw_data.info
                fs = info["sfreq"]
                n_channels = info['nchan']
                fdt_file = filepath.replace('.set', '.fdt')
                eeg_data = np.fromfile(fdt_file, dtype=np.float32)
                file_duration = (len(eeg_data) // n_channels) / fs
            except Exception as e:
                logger.error(f"Runtime error during file info retrieval: {e} \n Skipping file.")
                return {}
        except Exception as e:
            logger.error(f"Error during file info retrieval: {e}")
            raise

        start_datetime = info["meas_date"]
        return {"start_datetime": start_datetime, "file_duration": file_duration}

    def get_signal_data(self, logger, filepath, channel):
        """Get complete signal information for specific channel."""
        try:
            raw_data = read_raw_eeglab(filepath, verbose='WARNING', preload=True)

            signal = raw_data.get_data(picks=channel)[0]
            if np.all(np.isnan(signal)):
                logger.warning(f"Signal for channel {channel} contains only NaN values.")
                return {}

            info = raw_data.info

        except TypeError as e:
            # Probably because its an epoched file -> reading as epochs and concatenating
            # First load with mat to get channel names and n_epochs (needed to construct events for mne)
            eeg = read_mat(filepath)
            eeg = eeg.get("EEG", eeg)   # handle nested structure

            # Construct events because mne cannot handle epoched .set files without events
            n_epochs = eeg.get("trials")
            events = np.vstack((np.arange(n_epochs,dtype=int), np.zeros(n_epochs,dtype=int), np.ones(n_epochs, dtype=int))).T
            event_id=dict(unknown=1)
            raw_epoched_data = read_epochs_eeglab(filepath, verbose="WARNING", events=events, event_id=event_id)
        
            epoched_data = raw_epoched_data.get_data(picks=channel)[:,0,:]  # shape (n_epochs, n_times)
            
            signal = epoched_data.flatten()
            info = raw_epoched_data.info
        except RuntimeError as e:
            try:
                # File is truncated, try to reconstruct signal from .fdt file
                raw_data = read_raw_eeglab(filepath, verbose='WARNING', preload=False)
                info = raw_data.info
                n_channels = info['nchan']
                channel_idx = raw_data.ch_names.index(channel)
                fdt_file = filepath.replace('.set', '.fdt')
                eeg_data = np.fromfile(fdt_file, dtype=np.float32)
                eeg_data = eeg_data[:(len(eeg_data) // n_channels) * n_channels]  #  ensure it's divisible by n_channels

                eeg_data = eeg_data.reshape((n_channels,-1), order='F')  # shape: (n_channels, n_samples)

                signal = eeg_data[channel_idx,:]
            except Exception as e:
                logger.error(f"Runtime error during data retrieval: {e} \n Skipping file.")
                return {}
        except Exception as e:
            logger.error(f"Error during data retrieval: {e}")
            raise

        sampling_rate = info["sfreq"]
        unit = info['chs'][info['ch_names'].index(channel)]['unit']
        unit = _fiff.meas_info._unit2human[unit]
        return {
            "signal": signal,
            "sampling_rate": sampling_rate,
            "unit": unit
        }
