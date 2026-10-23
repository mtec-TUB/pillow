"""File handler for EEGLAB files (.set, optionally with separate .fdt data file)."""

import logging

from mne.io.eeglab import read_raw_eeglab, read_epochs_eeglab
import numpy as np
from pymatreader import read_mat

class EEGLABHandler:
    """Handler for EEGLAB .set files."""

    def get_channels(self, logger: logging.Logger, filepath):
        """Extract channel names file."""
        try:
            # Fast path: read only metadata from the .set file.
            mat = read_mat(filepath, variable_names=['chanlocs'])
            chanlocs = mat.get("chanlocs")
            labels = chanlocs.get('labels',[])
            return labels

        except Exception:
            # Fall back to slower mne that may preload the data
            try:
                raw_data = read_raw_eeglab(filepath, verbose="WARNING",preload=False)
                return raw_data.ch_names
            except TypeError:
                try:
                    raw_epoched_data = read_epochs_eeglab(filepath, verbose="WARNING")
                    return raw_epoched_data.ch_names
                except FileNotFoundError as e:
                    logger.error("File not found: %s", e)
                    return []
            except OSError as e:
                logger.error("Skipping corrupt/unreadable file: %s", e)
                return []
            except Exception as e:
                logger.error("Error during channel extraction from: %s", e)
                raise

    def read_signal(self, logger: logging.Logger, filepath, channel):
        """Read signal from file for specific channel."""
        try:
            raw_data = read_raw_eeglab(filepath, verbose='ERROR')
            if channel in raw_data.ch_names:
                return raw_data.get_data(picks=channel)[0]

            return None # channel not found in this file
        except TypeError:
            try:
                # Probably because its an epoched file -> reading as epochs and concatenating
                # First load with mat to get channel names and n_epochs
                # (needed to construct events for mne)
                eeg = read_mat(filepath)
                eeg = eeg.get("EEG", eeg)   # handle nested structure
                chanlocs = eeg.get("chanlocs")
                labels = chanlocs.get('labels',[])
                if channel in labels:
                    n_epochs = eeg.get("trials")
                    events = np.vstack((np.arange(n_epochs, dtype=int),
                                        np.zeros(n_epochs, dtype=int),
                                        np.ones(n_epochs, dtype=int))).T
                    event_id = {"unknown": 1}
                    raw_epoched_data = read_epochs_eeglab(filepath, verbose="WARNING",
                                                          events=events, event_id=event_id)

                    # shape (n_epochs, n_times)
                    epoched_data = raw_epoched_data.get_data(picks=channel)[:,0,:]
                    continuous = epoched_data.flatten()
                    return continuous

                return None
            except FileNotFoundError as e:
                logger.error("File not found: %s", e)
                return None
        except Exception as e:
            logger.error("Error reading signal: %s", e)
            return None

    def get_file_info(self, logger: logging.Logger, filepath):
        """Get start datetime and file duration."""
        try:
            raw_data = read_raw_eeglab(filepath, verbose='WARNING', preload=True)
            info = raw_data.info

            file_duration = raw_data.duration

        except TypeError:
            raw_data = read_epochs_eeglab(filepath, verbose="WARNING")
            info = raw_data.info

            # n_times * n_epochs / sfreq
            file_duration = len(raw_data.times) * len(raw_data) / info["sfreq"]

        except RuntimeError:
            try:
                logger.warning("File is truncated, reconstructing it from .fdt file.")
                raw_data = read_raw_eeglab(filepath, verbose='WARNING', preload=False)
                info = raw_data.info
                fs = info["sfreq"]
                n_channels = info['nchan']
                fdt_file = filepath.replace('.set', '.fdt')
                eeg_data = np.fromfile(fdt_file, dtype=np.float32)
                file_duration = (len(eeg_data) // n_channels) / fs
            except Exception as e:
                logger.error("Runtime error during file info retrieval: %s \n Skipping file.",
                             e)
                return {}
        except Exception as e:
            logger.error("Error during file info retrieval: %s", e)
            raise

        start_datetime = info["meas_date"]
        return {"start_datetime": start_datetime, "file_duration": file_duration}

    def get_signal_data(self, logger: logging.Logger, filepath, channel):
        """Get complete signal information for specific channel.

        EEGLAB .set/.fdt files have no per-channel unit header field; mne's EEGLAB
        reader unconditionally assumes the stored values are microvolts and
        calibrates them to Volts internally (`CAL = 1e-6` in mne's eeglab reader).
        We ask it to convert back to uV (via its own `units=` scaling) so the
        returned signal always matches the file's original stored amplitude. Any
        further rescaling to a different unit happens downstream, driven by config.
        """
        try:
            raw_data = read_raw_eeglab(filepath, verbose='WARNING', preload=True)

            ch_type = raw_data.get_channel_types(picks=channel)[0]
            signal = raw_data.get_data(picks=channel, units={ch_type: 'uV'})[0]
            if np.all(np.isnan(signal)):
                logger.warning("Signal for channel %s contains only NaN values.", channel)
                return {}

            info = raw_data.info

        except TypeError:
            # Probably because its an epoched file -> reading as epochs and concatenating
            # First load with mat to get channel names and n_epochs
            # (needed to construct events for mne)
            eeg = read_mat(filepath)
            eeg = eeg.get("EEG", eeg)   # handle nested structure

            # Construct events because mne cannot handle epoched .set files without events
            n_epochs = eeg.get("trials")
            events = np.vstack((np.arange(n_epochs, dtype=int),
                                np.zeros(n_epochs, dtype=int),
                                np.ones(n_epochs, dtype=int))).T
            event_id = {"unknown": 1}
            raw_epoched_data = read_epochs_eeglab(filepath, verbose="WARNING",
                                                  events=events, event_id=event_id)

            ch_type = raw_epoched_data.get_channel_types(picks=channel)[0]
            # shape (n_epochs, n_times)
            epoched_data = raw_epoched_data.get_data(picks=channel,
                                                     units={ch_type: 'uV'})[:,0,:]

            signal = epoched_data.flatten()
            info = raw_epoched_data.info
        except RuntimeError:
            try:
                # File is truncated, try to reconstruct signal from .fdt file.
                # This reads the raw binary directly (bypassing mne's Volt calibration
                # entirely), so the values are already in EEGLAB's native uV scale.
                raw_data = read_raw_eeglab(filepath, verbose='WARNING', preload=False)
                info = raw_data.info
                n_channels = info['nchan']
                channel_idx = raw_data.ch_names.index(channel)
                fdt_file = filepath.replace('.set', '.fdt')
                eeg_data = np.fromfile(fdt_file, dtype=np.float32)
                # ensure it's divisible by n_channels
                eeg_data = eeg_data[:(len(eeg_data) // n_channels) * n_channels]

                # shape: (n_channels, n_samples)
                eeg_data = eeg_data.reshape((n_channels,-1), order='F')

                signal = eeg_data[channel_idx,:]
            except Exception as e:
                logger.error("Runtime error during data retrieval: %s \n Skipping file.",
                             e)
                return {}
        except Exception as e:
            logger.error("Error during data retrieval: %s", e)
            raise

        sampling_rate = info["sfreq"]
        return {
            "signal": signal,
            "sampling_rate": sampling_rate,
            "unit": "uV"
        }
