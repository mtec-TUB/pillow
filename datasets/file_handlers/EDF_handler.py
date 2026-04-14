import os
import pyedflib
import mne
from warnings import catch_warnings

class EDFHandler:
    """Handler for EDF files
    Tries to get file content with pyedflib and falls back to mne if pyedflib fails
    """

    def get_channels(self, logger, filepath):
        """Extract channel names from file."""
        try:
            with pyedflib.EdfReader(filepath) as psg_f:
                labels = psg_f.getSignalLabels()
                # freqs = psg_f.getSampleFrequencies()
                # return labels, freqs
                return labels
        except KeyboardInterrupt:
            # Always re-raise KeyboardInterrupt
            raise
        except:
            # fall back to mne because it handles edfs more robustly
            try:
                with catch_warnings(record=True) as w:  # to handle duplicate channel names
                    raw = mne.io.read_raw_edf(filepath, preload=False, verbose='WARNING')
                    if w:
                        if w[0].message.args[0].startswith("Channel names are not unique"):
                            # duplicate channel names detected -> mne automatically maps them two unique names
                            # we want the orig channel name
                            # so we have to find the duplicates and remove the unique identifier
                            duplicates = []
                            for ch in raw.ch_names:
                                # try to load the file including only this channel name
                                dupl_test_raw = mne.io.read_raw_edf(filepath, include=ch,preload=False, verbose='WARNING')
                                if ch not in dupl_test_raw.ch_names:
                                    # Channel not found because it is one of the mapped unique names
                                    duplicates.append(ch)
                            prefix = os.path.commonprefix(duplicates)
                            orig_ch_name = prefix.rstrip("-")
                            labels = list(set(raw.ch_names) - set(duplicates)) + [orig_ch_name]
                        else:
                            logger.warning(str(w[0].message))
                            labels = raw.ch_names
                    else:
                        labels = raw.ch_names
                return labels
            except KeyboardInterrupt:
                # Always re-raise KeyboardInterrupt
                raise
            except Exception as e:
                logger.error(f"Error during channel extraction from {filepath}: {e}")
                logger.error(
                    "Maybe the repair_edfs.py script or EDF Browser header repairer can help."
                )
                return []

    def read_signal(self, logger, filepath, channel):
        """Read signal from file for specific channel."""
        try:
            with pyedflib.EdfReader(filepath) as psg_f:
                ch_names_file = psg_f.getSignalLabels()
                if channel in ch_names_file:
                    ch_idx = ch_names_file.index(channel)
                    return psg_f.readSignal(ch_idx)
        except KeyboardInterrupt:
            # Always re-raise KeyboardInterrupt
            raise
        except:
            # fall back to mne because it handles edfs more robustly
            try:
                with catch_warnings(record=True) as w:
                    raw = mne.io.read_raw_edf(filepath, include=channel,preload=True, verbose='WARNING')
                    if w:
                        if w[0].message.args[0].startswith("Channel names are not unique"):
                            channel = raw.ch_names[0]   # take only the first of the duplicate channels
                        else:
                            logger.warning(str(w[0].message))
                if channel in raw.ch_names:
                    signal = raw.get_data()
                    return signal
            except KeyboardInterrupt:
                # Always re-raise KeyboardInterrupt
                raise
            except Exception as e:
                logger.error(f"Error during signal extraction from {filepath}: {e}")
                logger.error(
                    "Maybe the repair_edfs.py script or EDF Browser header repairer can help."
                )
        return None     # channel was not found in this file
        

    def get_signal_data(self, logger, filepath, channel):
        """Get complete signal information for specific channel."""
        try:
            with pyedflib.EdfReader(filepath) as psg_f:

                start_datetime = psg_f.getStartdatetime()
                file_duration = psg_f.getFileDuration()

                ch_names = psg_f.getSignalLabels()
                select_ch_idx = ch_names.index(channel)

                ch_freq = psg_f.getSampleFrequencies()
                sampling_rate = ch_freq[select_ch_idx]

                signal = psg_f.readSignal(select_ch_idx)

                unit = psg_f.getPhysicalDimension(select_ch_idx)
        except KeyboardInterrupt:
            # Always re-raise KeyboardInterrupt
            raise
        except:
            try:
                with catch_warnings(record=True) as w:
                    raw = mne.io.read_raw_edf(filepath, include=[channel], preload=False, verbose='WARNING')
                    if w:
                        if w[0].message.args[0].startswith("Channel names are not unique"):
                            channel = raw.ch_names[0]    # take only the first of the duplicate channels
                        else:
                            logger.warning(str(w[0].message))
                
                signal = raw.get_data(picks=channel)[0]
                start_datetime = raw.info['meas_date']
                sampling_rate = raw.info['sfreq']
                file_duration = raw.n_times / sampling_rate
                unit = raw.info['chs'][0]['unit']
                unit = mne._fiff.meas_info._unit2human[unit]

            except KeyboardInterrupt:
                # Always re-raise KeyboardInterrupt
                raise
            except Exception as e:
                logger.error(f"Error during data retrieval {filepath}: {e}")
                logger.error(
                    "Maybe the repair_edfs.py script or EDF Browser header repairer can help."
                )
                raise

        return {
            "signal": signal,
            "sampling_rate": sampling_rate,
            "unit": unit,
            "start_datetime": start_datetime,
            "file_duration": file_duration,
        }
