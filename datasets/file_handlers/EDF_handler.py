import os
import mne
from warnings import catch_warnings

class EDFHandler:
    """Handler for EDF files
    """

    def get_channels(self, logger, filepath):
        """Extract channel names from file."""
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
            logger.error(f"Error during channel extraction: {e}")
            logger.error(
                "Maybe EDF Browser header repair can help."
            )
            return []

    def read_signal(self, logger, filepath, channel):
        """Read signal from file for specific channel."""
        try:
            with catch_warnings(record=True) as w:
                raw = mne.io.read_raw_edf(filepath, include=channel,preload=True, verbose='WARNING')
                if w:
                    if w[0].message.args[0].startswith("Channel names are not unique"):
                        channel = raw.ch_names[0]   # take only the first of the duplicate channels
                    else:
                        logger.warning(str(w[0].message))
            if channel in raw.ch_names:
                signal = raw.get_data()[0]
                return signal
        except KeyboardInterrupt:
            # Always re-raise KeyboardInterrupt
            raise
        except Exception as e:
            logger.error(f"Error during signal extraction: {e}")
            logger.error(
                "Maybe EDF Browser header repair can help."
            )
        return None     # channel was not found in this file
        
    def get_start_datetime(self, logger, filepath):
        """Get start datetime of file."""
        try:
            raw = mne.io.read_raw_edf(filepath, preload=False, verbose='WARNING')                
            start_datetime = raw.info['meas_date']

        except KeyboardInterrupt:
            # Always re-raise KeyboardInterrupt
            raise
        except Exception as e:
            logger.error(f"Error during start_datetime retrieval: {e}")
            logger.error("Maybe EDF Browser header repair can help.")
            raise

        return start_datetime

    def get_signal_data(self, logger, filepath, channel):
        """Get complete signal information for specific channel."""
        try:
            with catch_warnings(record=True) as w:
                raw = mne.io.read_raw_edf(filepath, include=[channel], preload=False, verbose='WARNING')
                if w:
                    if w[0].message.args[0].startswith("Channel names are not unique"):
                        channel = raw.ch_names[0]    # take only the first of the duplicate channels
                    else:
                        logger.warning(str(w[0].message))
            
            signal = raw.get_data(picks=channel)[0]
            sampling_rate = raw.info['sfreq']
            file_duration = raw.n_times / sampling_rate
            unit = raw.info['chs'][0]['unit']
            unit = mne._fiff.meas_info._unit2human[unit]

        except KeyboardInterrupt:
            # Always re-raise KeyboardInterrupt
            raise
        except Exception as e:
            logger.error(f"Error during data retrieval: {e}")
            logger.error("Maybe EDF Browser header repair can help.")
            raise

        return {
            "signal": signal,
            "sampling_rate": sampling_rate,
            "unit": unit,
            "file_duration": file_duration,
        }
