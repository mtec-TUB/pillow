import os
import pyedflib
from mne.io import read_raw_edf
from warnings import catch_warnings

class EDFHandler:
    """Handler for EDF files
    """

    def get_channels(self, logger, filepath):
        """Extract channel names from file."""
        try:
            with catch_warnings(record=True) as w:  # to handle duplicate channel names
                raw = read_raw_edf(filepath, preload=False, verbose='WARNING')
                if w:
                    if w[0].message.args[0].startswith("Channel names are not unique"):
                        # duplicate channel names detected -> mne automatically maps them to unique names 
                        # We wouldnt find them with the given channel name
                        # We have to find the duplicates and remove the unique identifier
                        duplicates = []
                        for ch in raw.ch_names:
                            # try to load the file including only this channel name
                            dupl_test_raw = read_raw_edf(filepath, include=ch,preload=False, verbose='WARNING')
                            if ch not in dupl_test_raw.ch_names:
                                # Channel not found because it is one of the mapped unique names
                                duplicates.append(ch)
                        prefix = os.path.commonprefix(duplicates)
                        orig_ch_name = prefix.rstrip("-")
                        labels = list(set(raw.ch_names) - set(duplicates)) + [orig_ch_name]
                    else:
                        logger.info("Warning during channel extraction with mne library: " + str(w[0].message))
                        labels = raw.ch_names
                else:
                    labels = raw.ch_names
            return labels
        except NotImplementedError as e:
            # This can happen for some EDF files which e.g. do not have the standard .edf extension, but are still internally in EDF format (.rec)
            try:
                with open(filepath, "rb") as f:
                    raw = read_raw_edf(f, preload=True, verbose="WARNING")   # only supported for mne version >= 1.10
                    labels = raw.ch_names
                    return labels
            except Exception as e:
                raise
        except KeyboardInterrupt:
            # Always re-raise KeyboardInterrupt
            raise
        except Exception as e:
            logger.error(f"Error during channel extraction: {e}")
            logger.error("Maybe EDF Browser header repair can help.")
            return []

    def read_signal(self, logger, filepath, channel):
        """Read signal from file for specific channel."""
        try:
            with catch_warnings(record=True) as w:
                raw = read_raw_edf(filepath, include=channel,preload=True, verbose='WARNING')
                if w:
                    if w[0].message.args[0].startswith("Channel names are not unique"):
                        channel = raw.ch_names[0]   # take only the first of the duplicate channels
                    else:
                        logger.info("Warning during signal extraction with mne library: " + str(w[0].message))
            if channel in raw.ch_names:
                signal = raw.get_data()[0]
                return signal
        except NotImplementedError as e:
            # This can happen for some EDF files which e.g. do not have the standard .edf extension, but are still internally in EDF format (.rec)
            try:
                with open(filepath, "rb") as f:
                    raw = read_raw_edf(f, include=channel, preload=True, verbose="WARNING")   # only supported for mne version >= 1.10
                    if channel in raw.ch_names:
                        signal = raw.get_data()[0]
                        return signal
            except Exception as e:
                raise
        except KeyboardInterrupt:
            # Always re-raise KeyboardInterrupt
            raise
        except Exception as e:
            logger.error(f"Error during signal extraction: {e}")
            logger.error("Maybe EDF Browser header repair can help.")
        return None     # channel was not found in this file
        
    def get_file_info(self, logger, filepath):
        """Get start datetime and file duration."""
        try:
            raw = read_raw_edf(filepath, preload=False, verbose='WARNING')                
            start_datetime = raw.info['meas_date']
            file_duration = raw.duration
        except NotImplementedError as e:
            # This can happen for some EDF files which e.g. do not have the standard .edf extension, but are still internally in EDF format (.rec)
            try:
                with open(filepath, "rb") as f:
                    raw = read_raw_edf(f, preload=True, verbose="WARNING")   # only supported for mne version >= 1.10
                    start_datetime = raw.info['meas_date']
                    file_duration = raw.duration
            except Exception as e:
                raise
        except KeyboardInterrupt:
            # Always re-raise KeyboardInterrupt
            raise
        except Exception as e:
            logger.error(f"Error during start_datetime retrieval: {e}")
            logger.error("Maybe EDF Browser header repair can help.")
            raise

        return {"start_datetime": start_datetime,
                "file_duration": file_duration}

    def get_signal_data(self, logger, filepath, channel):
        """Get complete signal information for specific channel.

        mne always calibrates recognized voltage-family channels (uV/mV/V) to Volts
        internally. We ask it to convert back to the unit originally recorded in the
        EDF header (via its own `units=` scaling), so the returned signal always
        matches the file's original physical amplitude, like a raw pyedflib read.
        Any further rescaling to a different unit happens downstream, driven by config.
        """
        try:
            with catch_warnings(record=True) as w:
                raw = read_raw_edf(filepath, include=[channel], preload=False, verbose='WARNING')
                if w:
                    if w[0].message.args[0].startswith("Channel names are not unique"):
                        channel = raw.ch_names[0]    # take only the first of the duplicate channels
                    elif w[0].message.args[0].startswith("Highpass cutoff frequency"):
                        pass
                    else:
                        logger.warning("Warning during data retrieval with mne library: " + str(w[0].message))

            sampling_rate = raw.info['sfreq']

        except NotImplementedError as e:
            # This can happen for some EDF files which e.g. do not have the standard .edf extension, but are still internally in EDF format (.rec)
            try:
                with open(filepath, "rb") as f:
                    raw = read_raw_edf(f, include=[channel], preload=True, verbose="WARNING")   # only supported for mne version >= 1.10
                    sampling_rate = raw.info['sfreq']
            except Exception as e:
                raise
        except KeyboardInterrupt:
            # Always re-raise KeyboardInterrupt
            raise
        except Exception as e:
            logger.error(f"Error during data retrieval: {e}")
            logger.error("Maybe EDF Browser header repair can help.")
            raise

        mne_unit = raw._orig_units.get(channel, "n/a")
        if mne_unit == "n/a":
            # mne only recognizes uV/mV/V; any other original unit (e.g. "%" for SpO2)
            # gets silently replaced with "n/a" in raw._orig_units (and left unconverted
            # in the signal values), so re-read the true unit straight from the EDF header.
            signal = raw.get_data(picks=channel)[0]
            unit = "n/a"
            try:
                with pyedflib.EdfReader(filepath) as f:
                    labels = f.getSignalLabels()
                    unit = f.getPhysicalDimension(labels.index(channel))
            except Exception as e:
                logger.warning(f"Could not recover original unit for channel {channel} from EDF header: {e}")
        else:
            ch_type = raw.get_channel_types(picks=channel)[0]
            signal = raw.get_data(picks=channel, units={ch_type: mne_unit})[0]
            # EDF header fields are ASCII-only (per spec); mne represents the micro
            # prefix with the unicode MICRO SIGN (µ/μ), which downstream EDF writers
            # (pyedflib) cannot round-trip (it gets corrupted, e.g. into "AuV"), so
            # normalize to the ascii 'u' pyedflib itself uses when reading the header.
            unit = mne_unit.replace("µ", "u").replace("μ", "u")

        return {
            "signal": signal,
            "sampling_rate": sampling_rate,
            "unit": unit,
        }
