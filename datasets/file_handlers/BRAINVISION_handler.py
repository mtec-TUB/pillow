from mne.io import read_raw_brainvision
from mne.io.brainvision.brainvision import _get_hdr_info
from mne import _fiff

class BRAINVISIONHandler:
    """Handler for Brainvision VHDR files."""

    def get_channels(self, logger, filepath):
        """Extract channel names file."""
        try:
            raw_data = read_raw_brainvision(filepath, verbose="WARNING",preload=False)
            return raw_data.ch_names
        except Exception as e:
            logger.error(f"Error during channel extraction from: {e}")
            raise

    def read_signal(self, logger, filepath, channel):
        """Read signal from file for specific channel."""
        try:
            raw_data = read_raw_brainvision(filepath, verbose='ERROR')
            if channel in raw_data.ch_names:
                return raw_data.get_data(picks=channel)[0]
            else:
                return None # channel not found in this file
        except Exception as e:
            logger.error(f"Error reading signal: {e}")
            return None
        
    def get_file_info(self, logger, filepath):
        """Get start datetime and file duration."""
        try:
            raw_data = read_raw_brainvision(filepath, verbose='WARNING', preload=False)
            info = raw_data.info
            file_duration = raw_data.duration
            start_datetime = info["meas_date"]
        except Exception as e:
            logger.error(f"Error during file info retrieval: {e}")
            raise

        return {"start_datetime": start_datetime, "file_duration": file_duration}

    def get_signal_data(self, logger, filepath, channel):
        """Get complete signal information for specific channel."""
        try:
            raw_data = read_raw_brainvision(filepath, verbose='WARNING', preload=False)
            signal = raw_data.get_data(picks=channel)[0]
            info = raw_data.info
        except Exception as e:
            logger.error(f"Error during data retrieval: {e}")
            raise

        sampling_rate = info["sfreq"]
        unit = info['chs'][info['ch_names'].index(channel)]['unit']
        unit = _fiff.meas_info._unit2human[unit]

        if raw_data._orig_units.get(channel, "n/a") == "n/a":
            # mne only recognizes uV/mV/V; any other original unit (e.g. "%") gets
            # silently replaced with "n/a" in raw._orig_units, so re-parse the .vhdr
            # header directly (before mne's sanitization step discards the real unit).
            try:
                # eog/misc only steer channel-kind classification, not unit parsing,
                # so their values here don't need to match read_raw_brainvision's.
                _, _, _, _, _, _, _, orig_units = _get_hdr_info(
                    filepath, eog=(), misc="auto", scale=1.0
                )
                unit = orig_units.get(channel, unit)
            except Exception as e:
                logger.warning(f"Could not recover original unit for channel {channel} from BrainVision header: {e}")

        return {
            "signal": signal,
            "sampling_rate": sampling_rate,
            "unit": unit
        }
