from mne.io import read_raw_brainvision
from mne.io.brainvision.brainvision import _get_hdr_info

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
        """Get complete signal information for specific channel.

        mne always calibrates recognized voltage-family channels (uV/mV/V) to Volts
        internally. We ask it to convert back to the unit originally recorded in the
        .vhdr header (via its own `units=` scaling), so the returned signal always
        matches the file's original physical amplitude. Any further rescaling to a
        different unit happens downstream, driven by config.
        """
        try:
            raw_data = read_raw_brainvision(filepath, verbose='WARNING', preload=False)
            info = raw_data.info
        except Exception as e:
            logger.error(f"Error during data retrieval: {e}")
            raise

        sampling_rate = info["sfreq"]

        mne_unit = raw_data._orig_units.get(channel, "n/a")
        if mne_unit == "n/a":
            # mne only recognizes uV/mV/V; any other original unit (e.g. "%") gets
            # silently replaced with "n/a" in raw._orig_units (and left unconverted
            # in the signal values), so re-parse the .vhdr header directly (before
            # mne's sanitization step discards the real unit).
            signal = raw_data.get_data(picks=channel)[0]
            unit = "n/a"
            try:
                # eog/misc only steer channel-kind classification, not unit parsing,
                # so their values here don't need to match read_raw_brainvision's.
                _, _, _, _, _, _, _, orig_units = _get_hdr_info(
                    filepath, eog=(), misc="auto", scale=1.0
                )
                unit = orig_units.get(channel, unit)
            except Exception as e:
                logger.warning(f"Could not recover original unit for channel {channel} from BrainVision header: {e}")
        else:
            ch_type = raw_data.get_channel_types(picks=channel)[0]
            signal = raw_data.get_data(picks=channel, units={ch_type: mne_unit})[0]
            # EDF header fields (and other downstream writers) are ASCII-only; mne
            # represents the micro prefix with the unicode MICRO SIGN (µ/μ), which
            # doesn't round-trip safely, so normalize to plain ascii 'u'.
            unit = mne_unit.replace("µ", "u").replace("μ", "u")

        return {
            "signal": signal,
            "sampling_rate": sampling_rate,
            "unit": unit
        }
