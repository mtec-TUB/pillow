"""
EDF file handler for PSG data processing.
"""

import pyedflib
import mne


class EDFHandler:
    """Handler for EDF files."""

    def get_channels(self, logger, filepath):
        """Extract channel names and frequencies from EDF files."""
        try:
            with pyedflib.EdfReader(filepath) as psg_f:
                labels = psg_f.getSignalLabels()
                # freqs = psg_f.getSampleFrequencies()
                # return labels, freqs
                return labels
        except:
            try:
                raw = mne.io.read_raw_edf(filepath, preload=False, verbose='WARNING')
                labels = raw.ch_names
                return labels
            except Exception as e:
                logger.error(f"Error processing EDF file with pyedflib and mne: {e}")
                logger.error(
                    "Maybe the repair_edfs.py script or EDF Browser header repairer can help."
                )
                raise

    def read_signal(self, logger, filepath, channel):
        """Read signal from EDF file for specific channel."""
        try:
            with pyedflib.EdfReader(filepath) as psg_f:
                ch_names_file = psg_f.getSignalLabels()
                if channel in ch_names_file:
                    ch_idx = ch_names_file.index(channel)
                    return psg_f.readSignal(ch_idx)
        except:
            try:
                raw = mne.io.read_raw_edf(filepath, include=channel,preload=True, verbose='WARNING')
                signal = raw.get_data()[0]
                return signal
            except Exception as e:
                logger.error(f"Error reading EDF signal from: {e}")
                logger.error("Maybe the repair_edfs.py script can help.")

    def get_signal_data(self, logger, filepath, channel):
        """Get complete EDF signal information for processing."""
        try:
            psg_f = pyedflib.EdfReader(filepath)

            ch_names = psg_f.getSignalLabels()

            select_ch_idx = ch_names.index(channel)

            start_datetime = psg_f.getStartdatetime()
            file_duration = psg_f.getFileDuration()

            ch_samples = psg_f.getNSamples()
            logger.info(f"Select channel samples: {ch_samples[select_ch_idx]}")
            ch_freq = psg_f.getSampleFrequencies()

            sampling_rate = ch_freq[select_ch_idx]
            signal = psg_f.readSignal(select_ch_idx)

            unit = psg_f.getPhysicalDimension(select_ch_idx)

            psg_f.close()

        except:
            try:
                raw = mne.io.read_raw_edf(filepath, include=channel,preload=False, verbose='WARNING')
                signal = raw.get_data()[0]
                start_datetime = raw.info['meas_date']
                sampling_rate = raw.info['sfreq']
                file_duration = raw.n_times / sampling_rate
                unit = raw.info['chs'][0]['unit']
                unit = mne._fiff.meas_info._unit2human[unit]

            except Exception as e:
                logger.error(f"Error processing EDF file with pyedflib and mne: {e}")
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