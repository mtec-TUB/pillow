"""
EDF file handler for PSG data processing.
"""

import os
import pyedflib
import mne
from warnings import catch_warnings


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
                with catch_warnings(record=True) as w:
                    raw = mne.io.read_raw_edf(filepath, preload=False, verbose='WARNING')
                    if w:
                        if w[0].message.args[0].startswith("Channel names are not unique"):
                            duplicates = []
                            for ch in raw.ch_names:
                                dupl_test_raw = mne.io.read_raw_edf(filepath, include=ch,preload=False, verbose='WARNING')
                                if ch not in dupl_test_raw.ch_names:
                                    # Channel not found because it was mapped to a different name due to duplication
                                    duplicates.append(ch)
                            prefix = os.path.commonprefix(duplicates)
                            common = prefix.rstrip("-")
                                
                            labels = list(set(raw.ch_names) - set(duplicates)) + [common]
                    else:
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
                with catch_warnings(record=True) as w:
                    raw = mne.io.read_raw_edf(filepath, include=channel,preload=True, verbose='WARNING')
                    if w:
                        if w[0].message.args[0].startswith("Channel names are not unique"):
                            channel = raw.ch_names[0]
                        else:
                            logger.warning(str(w.message))
                if channel in raw.ch_names:
                    signal = raw.get_data()
                    return signal
            except Exception as e:
                logger.error(f"Error processing EDF file with pyedflib and mne: {e}")
                logger.error(
                    "Maybe the repair_edfs.py script or EDF Browser header repairer can help."
                )
        return None

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
                with catch_warnings(record=True) as w:
                    raw = mne.io.read_raw_edf(filepath, include=channel,preload=False, verbose='WARNING')
                    if w:
                        if w[0].message.args[0].startswith("Channel names are not unique"):
                            channel = raw.ch_names[0]
                        else:
                            logger.warning(str(w.message))
                
                signal = raw.get_data(picks=channel)[0]
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