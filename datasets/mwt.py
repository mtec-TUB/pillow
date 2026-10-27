"""
MWT - Maintenance of Wakefulness Test
"""
import logging

from scipy.io import loadmat

from datasets.base import BaseDataset
from datasets.registry import register_dataset
from psg_processing.utils import SleepStage


@register_dataset("MWT")
class MWT(BaseDataset):
    """MWT - Maintenance of Wakefulness Testdataset"""

    def __init__(self):
        super().__init__("MWT","MWT - Maintenance of Wakefulness Test")
        self.use_both_scorers = True

        self._file_handler = None  # MWT uses custom Mat handling directly implemented here

    def _setup_dataset_config(self):
        # Dataset uses BERN scoring (0-wake, 1-MSE, 2-MSEc, 3-ED), we map more than 15s of MSE (micro-sleep-event) to "sleep" and the rest to "wake"
        # see also https://doi.org/10.1093/sleep/zsz163
        self.ann2label = {
            "wake": SleepStage.W,
            "sleep": SleepStage.SLEEP,
        }

        # https://zenodo.org/records/3251716
        self.inter_dataset_mapping = {
            "eeg_O1": self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            "eeg_O2": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "E1": self.Mapping(self.TTRef.EL, self.TTRef.LPA),
            "E2": self.Mapping(self.TTRef.ER, self.TTRef.LPA),
        }

        self.channel_names = ['E1', 'E2', 'eeg_O1', 'eeg_O2']


        self.channel_types = {'analog': ['eeg_O1', 'E2', 'E1', 'eeg_O2'], 'digital': []}



        self.channel_groups = {'eeg_eog': ['E1', 'E2', 'eeg_O1', 'eeg_O2']}

        self.file_extensions = {'psg_ext': '*.mat',
                                'ann_ext': '*.mat'
                            }

    def dataset_paths(self):
        return [
            '',
            ''
        ]

    def get_channels(self, logger: logging.Logger, filepath):
        """Extract channel names and frequencies from mat files."""
        try:
            psg_f = loadmat(filepath)['Data']
            return psg_f.dtype.names
        except Exception as e:
            logger.error("Error reading mat file: %s", e)
            return []

    def read_signal(self, logger: logging.Logger, filepath, channel):
        """Read signal from Mat file for specific channel."""
        try:
            psg_f = loadmat(filepath)['Data']
            if channel in psg_f.dtype.names:
                return psg_f[0,0][channel][:, 0]
        except Exception as e:
            logger.error("Error reading mat signal: %s", e)
        return None

    def get_file_info(self, logger: logging.Logger, filepath):
        """Get start datetime and file duration."""
        start_datetime = None
        psg_f = loadmat(filepath)['Data']
        file_duration = psg_f[0,0]['num_Labels'][0,0] / psg_f[0,0]['fs'][0,0]
        return {"start_datetime": start_datetime, "file_duration": file_duration}

    def get_signal_data(self, logger: logging.Logger, filepath, channel):
        """Get complete EDF signal information for processing."""
        try:
            psg_f = loadmat(filepath)['Data']

            signal = psg_f[0,0][channel][:, 0]
            samples = psg_f[0,0]['num_Labels'][0,0]
            assert len(signal) == samples
            logger.info("Select channel samples: %d", samples)

            sampling_rate = psg_f[0,0]['fs'][0,0]
            # file_duration = samples / sampling_rate

            return {
                "signal": signal,
                "unit": "uV",  # MWT signals are assumed to be in microvolts (uV)
                "sampling_rate": float(sampling_rate)
            }
        except Exception as e:
            logger.error("Error processing mat file: %s", e)
            raise

    def ann_parse(self, ann_fname: str):
        """
        Parse MWT annotation files.
        """
        ann_stage_events = []

        ann_f = loadmat(ann_fname)['Data']
        fs = int(ann_f[0,0]['fs'][0,0])

        epoch_duration = 30

        for scorer_label in ['labels_O1', 'labels_O2']:
            scorer_events = []
            labels = ann_f[0,0][scorer_label]
            for i in range(0, len(labels), fs*epoch_duration):
                epoch = labels[i:i + fs*epoch_duration]
                max_run = 0
                current = 0
                for x in epoch:
                    if x == 1:
                        current += 1
                        max_run = max(max_run, current)
                    else:
                        current = 0

                contains_sleep = max_run > fs*15

                if contains_sleep:
                    scorer_events.append({
                        'Start': i/fs,
                        'Duration': epoch_duration,
                        'Stage': "sleep"
                    })
                else:
                    scorer_events.append({
                        'Start': i/fs,
                        'Duration': epoch_duration,
                        'Stage': "wake"
                    })
            ann_stage_events.append(scorer_events)


        return ann_stage_events, None, None, None
