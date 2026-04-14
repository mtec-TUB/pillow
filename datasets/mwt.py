"""
MWT - Maintenance of Wakefulness Test
"""

from typing import Dict, List, Tuple
from datetime import datetime
from scipy.io import loadmat
import numpy as np

from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("MWT")
class MWT(BaseDataset):
    """MWT - Maintenance of Wakefulness Testdataset"""

    def __init__(self):
        super().__init__("MWT","MWT - Maintenance of Wakefulness Test")

        self._file_handler = None  # MWT uses custom Mat handling directly implemented here

    def _setup_dataset_config(self):
        # Dataset uses BERN scoring (0-wake, 1-MSE, 2-MSEc, 3-ED), we map more than 15s of MSE (micro-sleep-event) to "sleep" and the rest to "wake"
        # see also https://doi.org/10.1093/sleep/zsz163
        self.ann2label = {
            "wake": 0,
            "sleep": 1
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
    
    def get_channels(self, logger, filepath):
        """Extract channel names and frequencies from mat files."""
        try:
            psg_f = loadmat(filepath)['Data']
            return psg_f.dtype.names
        except Exception as e:
            logger.error(f"Error reading mat file {filepath}: {e}")
            return []

    def read_signal(self, logger, filepath, channel):
        """Read signal from Mat file for specific channel."""
        try:
            psg_f = loadmat(filepath)['Data']
            if channel in psg_f.dtype.names:
                return psg_f[0,0][channel][:, 0]
        except Exception as e:
            logger.error(f"Error reading mat signal from {filepath}: {e}")
        return None

    def get_signal_data(self, logger, filepath, channel):
        """Get complete EDF signal information for processing."""
        try:
            psg_f = loadmat(filepath)['Data']

            signal = psg_f[0,0][channel][:, 0]
            samples = psg_f[0,0]['num_Labels'][0,0]
            assert len(signal) == samples
            logger.info(f"Select channel samples: {samples}")

            sampling_rate = psg_f[0,0]['fs'][0,0]
            file_duration = samples / sampling_rate

            return {
                "signal": signal,
                "sampling_rate": float(sampling_rate),
                "start_datetime": None,
                "file_duration": file_duration,
            }
        except Exception as e:
            logger.error(f"Error processing mat file {filepath}: {e}")
            raise
    
    def ann_parse(self, ann_fname: str) -> Tuple[List[List[Dict]], datetime]:
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
    
    def ann_label(self, logger, ann_stage_events: List[List[Dict]], epoch_duration: int) -> np.ndarray:
        """
        Convert multi-scorer sleep stage events to epoch-wise labels for ISRUC dataset.
        Returns 2D array (n_epochs, n_scorers).
        """
        labels = [np.array([]), np.array([])]

        for i, annotation in enumerate(ann_stage_events):  # two scorers
            total_duration = 0
            for event in annotation:
                onset_sec = event['Start']
                duration_sec = event['Duration']
                ann_str = event['Stage']

                # Sanity check
                assert onset_sec == total_duration, f"Onset sec of epoch is {onset_sec} but last epoch ended at {total_duration}"

                # Get label value
                if ann_str in self.ann2label:
                    label = self.ann2label[ann_str]
                else:
                    logger.info(f"Something unexpected: label {ann_str} not found")
                    raise Exception(f"Something unexpected: label {ann_str} not found")

                # Compute # of epoch for this stage
                if duration_sec % epoch_duration != 0:
                    logger.info(f"Something wrong: {duration_sec} {epoch_duration}")
                    raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
                duration_epoch = int(duration_sec / epoch_duration)

                # Generate sleep stage labels
                label_epoch = np.ones(duration_epoch, dtype=np.int32) * label
                labels[i] = np.append(labels[i], label_epoch)

                total_duration += duration_sec

                # logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
                #     onset_sec, duration_sec, label, ann_str
                # ))

        # Pad shorter annotation to match longer one
        if len(labels[0]) != len(labels[1]):
            max_len = max(len(labels[0]), len(labels[1]))
            labels[0] = np.pad(labels[0], (0, max_len - len(labels[0])), mode='constant', constant_values=6)
            labels[1] = np.pad(labels[1], (0, max_len - len(labels[1])), mode='constant', constant_values=6)

        labels = np.array(labels).T  # Transpose to (n_epochs, n_scorers)
        
        return labels

