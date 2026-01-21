"""
MWT - Maintenance of Wakefulness Test
"""

from typing import Dict, List, Tuple
from datetime import datetime
from scipy.io import loadmat

from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("MWT")
class MWT(BaseDataset):
    """MWT - Maintenance of Wakefulness Testdataset"""

    def __init__(self):
        super().__init__("MWT","MWT - Maintenance of Wakefulness Test")

        self._file_handler = None  # MWT uses custom Mat handling directly implemented here

    def _setup_dataset_config(self):
        #### No usable scoring (BERN)
        self.ann2label = {
        }
    
    
        self.channel_names = ['E1', 'E2', 'eeg_O1', 'eeg_O2']
    
        
        self.channel_types = {'analog': ['eeg_O1', 'E2', 'E1', 'eeg_O2'], 'digital': []}

                
        
        self.channel_groups = {'eeg_eog': ['E1', 'E2', 'eeg_O1', 'eeg_O2']}
        
        self.file_extensions = {'psg_ext': '*.mat',
                                'ann_ext': '*.mat'
                            }

    def dataset_paths(self):
        return [
            "MWT - Maintenance of Wakefulness Test",
            "MWT - Maintenance of Wakefulness Test"
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
                "sampling_rate": sampling_rate,
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
        labels_1 = ann_f[0,0]['labels_O1']
        fs = ann_f[0,0]['fs'][0,0]
        
        epoch_duration = 30  # MWT uses 30-second epochs

        for i, stage in enumerate(labels_1):
            if (i/fs)%epoch_duration == 0:
                ann_stage_events.append({
                    'Start': i/fs,
                    'Duration': epoch_duration,
                    'Stage': stage[0]
                })
            else:
                assert stage[0] == ann_stage_events[-1]['Stage'], f"label changes at sec {i/fs} in between 30sec epoch ({stage[0]}, {ann_stage_events[-1]['Stage']}))"

        return ann_stage_events, None

