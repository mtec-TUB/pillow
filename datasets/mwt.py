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

