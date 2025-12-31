import h5py
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("DOD-H")
class DODH(BaseDataset):
    """DOD-H (Dreem Open Dataset - Healthy) dataset."""
    
    def __init__(self):
        super().__init__("DOD-H","DOD-H - Dreem Open Dataset - Healthy")

    def _setup_dataset_config(self):
        self.ann2label = {
            0: 0,   # Wake
            1: 1,   # NREM Stage 1
            2: 2,   # NREM Stage 2
            3: 3,   # NREM Stage 3
            4: 4,   # REM sleep
            -1: 6   # Unscored
        }
        
        
        self.alias_mapping = {
            "F4_O2": ["signals/eeg/F4_O2"],
            "FP2_F4": ["signals/eeg/FP2_F4"],
            "F3_M2": ["signals/eeg/F3_M2"],
            "F3_O1": ["signals/eeg/F3_O1"],
            "F4_M1": ["signals/eeg/F4_M1"],
            "C3_M2": ["signals/eeg/C3_M2"],
            "FP1_M2": ["signals/eeg/FP1_M2"],
            "F3_F4": ["signals/eeg/F3_F4"],
            "FP2_O2": ["signals/eeg/FP2_O2"],
            "FP1_O1": ["signals/eeg/FP1_O1"],
            "FP2_M1": ["signals/eeg/FP2_M1"],
            "FP1_F3": ["signals/eeg/FP1_F3"],
            "EOG1": ["signals/eog/EOG1"],
            "EOG2": ["signals/eog/EOG2"],
            "EMG": ["signals/emg/EMG"],
            "ECG": ["signals/emg/ECG"]
        }
        
        
        self.channel_names = [
            'signals/eeg/F4_O2', 'signals/eeg/FP2_F4', 'signals/eeg/F3_M2', 'signals/eeg/F3_O1',
            'signals/eeg/F4_M1', 'signals/eeg/C3_M2', 'signals/eeg/FP1_M2', 'signals/eeg/F3_F4',
            'signals/eeg/FP2_O2', 'signals/eeg/FP1_O1', 'signals/eeg/FP2_M1', 'signals/eeg/FP1_F3',
            'signals/eog/EOG1', 'signals/eog/EOG2',
            'signals/emg/EMG',
            'signals/emg/ECG'
        ]
        
        
        self.channel_types = {
            'analog': [
                'signals/eeg/F4_O2', 'signals/eeg/FP2_F4', 'signals/emg/EMG', 'signals/eeg/F3_M2',
                'signals/emg/ECG', 'signals/eeg/F3_O1', 'signals/eeg/F4_M1', 'signals/eeg/C3_M2',
                'signals/eeg/FP1_M2', 'signals/eeg/F3_F4', 'signals/eeg/FP2_O2', 'signals/eeg/FP1_O1',
                'signals/eeg/FP2_M1', 'signals/eog/EOG2', 'signals/eeg/FP1_F3', 'signals/eog/EOG1'
            ],
            'digital': []
        }
        
        
        
        self.channel_groups = {
            'eeg_eog': ['signals/eeg/F4_O2', 'signals/eeg/FP2_F4', 'signals/eeg/F3_M2', 'signals/eeg/F3_O1', 'signals/eeg/F4_M1', 'signals/eeg/C3_M2', 'signals/eeg/FP1_M2', 'signals/eeg/F3_F4', 'signals/eeg/FP2_O2', 'signals/eeg/FP1_O1', 'signals/eeg/FP2_M1', 'signals/eeg/FP1_F3', 'signals/eog/EOG2', 'signals/eog/EOG1'],
            'emg': ['signals/emg/EMG'],
            'ecg': ['signals/emg/ECG']
        }
        
        self.file_extensions = {
            'psg_ext': '*.h5',
            'ann_ext': '*.h5'  # Annotations are embedded in data H5 files
        }
    
    def dataset_paths(self) -> Tuple[str, str]:
        """
        DOD-H dataset paths.
        Data and annotations are in the same H5 files.
        """
        data_dir = "DOD-H - Dreem Open Dataset - Healthy"
        ann_dir = "DOD-H - Dreem Open Dataset - Healthy"
        return data_dir, ann_dir
    
    def collect_h5_dataset(self, name, obj, dataset):
        """
        Helper function to collect H5 datasets during file iteration.

        Args:
            name: Dataset name
            obj: H5 object
            dataset: List to append dataset information to
        """
        if isinstance(obj, h5py.Dataset):
            dataset.append((name, obj[:]))
    
    
    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        DOD-H doesn't use separate annotation files.
        Annotations are embedded in H5 files.
        """
        ann_stage_events = []
        epoch_duration = 30  # DOD-H uses 30-second epochs

        with h5py.File(ann_fname, "r") as f:
            dataset = []
            f.visititems(lambda name, obj: self.collect_h5_dataset(name, obj, dataset))

            ch_names = [data[0] for data in dataset]
            
            hypnogram_idx = ch_names.index("hypnogram")
            stages = dataset[hypnogram_idx][1]
            for i, stage in enumerate(stages):
                onset = i * epoch_duration
                duration = epoch_duration
                ann = {
                    'Stage': stage,
                    'Start': onset,
                    'Duration': duration
                }
                ann_stage_events.append(ann)

        return ann_stage_events, None
    