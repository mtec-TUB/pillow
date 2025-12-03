import h5py
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from .base import BaseDataset
from .registry import register_dataset

@register_dataset("DOD-O")
class DODO(BaseDataset):
    """DOD-O (Dreem Open Dataset - Obstructive) dataset."""
    
    def __init__(self):
        super().__init__("DOD-O","DOD-O - Dreem Open Dataset - Obstructive")

    def _setup_dataset_config(self):
        self.ann2label =  {
            0: 0,   # Wake
            1: 1,   # NREM Stage 1
            2: 2,   # NREM Stage 2
            3: 3,   # NREM Stage 3
            4: 4,   # REM sleep
            -1: 6   # Unscored
        }
        
        
        self.alias_mapping = {
            "O1_M1": ["signals/eeg/O1_M1"],
            "C4_M1": ["signals/eeg/C4_M1"],
            "F3_O1": ["signals/eeg/F3_O1"],
            "O1_M2": ["signals/eeg/O1_M2"],
            "F4_O2": ["signals/eeg/F4_O2"],
            "F3_F4": ["signals/eeg/F3_F4"],
            "C3_M2": ["signals/eeg/C3_M2"],
            "F3_M2": ["signals/eeg/F3_M2"],
            "O2_M1": ["signals/eeg/O2_M1"],
            "EOG1": ["signals/eog/EOG1"],
            "EOG2": ["signals/eog/EOG2"],
            "EMG": ["signals/emg/EMG"],
            "ECG": ["signals/emg/ECG"]
        }
        
        
        self.channel_names = [
            'signals/eeg/O1_M1', 'signals/eeg/C4_M1', 'signals/eeg/F3_O1', 'signals/eeg/O1_M2',
            'signals/eeg/F4_O2', 'signals/eeg/F3_F4', 'signals/eeg/C3_M2', 'signals/eeg/F3_M2',
            'signals/eeg/O2_M1',
            'signals/eog/EOG1', 'signals/eog/EOG2',
            'signals/emg/EMG',
            'signals/emg/ECG'
        ]
        
        
        self.channel_types = {
            'analog': [
                'signals/eeg/O2_M1', 'signals/emg/ECG', 'signals/emg/EMG', 'signals/eeg/C4_M1', 
                'signals/eeg/F3_O1', 'signals/eeg/O1_M2', 'signals/eog/EOG1', 'signals/eeg/F4_O2', 
                'signals/eeg/F3_F4', 'signals/eog/EOG2', 'signals/eeg/C3_M2', 'signals/eeg/F3_M2'
            ],
            'digital': []
        }
    
        
        self.channel_groups = {
            'eeg_eog': ['signals/eeg/O1_M1', 'signals/eeg/C4_M1', 'signals/eeg/F3_O1', 'signals/eeg/O1_M2', 'signals/eog/EOG1', 'signals/eeg/F4_O2', 'signals/eeg/F3_F4', 'signals/eog/EOG2', 'signals/eeg/C3_M2', 'signals/eeg/F3_M2', 'signals/eeg/O2_M1'],
            'emg': ['signals/emg/EMG'],
            'ecg': ['signals/emg/ECG']
        }
                
        self.file_extensions = {
            'psg_ext': '*.h5',
            'ann_ext': '*.h5'  # Annotations are embedded in data H5 files
        }
        
    
    def dataset_paths(self) -> Tuple[str, str]:
        """
        DOD-O dataset paths.
        Data and annotations are in the same H5 files.
        """
        data_dir = "DOD-O - Dreem Open Dataset - Obstructive"
        ann_dir = "DOD-O - Dreem Open Dataset - Obstructive"
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
    
    def ann_parse(self, ann_fname: str, epoch_duration: Optional[int] = None) -> Tuple[List[Dict], datetime]:
        """
        DOD-O doesn't use separate annotation files.
        Annotations are embedded in H5 files.
        """
        ann_stage_events = []

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