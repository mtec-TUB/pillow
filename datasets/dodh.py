import os
import h5py
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from datasets.base import BaseDataset
from datasets.registry import register_dataset
from datasets.file_handlers import DOD_H5Handler

@register_dataset("DOD-H")
class DODH(BaseDataset):
    """DOD-H (Dreem Open Dataset - Healthy) dataset."""
    
    def __init__(self):
        super().__init__("DOD-H","DOD-H - Dreem Open Dataset - Healthy")

        self._file_handler = DOD_H5Handler()

    def _setup_dataset_config(self):
        self.ann2label = {
            0: "W",   # Wake
            1: "N1",   # NREM Stage 1
            2: "N2",   # NREM Stage 2
            3: "N3",   # NREM Stage 3
            4: "REM",   # REM sleep
            -1: "UNK",   # Unscored
        }
        
        
        self.intra_dataset_mapping = {
            "F4_O2": ["eeg/F4_O2"],
            "FP2_F4": ["eeg/FP2_F4"],
            "F3_M2": ["eeg/F3_M2"],
            "F3_O1": ["eeg/F3_O1"],
            "F4_M1": ["eeg/F4_M1"],
            "C3_M2": ["eeg/C3_M2"],
            "FP1_M2": ["eeg/FP1_M2"],
            "F3_F4": ["eeg/F3_F4"],
            "FP2_O2": ["eeg/FP2_O2"],
            "FP1_O1": ["eeg/FP1_O1"],
            "FP2_M1": ["eeg/FP2_M1"],
            "FP1_F3": ["eeg/FP1_F3"],
            "EOG1": ["eog/EOG1"],
            "EOG2": ["eog/EOG2"],
            "EMG": ["emg/EMG"],
            "ECG": ["emg/ECG"]
        }

        self.inter_dataset_mapping= {
            "C3_M2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "F4_M1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            "FP2_F4": self.Mapping(self.TTRef.Fp2, self.TTRef.F4),
            "FP1_M2": self.Mapping(self.TTRef.Fp1, self.TTRef.RPA),
            "F3_F4": self.Mapping(self.TTRef.F3, self.TTRef.F4),
            "FP2_O2": self.Mapping(self.TTRef.Fp2, self.TTRef.O2),
            "FP1_O1": self.Mapping(self.TTRef.Fp1, self.TTRef.O1),
            "FP2_M1": self.Mapping(self.TTRef.Fp2, self.TTRef.LPA),
            "FP1_F3": self.Mapping(self.TTRef.Fp1, self.TTRef.F3),
            "F3_M2": self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            "F3_O1": self.Mapping(self.TTRef.F3, self.TTRef.O1),
            "F4_O2": self.Mapping(self.TTRef.F4, self.TTRef.O2),
            "EOG1": self.Mapping(self.TTRef.EL, None),
            "EOG2": self.Mapping(self.TTRef.ER, None),
            "ECG": self.Mapping(self.TTRef.ECG, None),
            "EMG": self.Mapping(self.TTRef.EMG_CHIN, None)
        }
        
        
        self.channel_names = [
            'eeg/F4_O2', 'eeg/FP2_F4', 'eeg/F3_M2', 'eeg/F3_O1',
            'eeg/F4_M1', 'eeg/C3_M2', 'eeg/FP1_M2', 'eeg/F3_F4',
            'eeg/FP2_O2', 'eeg/FP1_O1', 'eeg/FP2_M1', 'eeg/FP1_F3',
            'eog/EOG1', 'eog/EOG2',
            'emg/EMG',
            'emg/ECG'
        ]
        
        
        self.channel_types = {
            'analog': [
                'eeg/F4_O2', 'eeg/FP2_F4', 'emg/EMG', 'eeg/F3_M2',
                'emg/ECG', 'eeg/F3_O1', 'eeg/F4_M1', 'eeg/C3_M2',
                'eeg/FP1_M2', 'eeg/F3_F4', 'eeg/FP2_O2', 'eeg/FP1_O1',
                'eeg/FP2_M1', 'eog/EOG2', 'eeg/FP1_F3', 'eog/EOG1'
            ],
            'digital': []
        }
        
        
        
        self.channel_groups = {
            'eeg_eog': ['eeg/F4_O2', 'eeg/FP2_F4', 'eeg/F3_M2', 'eeg/F3_O1', 'eeg/F4_M1', 'eeg/C3_M2', 'eeg/FP1_M2', 'eeg/F3_F4', 'eeg/FP2_O2', 'eeg/FP1_O1', 'eeg/FP2_M1', 'eeg/FP1_F3', 'eog/EOG2', 'eog/EOG1'],
            'emg': ['emg/EMG'],
            'ecg': ['emg/ECG']
        }
        
        self.file_extensions = {
            'psg_ext': '*.h5',
            'ann_ext': '*.h5'  # Annotations are embedded in data H5 files
        }
    
    def dataset_paths(self):
        return [
            '',
            ''
        ]
    
    def ann_parse(self, ann_fname: str):
        """
        DOD-H doesn't use separate annotation files.
        Annotations are embedded in H5 files.
        """
        ann_stage_events = []
        epoch_duration = 30  # DOD-H uses 30-second epochs

        with h5py.File(ann_fname, "r") as f:
            
            stages = f["hypnogram"][:]
            for i, stage in enumerate(stages):
                onset = i * epoch_duration
                duration = epoch_duration
                ann = {
                    'Stage': stage,
                    'Start': onset,
                    'Duration': duration
                }
                ann_stage_events.append(ann)

        return ann_stage_events, None, None, None
    