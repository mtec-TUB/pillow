import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("UCDDB")
class UCDDB(BaseDataset):
    """UCDDB - St. Vincent's University Hospital, University College Dublin Sleep Apnea Database dataset with multiple scorers"""
    
    def __init__(self):
        super().__init__("UCDDB","UCDDB - St. Vincent's University Hospital, University College Dublin Sleep Apnea Database")
        
    def _setup_dataset_config(self):
        self.ann2label =  {
            0: 0,   # Wake
            2: 1,   # Stage 1
            3: 2,   # Stage 2
            4: 3,   # Stage 3
            5: 3,   # Stage 4 according to AASM
            1: 4,    # REM
            6: 6,
            7: 6,
            8: 6,
        }
        
        
        self.intra_dataset_mapping = {
            'Sound': ['Sound', 'Soud'],
        }

        self.inter_dataset_mapping = {
            'BodyPos': self.Mapping(self.TTRef.POSITION, None),
            'C3A2': self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            'C4A1': self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            'ECG': self.Mapping(self.TTRef.ECG, None),
            'EMG': self.Mapping(self.TTRef.EMG_CHIN, None),
            'Flow': self.Mapping(self.TTRef.AIRFLOW, None),
            'Left leg': self.Mapping(self.TTRef.EMG_LLEG, None),
            'Lefteye': self.Mapping(self.TTRef.EL, None),
            'Right leg': self.Mapping(self.TTRef.EMG_RLEG, None),
            'RightEye': self.Mapping(self.TTRef.ER, None),
            'Sound': self.Mapping(self.TTRef.SNORE, None),
            'SpO2': self.Mapping(self.TTRef.SPO2, None),
            'abdo': self.Mapping(self.TTRef.ABDOMINAL, None),
            'ribcage': self.Mapping(self.TTRef.THORACIC, None),
        }
        
        
        self.channel_names = ['BodyPos', 'C3A2', 'C4A1', 'ECG', 'EMG', 'Flow', 'Left leg', 'Lefteye', 'Pulse',
                               'Right leg', 'RightEye', 'Soud', 'Sound', 'SpO2', 'Sum', 'abdo', 'ribcage']
        
        
        self.channel_types =  {'analog': ['SpO2', 'RightEye', 'Sound', 'Lefteye', 'abdo', 'Pulse', 'ribcage', 
                                          'Sum', 'EMG', 'C4A1', 'ECG', 'Left leg', 'Soud', 'BodyPos', 'C3A2', 'Flow', 'Right leg'],
                               'digital': []}
        
        
        self.channel_groups = {
            'eeg_eog': ['C3A2', 'C4A1','RightEye', 'Lefteye',],
            'emg': ['EMG', 'Right leg', 'Left leg', ],
            'ecg': ['ECG'],
            'thoraco_abdo_resp': ['abdo', 'ribcage'],
        }
        
        self.file_extensions = {
            'psg_ext': '*.rec',
            'ann_ext': '*_stage.txt'
        }
    
    def dataset_paths(self) -> Tuple[str, str]:
        return [
            '',
            ''
        ]
    
    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """Parse ISRUC annotation files (multiple scorers in separate files)"""
        
        ann_stage_events = []

        epoch_duration = 30  # UCDDB uses 30-second epochs

        stages = np.loadtxt(ann_fname, dtype=int)
        for i, stage in enumerate(stages):
            ann_stage_events.append({
                'Start': i * epoch_duration,  # Assuming 30-second epochs
                'Duration': epoch_duration,
                'Stage': stage
            })
        
        return ann_stage_events, None, None, None

    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(signals) == len(labels) + 1:
            return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)        
    
        