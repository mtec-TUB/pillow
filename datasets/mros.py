from typing import Dict, List, Tuple, Union
from datasets.base import BaseDataset
from datasets.registry import register_dataset

import numpy as np

@register_dataset("MROS")
class MROS(BaseDataset):
    """MROS (MrOS Sleep Study) dataset."""
    
    def __init__(self):
        super().__init__("MROS","MROS - MrOS Sleep Study")
  
    def _setup_dataset_config(self):
        self.ann2label = {
            "Wake": 0,
            "Stage 1 sleep": 1,
            "Stage 2 sleep": 2,
            "Stage 3 sleep": 3,
            "Stage 4 sleep": 3, # Follow AASM Manual
            "REM sleep": 4,
            "Unscored": 6
        }
        
        
        self.intra_dataset_mapping = {
                'ECGR': ['ECG R','ECGR'],
                'ECGL': ['ECGL','ECG L'],
                'RChin': ['R Chin','RChin'],
                'LChin': [ 'LChin','L Chin'],
                'Chest': ['Thoracic'],
                'LLeg': ['LegL','Leg L'],
                'RLeg': [ 'Leg R', 'LegR',],
                'Nasal_press': ['Cannula Flow', 'CannulaFlow','CH37'],
                'Abdominal': ['Abdominal','ABD'],
                'SpO2': ['SpO2','SaO2'],
                'E1': ['E1','LOC'],
                'E2': ['E2','ROC'],
                }
        
        
        self.channel_names = ['STAT', 'M1', 'Chest', 'DHR', 'CH37', 'ECG R', 'R Chin', 'Leg R', 'LegL', 'M2', 'ECGL', 'C4', 'Airflow', 'CH36', 'ECG L', 'HR', 'Thoracic',
                'Abdominal', 'SaO2', 'ECGR', 'L Chin', 'SpO2', 'LChin', 'LegR', 'SUM', 'ROC', 'Position', 'RChin', 'Cannula Flow', 'E1', 'CannulaFlow', 'LOC',
                'C3', 'Leg L', 'E2', 'ABD','ECG L-ECG R','A2','A1','L Chin-R Chin','C3-A2','C4-A1']
        
        
        self.channel_types = {'analog': ['M1', 'Chest', 'DHR', 'CH37', 'ECG R', 'R Chin', 'Leg R', 'LegL', 'M2', 'ECGL', 'C4', 'Airflow', 'CH36', 'ECG L', 'Thoracic',
                           'Abdominal', 'ECGR', 'L Chin', 'LChin', 'LegR', 'SUM', 'ROC', 'RChin', 'Cannula Flow', 'E1', 'CannulaFlow', 'LOC', 'C3', 'Leg L', 
                           'E2', 'ABD', 'HR','ECG L-ECG R','A2','A1','L Chin-R Chin','C3-A2','C4-A1'], 
                'digital': ['STAT', 'SpO2', 'SaO2', 'Position']}
        
        
        self.channel_groups = {
            'eeg_eog': ['E1','LOC','E2','ROC', 'M1', 'M2',  'C4','C3'],
            'emg': ['R Chin', 'RChin', 'LChin', 'L Chin', 'LegL', 'Leg L', 'Leg R', 'LegR'],
            'ecg': ['ECG R', 'ECGR', 'ECGL', 'ECG L'],
            'thoraco_abdo_resp': ['Abdominal', 'ABD', 'Chest', 'Thoracic'],
            'nasal_pressure': ['Cannula Flow', 'CannulaFlow', 'CH37']
        }

        self.inter_dataset_mapping = {
            "C3": self.Mapping(self.TTRef.C3, self.TTRef.Fpz),
            "C4": self.Mapping(self.TTRef.C4, self.TTRef.Fpz),
            "A1": self.Mapping(self.TTRef.LPA, self.TTRef.Fpz),
            "A2": self.Mapping(self.TTRef.RPA, self.TTRef.Fpz),
            "ROC": self.Mapping(self.TTRef.ER, self.TTRef.Fpz),
            "LOC": self.Mapping(self.TTRef.EL, self.TTRef.Fpz)
        }
        
        self.file_extensions = {
            'psg_ext': '**/*.edf',
            'ann_ext': '**/*-nsrr.xml'
        }

    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if ann_fname == "mros-visit1-aa2931-nsrr.xml" and len(signals) > len(labels):
            return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)        
    
