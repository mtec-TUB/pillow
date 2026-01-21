from typing import Dict, List, Union

from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("SOF")
class SOF(BaseDataset):
    """SOF (Study of Osteoporotic Fractures) dataset"""
    
    def __init__(self):
        super().__init__("SOF","SOF - Study of Osteoporotic Fractures")
   
    def _setup_dataset_config(self):
        self.ann2label = {
            "Wake": 0,
            "Stage 1 sleep": 1,
            "Stage 2 sleep": 2,
            "Stage 3 sleep": 3,
            "Stage 4 sleep": 3,  # Follow AASM Manual
            "REM sleep": 4,
            "Unscored": 6
        }
        
        
        self.intra_dataset_mapping = {
            'nasal_pressure': ['nasal_pressure', 'NASAL_PRESSURE'],
            'lchin': ['L Chin','EMG/L'],
            'rchin': ['R Chin','EMG/R'],
        }
        
        
        self.channel_names = ['ROC', 'C3', 'LOC', 'A1', 'O1', 'O2', 'A2', 'C4',
                'L Chin', 'Leg/L', 'R Chin', 'EMG/R', 'EMG/L', 'Leg/R',
                'ECG1', 'ECG2', 'Thoracic', 'Abdominal', 'Airflow',
                'NASAL PRESSURE', 'Nasal Pressure', 'Cannula Flow',
                'SAO2', 'HR', 'CH30', 'Position', 'STAT', 'DHR']
        
        
        self.channel_types ={'analog': ['NASAL PRESSURE', 'A2', 'DHR', 'HR', 'Leg/L', 'CH30', 'C4', 'L Chin', 'EMG/L', 'Leg/R', 'O2', 'EMG/R', 'ECG2', 
                                        'R Chin', 'Cannula Flow', 'ECG1', 'ROC', 'Airflow', 'O1', 'C3', 'A1', 'Nasal Pressure', 'Abdominal', 'LOC', 
                                        'Thoracic'], 
                             'digital': ['SAO2', 'STAT', 'Position']}
        
        
        self.channel_groups = {
            'eeg_eog': ['ROC', 'C3', 'LOC', 'A1', 'O1', 'O2', 'A2', 'C4'],
            'emg': ['L Chin', 'Leg/L', 'R Chin', 'EMG/R', 'EMG/L', 'Leg/R'],
            'ecg': ['ECG1', 'ECG2'],
            'thoraco_abdo_resp': ['Thoracic', 'Abdominal', 'Airflow'],
            'nasal_pressure': ['NASAL PRESSURE', 'Nasal Pressure']
        }

        self.inter_dataset_mapping = {
            "C3": self.Mapping(self.TTRef.C3, self.TTRef.Fpz),
            "C4": self.Mapping(self.TTRef.C4, self.TTRef.Fpz),
            "A1": self.Mapping(self.TTRef.LPA, self.TTRef.Fpz),
            "A2": self.Mapping(self.TTRef.RPA, self.TTRef.Fpz),
            "O1": self.Mapping(self.TTRef.O1, self.TTRef.Fpz),
            "O2": self.Mapping(self.TTRef.O2, self.TTRef.Fpz),
            "ROC": self.Mapping(self.TTRef.ER, self.TTRef.Fpz),
            "LOC": self.Mapping(self.TTRef.EL, self.TTRef.Fpz),
            "SAO2": self.Mapping(self.TTRef.SPO2, None),
            "lchin": self.Mapping(self.TTRef.EMG_CHIN, self.TTRef.Fpz),
            "ECG1": self.Mapping(self.TTRef.ECG, self.TTRef.Fpz),
            "Position": self.Mapping(self.TTRef.POSITION, None),
            "Abdominal": self.Mapping(self.TTRef.ABDOMINAL, None),
            "Thoracic": self.Mapping(self.TTRef.THORACIC, None),
            "Airflow": self.Mapping(self.TTRef.AIRFLOW, None),
            "Leg/L": self.Mapping(self.TTRef.EMG_LLEG, None),
            "Leg/R": self.Mapping(self.TTRef.EMG_RLEG, None),
        }
        
        
        self.file_extensions = {
            'psg_ext': '*.edf',
            'ann_ext': '*-nsrr.xml'
        }