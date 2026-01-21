from typing import Dict, List

from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("CFS")
class CFS(BaseDataset):
    """CFS (Cleveland Family Study) dataset"""

    def __init__(self):
        super().__init__("CFS","CFS - Cleveland Family Study")

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
            "SpO2": ["SpO2", "SaO2"],
        }
        
        
        self.channel_names = ['L Leg', 'PAP FLOW', 'SNORE', 'PlethWV', 'POSITION', 'Masimo', 'NASAL PRES', 'SUM', 'ABDO EFFORT', 'PULSE',
                'THOR EFFORT', 'SpO2', 'SaO2', 'OX STATUS', 'HRate', 'Light', 'R Leg', 'AIRFLOW',
                'M2', 'C4', 'C3', 'EMG3', 'ECG2', 'ROC', 'LOC', 'ECG1', 'EMG1', 'EMG2', 'M1']
        
        
        self.channel_types = {
            'analog': ['L Leg', 'M2', 'PAP FLOW', 'C4', 'SNORE', 'C3', 'EMG3', 'PlethWV', 'Masimo', 'ECG2', 'NASAL PRES', 'ROC', 'SUM', 'ABDO EFFORT',
                       'LOC', 'ECG1', 'PULSE', 'THOR EFFORT', 'EMG1', 'SaO2', 'HRate', 'EMG2', 'M1', 'R Leg', 'AIRFLOW'],
            'digital': ['POSITION', 'SpO2', 'OX STATUS', 'Light']
        }
        
        
        self.channel_groups = {
            'eeg_eog': ['M2', 'C4', 'C3', 'M1', 'ROC', 'LOC'],
            'emg': ['L Leg', 'EMG3', 'R Leg', 'EMG2', 'EMG1'],
            'ecg': ['ECG2', 'ECG1'],
            'thoraco_abdo_resp': ['ABDO EFFORT', 'THOR EFFORT', 'AIRFLOW'],
            'nasal_pressure': ['NASAL PRES'],
            'snoring': ['SNORE']
        }

        self.inter_dataset_mapping = {
            "C3": self.Mapping(self.TTRef.C3, self.TTRef.Fpz),
            "C4": self.Mapping(self.TTRef.C4, self.TTRef.Fpz),
            "M1": self.Mapping(self.TTRef.LPA, self.TTRef.Fpz),
            "M2": self.Mapping(self.TTRef.RPA, self.TTRef.Fpz),
            "LOC": self.Mapping(self.TTRef.EL, self.TTRef.Fpz),
            "ROC": self.Mapping(self.TTRef.ER, self.TTRef.Fpz),
            "AIRFLOW": self.Mapping(self.TTRef.AIRFLOW, None),
            "ABDO EFFORT": self.Mapping(self.TTRef.ABDOMINAL, None),
            "SpO2": self.Mapping(self.TTRef.SPO2, None),
            "POSITION": self.Mapping(self.TTRef.POSITION, None),
            "ECG1": self.Mapping(self.TTRef.ECG, self.TTRef.Fpz),
            "L Leg": self.Mapping(self.TTRef.EMG_LLEG, self.TTRef.Fpz),
            "R Leg": self.Mapping(self.TTRef.EMG_RLEG, self.TTRef.Fpz),
            "EMG1": self.Mapping(self.TTRef.EMG_CHIN, self.TTRef.Fpz),
            "SNORE": self.Mapping(self.TTRef.SNORE, None),
        }
                
        
        self.file_extensions = {
            'psg_ext': '*.edf',
            'ann_ext': '*-nsrr.xml'
        }
