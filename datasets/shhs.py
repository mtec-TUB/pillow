"""
SHHS - Sleep Heart Health Study
"""

from .base import BaseDataset
from .registry import register_dataset


@register_dataset("SHHS")
class SHHS(BaseDataset):
    """SHHS - Sleep Heart Health Study"""

    def __init__(self):
        super().__init__("SHHS","SHHS - Sleep Heart Health Study")

    def _setup_dataset_config(self):
        self.ann2label = {
            "Wake": 0,
            "Stage 1 sleep": 1,
            "Stage 2 sleep": 2,
            "Stage 3 sleep": 3, #    
            "Stage 4 sleep": 3, # Follow AASM Manual
            "REM sleep": 4,
            "Unscored": 6,
            "Movement": 5
        }
        
        
        self.alias_mapping = {'NewAirflow':['new air','NEW AIR','New AIR','New Air','NEWAIR','New A/F'],
                            'OxStatus': ['OX STAT','OX stat',],
                            'EEG_sec': ['EEG(sec)', 'EEG(SEC)', 'EEG sec', ],
                            'EEG2': ['EEG2','EEG 2',]
                            }
        
        
        self.channel_names = ['EEG', 'EEG(sec)', 'EMG', 'ABDO RES', 'new air', 'CPAP', 'POSITION', 'EOG(L)', 'H.R.', 'LEG(L)', 'EEG2', 'LIGHT', 'epms', 'NEW AIR', 'New AIR', 'SOUND', 'THOR RES', 'SaO2', 'OX STAT', 'New Air', 'NASAL', 'New A/F', 'NEWAIR', 'PR', 'EEG 2', 'ECG', 'EEG(SEC)', 'EEG sec', 'OX stat', 'AIRFLOW', 'EPMS', 'LEG(R)', 'EOG(R)', 'AUX']
        
        
        self.channel_types = {'analog': ['EMG', 'EEG sec', 'AIRFLOW', 'EOG(R)', 'ECG', 'EEG', 'NEW AIR', 'ABDO RES', 'New A/F', 'SOUND', 'PR', 'H.R.',
                               'AUX', 'EEG 2',  'NEWAIR', 'New Air', 'EEG2', 'new air', 'EOG(L)', 'New AIR', 'EEG(sec)', 'THOR RES', 'EEG(SEC)'],
                    'digital': ['LEG(L)', 'OX STAT', 'epms', 'EPMS','NASAL', 'LEG(R)', 'OX stat', 'SaO2', 'CPAP', 'POSITION', 'LIGHT']}

        
        self.channel_groups = {
            'eeg_eog': ['EEG','EEG2', 'EEG 2','EOG(L)','EOG(R)', 'EEG(SEC)', 'EEG sec'],
            'emg': ['EMG'],
            'ecg': ['ECG'],
            'thoraco_abdo_resp': ['ABDO RES', 'THOR RES','AIRFLOW','new air','NEW AIR','New AIR','New Air','NEWAIR','New A/F'],
            'nasal_pressure': [],
            'snoring': ['SOUND']
        }
        
        
        self.file_extensions = {
            'psg_ext': '**/*.edf',
            'ann_ext': '**/*-nsrr.xml'
        }
