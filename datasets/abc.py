from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("ABC")
class ABC(BaseDataset):
    """ABC (Apnea, Bariatric surgery, and CPAP study) dataset"""
    
    def __init__(self):
        super().__init__("ABC","ABC - Apnea, Bariatric surgery, and CPAP study")
    

    def _setup_dataset_config(self):
        self.ann2label = {"Wake": 0,
                        "Stage 1 sleep": 1,
                        "Stage 2 sleep": 2,
                        "Stage 3 sleep": 3,
                        "REM sleep": 4,
                        "Unscored": 6
                        }
        
        
        self.channel_names = ['O1',  'O2', 'E1',  'C4',  'M1',  'C3', 'M2',  'E2', 'F4', 'F3','Manual Pos','Sum', 'RLeg2','Snore', 'Derived HR',
                            'Respiratory Rate', 'Pulse', 'Chin3', 'ECG2', 'Thor', 'CPAP Flow', 'Ox Status', 'Pleth', 'LLeg1', 'PosSensor',
                            'Chin1', 'LLeg2', 'Chin2', 'Airflow', 'Abdo', 'RLeg1', 'CPAP Press', 'Nasal Pressure', 'SpO2','Light', 'ECG1']
        
        
        self.channel_types = {'analog': ['O1',  'O2', 'E1',  'C4',  'M1',  'C3', 'M2',  'E2', 'F4', 'F3','Sum', 'RLeg2','Snore', 'Derived HR',
                            'Respiratory Rate', 'Pulse', 'Chin3', 'ECG2', 'Thor', 'CPAP Flow', 'Pleth', 'LLeg1', 
                            'Chin1', 'LLeg2', 'Chin2', 'Airflow', 'Abdo', 'RLeg1', 'CPAP Press', 'Nasal Pressure', 'ECG1'],
                            'digital': ['SpO2','PosSensor','Manual Pos','Ox Status','Light']}
        
        
        self.channel_groups = {'eeg_eog': ['O1', 'O2', 'C4', 'M1', 'C3', 'M2', 'F4', 'F3', 'E1', 'E2'],
                                'emg': ['RLeg2', 'Chin3', 'RLeg1', 'Chin1', 'LLeg2', 'Chin2', 'LLeg1'],
                                'ecg': ['ECG2', 'ECG1'],
                                'thoraco_abdo_resp': ['Abdo', 'Airflow', 'Thor'],
                                'nasal_pressure': ['Nasal Pressure'],
                                'snoring': ['Snore']
                                }
                
        
        self.file_extensions = {'psg_ext': '**/*.edf',
                                'ann_ext': '**/*-nsrr.xml'}
    