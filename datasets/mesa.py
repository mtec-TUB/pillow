from .base import BaseDataset
from .registry import register_dataset


@register_dataset("MESA")
class MESA(BaseDataset):
    """MESA (Multi-Ethnic Study of Atherosclerosis) dataset"""
    
    def __init__(self):
        super().__init__("MESA","MESA - Multi-Ethnic Study of Atherosclerosis")
  
    def _setup_dataset_config(self):
        self.ann2label = {
            "Wake": 0,
            "Stage 1 sleep": 1,
            "Stage 2 sleep": 2,
            "Stage 3 sleep": 3,
            "Stage 4 sleep": 3, # Follow AASM Manual
            "REM sleep": 4,
            "Unscored": 6,
            # "Movement": 5,
            # '0': 0,  # Wake
            # '1': 1,  # Stage 1
            # '2': 2,  # Stage 2  
            # '3': 3,  # Stage 3
            # '4': 3,  # Stage 4 -> Stage 3 (AASM)
            # '5': 4,  # REM
            # '9': 6   # Unscored
        }
        

        self.channel_names = ['Thor', 'DHR', 'Snore', 'EKG', 'HR', 'EEG2_Off', 'EMG', 'EEG1', 'Pos', 'Therm',
                'Leg', 'EEG2', 'EOG-R', 'Flow', 'Pleth', 'EEG3_Off', 'Aux_AC', 'SpO2', 'EOG-R_Off','Pres',
                'EOG-L_Off', 'EMG_Off', 'EKG_Off', 'EEG3', 'Abdo', 'PTT', 'OxStatus', 'EEG1_Off', 'EOG-L']
        
        
        self.channel_types = {'analog': ['Thor', 'DHR', 'Snore', 'EKG', 'EMG', 'EEG1', 'Therm', 'Leg', 'EEG2', 'EOG-R', 'Flow', 'Pleth', 'Aux_AC', 'EEG3', 'Abdo', 'PTT', 
                           'EOG-L', 'Pres'], 
                'digital': ['EEG2_Off', 'Pos', 'SpO2', 'EEG3_Off', 'EOG-R_Off', 'EOG-L_Off', 'EMG_Off', 'EKG_Off', 'EEG1_Off', 'OxStatus', 'HR']}
    
        
        
        self.channel_groups = {
            'eeg_eog': ['EEG1', 'EEG2', 'EEG3', 'EOG-R', 'EOG-L'],
            'emg': ['EMG', 'Leg'],
            'ecg': ['EKG'],
            'thoraco_abdo_resp': ['Thor', 'Abdo'],
            'snoring': ['Snore']
        }
        
        
        self.file_extensions = {
            'psg_ext': '*.edf',
            'ann_ext': '*.xml'
        }
