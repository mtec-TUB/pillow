from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("CCSHS")
class CCSHS(BaseDataset):
    """CCSHS (CCSHS - Cleveland Children's Sleep and Health Study) dataset"""
    
    def __init__(self):
        super().__init__("CCSHS","CCSHS - Cleveland Children's Sleep and Health Study")
  
    def _setup_dataset_config(self):
        self.ann2label = {
            "Wake": 0,
            "Stage 1 sleep": 1,
            "Stage 2 sleep": 2,
            "Stage 3 sleep": 3,
            "Stage 4 sleep": 3, # according to AASM (only existing in one epoch in one single file)
            "REM sleep": 4,
            "Movement": 5,
        }

        self.intra_dataset_mapping = {
            "POSITION": ["position", "POSITION"],
        }

        self.inter_dataset_mapping = {
            "C3": self.Mapping(self.TTRef.C3, self.TTRef.Fpz),
            "A1": self.Mapping(self.TTRef.LPA, self.TTRef.Fpz),
            "LOC": self.Mapping(self.TTRef.EL, self.TTRef.Fpz),
            "A2": self.Mapping(self.TTRef.RPA, self.TTRef.Fpz),
            "C4": self.Mapping(self.TTRef.C4, self.TTRef.Fpz),
            "ROC": self.Mapping(self.TTRef.ER, self.TTRef.Fpz),
            "LEFT LEG1": self.Mapping(self.TTRef.EMG_LLEG, self.TTRef.Fpz),
            "RIGHT LEG1": self.Mapping(self.TTRef.EMG_RLEG, self.TTRef.Fpz),
            "EMG1": self.Mapping(self.TTRef.EMG_CHIN, self.TTRef.Fpz),
            "EMG2": self.Mapping(self.TTRef.EMG_LCHIN, self.TTRef.Fpz),
            "EMG3": self.Mapping(self.TTRef.EMG_RCHIN, self.TTRef.Fpz),
            "ECG1": self.Mapping(self.TTRef.ECG, self.TTRef.Fpz),
            "SNORE": self.Mapping(self.TTRef.SNORE, None),
            "POSITION": self.Mapping(self.TTRef.POSITION, None),
            "SpO2": self.Mapping(self.TTRef.SPO2, None),
            "AIRFLOW": self.Mapping(self.TTRef.AIRFLOW, None),
            "ABDO EFFORT": self.Mapping(self.TTRef.ABDOMINAL, None),
            "THOR EFFORT": self.Mapping(self.TTRef.THORACIC, None),
        }
        

        self.channel_names = ['C3', 'EMG3', 'OX STATUS', 'PTT', 'A1', 'SUM', 'EMG1', 'ECG2', 'LEFT LEG1', 'ABDO EFFORT', 'POSITION', 'THOR EFFORT', 'Light', 
                              'L Leg', 'DHR', 'LOC', 'A2', 'LEFT LEG2', 'AIRFLOW', 'SpO2', 'RIGHT LEG2', 'ECG1', 'C4', 'HRate', 'SNORE', 'RIGHT LEG1', 'ROC', 
                              'PlethWV', 'position', 'EMG2', 'NASAL PRES', 'PULSE', 'R Leg']
        
        
        self.channel_types = {'analog': ['PlethWV', 'RIGHT LEG1', 'C3', 'ECG2', 'C4', 'LEFT LEG1', 'ABDO EFFORT', 'SUM', 'R Leg', 'A2', 'ECG1', 'LEFT LEG2', 
                                         'RIGHT LEG2', 'SNORE', 'ROC', 'EMG3', 'EMG2', 'AIRFLOW', 'L Leg', 'THOR EFFORT', 'A1', 'LOC', 'NASAL PRES', 'EMG1'], 
                              'digital': ['PULSE', 'DHR', 'SpO2', 'OX STATUS', 'position', 'Light', 'HRate', 'PTT', 'POSITION']}
    
        
        
        self.channel_groups = {
            'eeg_eog': ['C3','A1','LOC', 'A2','C4','ROC', ],
            'emg': ['EMG3','EMG1','LEFT LEG1','L Leg','LEFT LEG2','RIGHT LEG2','RIGHT LEG1','EMG2','R Leg' ],
            'ecg': ['ECG2', 'ECG1',],
            'thoraco_abdo_resp': ['ABDO EFFORT','THOR EFFORT','AIRFLOW',],
            'nasal_pressure': ['NASAL PRES',],
            'snoring': ['SNORE',]
        }        
        
        self.file_extensions = {
            'psg_ext': '*.edf',
            'ann_ext': '*-nsrr.xml'
        }