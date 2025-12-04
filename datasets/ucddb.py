import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

from .base import BaseDataset
from .registry import register_dataset


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
        
        
        self.alias_mapping = {
            'Sound': ['Sound', 'Soud'],
        }
        
        
        self.channel_names = ['BodyPos', 'C3A2', 'C4A1', 'ECG', 'EMG', 'Flow', 'Left leg', 'Lefteye', 'Pulse',
                               'Right leg', 'RightEye', 'Soud', 'Sound', 'SpO2', 'Sum', 'abdo', 'ribcage']
        
        
        self.channel_types =  {'analog': ['SpO2', 'RightEye', 'Sound', 'Lefteye', 'abdo', 'Pulse', 'ribcage', 
                                          'Sum', 'EMG', 'C4A1', 'ECG', 'Left leg', 'Soud', 'BodyPos', 'C3A2', 'Flow', 'Right leg'],
                               'digital': []}
        
        
        self.channel_groups = {
            'eeg_eog': ['C3A2', 'C4A1','RightEye',  'Lefteye',],
            'emg': ['EMG', 'Right leg', 'Left leg', ],
            'ecg': ['ECG'],
            'thoraco_abdo_resp': ['abdo', 'ribcage','Flow'],
        }
        
        self.file_extensions = {
            'psg_ext': '*.rec',
            'ann_ext': '*_stage.txt'
        }
    
    def dataset_paths(self) -> Tuple[str, str]:
        """
        ISRUC dataset paths.
        """
        data_dir = "UCDDB - St. Vincent's University Hospital, University College Dublin Sleep Apnea Database"
        ann_dir = "UCDDB - St. Vincent's University Hospital, University College Dublin Sleep Apnea Database"
        return data_dir, ann_dir
    
    def ann_parse(self, ann_fname: str, epoch_duration = None) -> Tuple[List[Dict], datetime]:
        """Parse ISRUC annotation files (multiple scorers in separate files)"""
        
        ann_stage_events = []

        stages = np.loadtxt(ann_fname, dtype=int)
        for i, stage in enumerate(stages):
            ann_stage_events.append({
                'Start': i * epoch_duration,  # Assuming 30-second epochs
                'Duration': epoch_duration,
                'Stage': stage
            })
        
        return ann_stage_events, None
    
    def align_end(self, logger, psg_fname: str, ann_fname:str, signals: np.ndarray,
                  labels: np.ndarray,
                  ) -> Tuple[np.ndarray, np.ndarray]:
        
        if len(signals) == len(labels) + 1:
            logger.info(f"Signal (len: {len(signals)}) is shortend to match label length (len: {len(labels)})")
            signals = signals[:len(labels)]
        
        assert len(signals) == len(labels), f"Length mismatch: signal={len(signals)}, labels={len(labels)} \n TODO: implement alignment function"
        
        return signals, labels
        