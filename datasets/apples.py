import os
from typing import Dict, List, Tuple
from datetime import datetime
import pandas as pd

from .base import BaseDataset
from .registry import register_dataset


@register_dataset("APPLES")
class APPLES(BaseDataset):
    """APPLES (Apnea Positive Pressure Long-term Efficacy Study) dataset"""

    def __init__(self):
        super().__init__("APPLES","APPLES - Apnea Positive Pressure Long-term Efficacy Study")

    def _setup_dataset_config(self):
        self.ann2label ={'W': 0,
                        'N1': 1,
                        'N2': 2,
                        'N3': 3,
                        'R': 4,
                        '?': 6,
                        'L': 6,
                    }
    
    
        self.channel_names = ['EMG', 'ECG', 'pulse', 'snore', 'SpO2', 'thermistor', 'nasal_pres', 
                            'abdomen', 'thorax', 'LEG', 'O1_M2', 'C3_M2', 'C4_M1', 'O2_M1', 
                            'LOC', 'ROC']
    
        
        self.channel_types = {'analog': ['C4_M1', 'LOC', 'snore', 'nasal_pres', 'thermistor', 'ROC', 
                                          'O1_M2', 'O2_M1', 'C3_M2', 'abdomen', 'LEG', 'EMG', 'thorax', 'ECG'],
                              'digital': ['SpO2', 'pulse']}
                
        
        self.channel_groups = {'eeg_eog': ['O1_M2', 'C3_M2', 'C4_M1', 'O2_M1', 'LOC', 'ROC'],
                                'emg': ['EMG', 'LEG'],
                                'ecg': ['ECG'],
                                'thoraco_abdo_resp': ['abdomen', 'thorax'],
                                'nasal_pressure': ['nasal_pres'],
                                'snoring': ['snore']
                                }
        
        self.file_extensions = {'psg_ext': '*.edf',
                                'ann_ext': '*.annot'
                            }

    def dataset_paths(self):
        return [
            "APPLES - Apnea Positive Pressure Long-term Efficacy Study/polysomnography",
            "APPLES - Apnea Positive Pressure Long-term Efficacy Study/polysomnography"
        ]
    
    def ann_parse(self, ann_fname: str, epoch_duration: int = 30) -> Tuple[List[Dict], datetime]:
        """
        Parse APPLES annotation files.
        """
        ann_stage_events = []
        
        # Handle specific problematic file
        if os.path.basename(ann_fname) == 'apples-170408.annot':
            return ann_stage_events, None
        
        ann_df = pd.read_csv(ann_fname,header = 0, sep='\t')
            
        ann_startdatetime = None
        
        for i, row in ann_df.iterrows():
            event = row['class']
            if event in self.ann2label:
                start = datetime.strptime(row['start'], '%H:%M:%S')
                if ann_startdatetime == None:
                    ann_startdatetime = start
                end = datetime.strptime(row['stop'], '%H:%M:%S')
                
                duration = int((end - start).seconds)
                start_sec = int((start - ann_startdatetime).seconds)
                
                ann_stage_events.append({
                    'Stage': event,
                    'Start': start_sec,
                    'Duration': duration
                })
        
        return ann_stage_events, ann_startdatetime
