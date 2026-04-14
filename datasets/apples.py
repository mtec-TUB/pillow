import os
from typing import Dict, List, Optional, Tuple
from datetime import _Time, datetime
import pandas as pd

from datasets.base import BaseDataset
from datasets.registry import register_dataset


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
                        'L': 6, # treat epochs with L as unknown
                    }
        
        self.inter_dataset_mapping = {
            'EMG': self.Mapping(self.TTRef.EMG_CHIN, None),
            'ECG': self.Mapping(self.TTRef.ECG, None),
            'snore': self.Mapping(self.TTRef.SNORE, None),
            'SpO2': self.Mapping(self.TTRef.SPO2, None),
            'abdomen': self.Mapping(self.TTRef.ABDOMINAL, None),
            'thorax': self.Mapping(self.TTRef.THORACIC, None),
            'LEG': self.Mapping(self.TTRef.EMG_LLEG, None),
            'O1_M2': self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            'C3_M2': self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            'C4_M1': self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            'O2_M1': self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            'LOC': self.Mapping(self.TTRef.EL, None),
            'ROC': self.Mapping(self.TTRef.ER, None),
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
                                'thoraco_abdo_resp': ['abdomen', 'thorax','thermistor'],
                                'nasal_pressure': ['nasal_pres'],
                                'snoring': ['snore']
                                }
        
        self.file_extensions = {'psg_ext': '*.edf',
                                'ann_ext': '*.annot'
                            }

    def dataset_paths(self):
        return [
            "polysomnography",
            "polysomnography"
        ]
    
    def ann_parse(self, ann_fname: str)-> tuple[list, datetime, _Time, _Time]:
        """
        Parse APPLES annotation files.
        """
        ann_stage_events = []
        
        # Handle specific problematic file
        # if os.path.basename(ann_fname) == 'apples-170408.annot':
        #     return ann_stage_events, None
        
        ann_df = pd.read_csv(ann_fname,header = 0, sep='\t')
        ann_df = ann_df[ann_df['class'].isin(self.ann2label.keys())].reset_index(drop=True)

        lights_off = datetime.strptime(ann_df.loc[ann_df['class'] != 'L','start'].values[0], '%H:%M:%S').time() # first non-L event is lights off
        lights_on = datetime.strptime(ann_df.loc[ann_df['class'] != 'L','stop'].values[-1], '%H:%M:%S').time()  # last non-L event is lights on
            
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

        ann_startdatetime = datetime.combine(datetime(1985,1,1), ann_startdatetime.time())
        
        return ann_stage_events, ann_startdatetime, lights_off, lights_on