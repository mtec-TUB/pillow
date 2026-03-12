"""
MSP - Maternal Sleep in Pregnancy and the Fetus
"""
import os
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("MSP")
class MSP(BaseDataset):
    """MSP (Maternal Sleep in Pregnancy and the Fetus) dataset."""
    
    def __init__(self):
        super().__init__("MSP","MSP - Maternal Sleep in Pregnancy and the Fetus")
    
    def _setup_dataset_config(self):
        self.ann2label = {
                'W': 0,
                'N1': 1,
                'N2': 2,
                'N3': 3,
                'R': 4,
                '?': 6,
        }

        self.inter_dataset_mapping = {
            "C3_M2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "C4_M1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "F3_M2": self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            "F4_M1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            "O1_M2": self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            "O2_M1": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "LOC": self.Mapping(self.TTRef.EL, None),
            "ROC": self.Mapping(self.TTRef.ER, None),
            "RAT": self.Mapping(self.TTRef.EMG_RLEG, None),
            "LAT": self.Mapping(self.TTRef.EMG_LLEG, None),
            "abdomen": self.Mapping(self.TTRef.ABDOMINAL, None),
            "thorax": self.Mapping(self.TTRef.THORACIC, None),
            "SpO2": self.Mapping(self.TTRef.SPO2, None),
            "ECG": self.Mapping(self.TTRef.ECG, None),
            "EMG1": self.Mapping(self.TTRef.EMG_LCHIN, None),   # not sure if this is correct (left/right)
            "EMG2": self.Mapping(self.TTRef.EMG_RCHIN, None),
        }
        
        self.channel_names =  ['RAT', 'LAT', 'EMG2', 'EMG1', 'ECG', 'abdomen', 'thorax', 'nasal_pres',
                               'pulse', 'F4_M1', 'ROC', 'O2_M1', 'HR', 'thermistor', 'O1_M2', 'C4_M1', 
                               'SpO2', 'LOC', 'fetal_HR', 'F3_M2', 'C3_M2'
                               ]
        
        
        self.channel_types = {'analog': ['O1_M2', 'nasal_pres', 'LOC', 'RAT', 'ECG', 'ROC', 'fetal_HR', 'C3_M2', 
                                        'HR', 'C4_M1', 'abdomen', 'LAT', 'thermistor', 'thorax', 'O2_M1', 'F4_M1', 
                                        'EMG1', 'EMG2', 'F3_M2'
                                        ], 
                              'digital': ['SpO2', 'pulse']
                              }
        
        
        self.channel_groups = {'eeg_eog': ['F4_M1','ROC','O2_M1','O1_M2', 'C4_M1','LOC','F3_M2', 'C3_M2'],
                                'emg': ['RAT', 'LAT', 'EMG2', 'EMG1'],
                                'ecg': ['ECG'],
                                'thoraco_abdo_resp': ['abdomen', 'thorax'],
                                'nasal_pressure': ['nasal_pres']
                                }
        
        
        self.file_extensions = {
                                'psg_ext': '*.edf',
                                'ann_ext': '*.annot'
                                }
        

    def dataset_paths(self) -> Tuple[str, str]:
        return [
            'polysomnography',
            'polysomnography'
        ]
        
    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        function to parse the annotation file of the dataset into sleep stage events with start and duration

        """

        ann_stage_events = []
        ann_df = pd.read_csv(ann_fname,header = 0, sep='\t')
        ann_Startdatetime = None

        ann_stage_events = []
        for i,row in ann_df.iterrows():
            event = row['class']
            if event not in ['biocal','lights_off','lights_on','arousal','hypopnea','desat','apnea','snoring','PLM']:
                # if event not in self.ann2label:
                #     print(event)
                #     raise Exception
                start = datetime.strptime(row['start'],'%H:%M:%S') 
                if ann_Startdatetime == None:
                    ann_Startdatetime = start
                end =  datetime.strptime(row['stop'],'%H:%M:%S')
                duration = int((end - start).seconds)
                start = int((start - ann_Startdatetime).seconds)
                ann_stage_events.append({'Stage': event,
                                        'Start': start,
                                        'Duration': duration})

        return ann_stage_events, ann_Startdatetime
        
    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(labels) > len(signals):
            return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)    
    