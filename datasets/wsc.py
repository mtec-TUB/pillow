import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("WSC")
class WSC(BaseDataset):
    """WSC (Wisconsin Sleep Cohort) dataset."""
    
    def __init__(self):
        super().__init__("WSC","WSC - Wisconsin Sleep Cohort")
    
    def _setup_dataset_config(self):
        self.ann2label = {
            # String-based labels
            "STAGE - W": 0,
            "STAGE - N1": 1,
            "STAGE - N2": 2,
            "STAGE - N3": 3,
            "STAGE - N4": 3,  # Follow AASM Manual
            "STAGE - R": 4,
            "STAGE - NO STAGE": 6,
            "STAGE - MVT": 6,
            # Numeric labels
            0: 0,  # Wake
            1: 1,  # NREM Stage 1
            2: 2,  # NREM Stage 2
            3: 3,  # NREM Stage 3
            4: 3,  # NREM Stage 4 (Follow AASM Manual)
            5: 4,  # REM
            6: 6,  # Unscored
            7: 6   # Movement
        }

        self.inter_dataset_mapping = {
            'thorax': self.Mapping(self.TTRef.THORACIC, None),
            'abdomen': self.Mapping(self.TTRef.ABDOMINAL, None),
            'C3_AVG': self.Mapping(self.TTRef.C3, None),
            'C4_AVG': self.Mapping(self.TTRef.C4, None),
            'nasalflow': self.Mapping(self.TTRef.AIRFLOW, None),
            'O1_M2': self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            'C4_M1': self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            'O1_M1': self.Mapping(self.TTRef.O1, self.TTRef.LPA),
            'F3_AVG': self.Mapping(self.TTRef.F3, None),
            'Pz_M2': self.Mapping(self.TTRef.Pz, self.TTRef.RPA),
            'ECG': self.Mapping(self.TTRef.ECG, None),
            'C4_M2': self.Mapping(self.TTRef.C4, self.TTRef.RPA),
            'Fz_AVG': self.Mapping(self.TTRef.Fz, None),
            'E1': self.Mapping(self.TTRef.EL, None),
            'Pz_Cz': self.Mapping(self.TTRef.Pz, self.TTRef.Cz),
            'F3_M1': self.Mapping(self.TTRef.F3, self.TTRef.LPA),
            'F3_M2': self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            'snore': self.Mapping(self.TTRef.SNORE, None),
            'C3_M1': self.Mapping(self.TTRef.C3, self.TTRef.LPA),
            'chin': self.Mapping(self.TTRef.EMG_CHIN, None),
            'cchin_l': self.Mapping(self.TTRef.EMG_LCHIN, None),
            'cchin_r': self.Mapping(self.TTRef.EMG_RCHIN, None),
            'Cz_M1': self.Mapping(self.TTRef.Cz, self.TTRef.LPA),
            'E2': self.Mapping(self.TTRef.ER, None),
            'F4_M2': self.Mapping(self.TTRef.F4, self.TTRef.RPA),
            'Pz_AVG': self.Mapping(self.TTRef.Pz, None),
            'Cz_M2': self.Mapping(self.TTRef.Cz, self.TTRef.RPA),
            'C3_M2': self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            'position': self.Mapping(self.TTRef.POSITION, None),
            'O2_M2': self.Mapping(self.TTRef.O2, self.TTRef.RPA),
            'O2_M1': self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            'spo2': self.Mapping(self.TTRef.SPO2, None),
            'Fz_M1': self.Mapping(self.TTRef.Fz, self.TTRef.LPA),
            'F4_M1': self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            'C4_AVG': self.Mapping(self.TTRef.C4, None),
            'F4_AVG': self.Mapping(self.TTRef.F4, None),
            'O1_AVG': self.Mapping(self.TTRef.O1, None),
            'Cz_AVG': self.Mapping(self.TTRef.Cz, None),
            'Fz_M2': self.Mapping(self.TTRef.Fz, self.TTRef.RPA),
            'lleg_r': self.Mapping(self.TTRef.EMG_LLEG, self.TTRef.EMG_RLEG),
        }
        
        
        self.channel_names = ['thorax', 'C3_AVG', 'flow', 'O1_M2', 'C4_M1', 'nas_pres', 
                'O1_M1', 'F3_AVG', 'cchin_r', 'Pz_M2', 'cchin_l', 'ECG',
                'C4_M2', 'Fz_AVG', 'E1', 'Pz_Cz', 'lleg1_2', 'pap_flow', 
                'abdomen', 'oralflow', 'F3_M1', 'rleg1_2', 'F3_M2', 'snore',
                'C3_M1', 'chin', 'Cz_M1', 'E2', 'F4_M2', 'Pz_AVG', 'Cz_M2', 
                'C3_M2', 'position', 'O2_M2', 'sum', 'O2_M1', 'pap_pres',
                'nasalflow', 'spo2', 'rchin_l', 'Fz_M1', 'F4_M1', 'C4_AVG',
                'F4_AVG', 'O1_AVG', 'lleg_r', 'Cz_AVG', 'Fz_M2']
        
        
        self.channel_types = {'analog': ['thorax', 'C3_AVG', 'flow', 'O1_M2', 'C4_M1', 'nas_pres', 'O1_M1', 'F3_AVG', 'cchin_r', 'Pz_M2', 'cchin_l', 'ECG', 'C4_M2', 
                           'Fz_AVG', 'E1', 'Pz_Cz', 'lleg1_2', 'pap_flow', 'abdomen', 'oralflow', 'F3_M1', 'rleg1_2', 'F3_M2', 'snore', 'C3_M1', 'chin', 
                           'Cz_M1', 'E2', 'F4_M2', 'Pz_AVG', 'Cz_M2', 'C3_M2', 'position', 'O2_M2', 'sum', 'O2_M1', 'pap_pres', 'nasalflow', 'spo2',
                           'rchin_l', 'Fz_M1', 'F4_M1', 'C4_AVG', 'F4_AVG', 'O1_AVG', 'lleg_r', 'Cz_AVG', 'Fz_M2'], 
                'digital': []}
        
        
        self.channel_groups = {
            'eeg_eog': ['Cz_AVG', 'Fz_M2', 'Fz_M1', 'F4_M1', 'C4_AVG', 'F4_AVG', 'O1_AVG', 'O2_M1', 'O2_M2', 'Cz_M1', 'E2', 'F4_M2', 'Pz_AVG', 'Cz_M2', 'C3_M2', 'C3_M1', 'Cz_M1', 'F3_M2', 'F3_M1', 'C3_AVG', 'O1_M2', 'C4_M1', 'O1_M1', 'F3_AVG', 'Pz_M2', 'C4_M2', 'Fz_AVG', 'E1', 'Pz_Cz'],
            'emg': ['lleg_r', 'cchin_r', 'cchin_l', 'lleg1_2', 'rleg1_2', 'chin', 'rchin_l'],
            'ecg': ['ECG'],
            'thoraco_abdo_resp': ['thorax', 'abdomen', 'oralflow', 'nasalflow'],
            'nasal_pressure': ['nas_pres'],
            'snoring': ['snore']
        }
        
        
        self.file_extensions = {
            'psg_ext': '*.edf',
            'ann_ext': '*.stg.txt',
            'ann_ext2': '*.allscore.txt'  # WSC has dual annotation file types
        }

    def get_file_identifier(self, psg_fname=None, ann_fname=None):
        psg_id, ann_id = None, None

        if psg_fname:
            psg_ext = self.file_extensions['psg_ext'].split('*')[-1]
            psg_id = psg_fname.split(psg_ext)[0]

        if ann_fname:
            ann_ext = self.file_extensions['ann_ext'].split('*')[-1]
            ann_ext2 = self.file_extensions['ann_ext2'].split('*')[-1]
            if ann_fname.endswith(ann_ext):
                ann_id = ann_fname.split(ann_ext)[0]
            else:
                ann_id = ann_fname.split(ann_ext2)[0]
        
        return psg_id, ann_id
    
    def dataset_paths(self) -> Tuple[str, str]:
        return [
            'polysomnography',
            'polysomnography'
        ]
    
    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        Parse WSC CSV annotation files.
        
        Args:
            ann_fname: Path to CSV annotation file
            
        Returns:
            Tuple of (sleep_stage_events, start_datetime)
        """
        ann_stage_events = []
        
        epoch_duration = 30  # WSC uses 30-second epochs

        if 'stg.txt' in ann_fname:
            ann_Startdatetime = None
            data = pd.read_csv(ann_fname, sep="\t", header=0, names=['Epoch','Stage', 'CAST_Stage'])
            
            for i,row in data.iterrows():
                ann_stage_events.append({'Stage': row['Stage'],
                                            'Start': i * epoch_duration,
                                            'Duration': epoch_duration})
            
        elif 'allscore.txt' in ann_fname:
            df = pd.read_csv(ann_fname, sep="\t", names=['Timestamp','Info'], encoding='latin',na_filter=False)
            
            start_idx = df[df['Info'] == 'START RECORDING'].index
            if len(start_idx) > 1:
                raise Exception
            elif len(start_idx) == 0:
                start_idx = df[df['Info'] == 'STAGE - NO STAGE'].index[0].astype(int)
            else:
                start_idx = start_idx[0].astype(int)
                
            ann_Startdatetime = datetime.strptime(df.iloc[start_idx]['Timestamp'],'%H:%M:%S.%f')
            
            df = df.iloc[start_idx:].reset_index()
            
            
            df = df[df['Info'].str.contains("STAGE")].reset_index()

            
            for i,row in df.iterrows():
                stage = row['Info']
                start = int((datetime.strptime(row['Timestamp'],'%H:%M:%S.%f') - ann_Startdatetime).seconds)
                if i == 0 and start != 0:
                    ann_stage_events.append({'Stage': 6,
                                                    'Start': 0,
                                                    'Duration': start})
                if i+1 != len(df):
                    duration = int((datetime.strptime(df.iloc[i+1]['Timestamp'],'%H:%M:%S.%f') - datetime.strptime(row['Timestamp'],'%H:%M:%S.%f')).seconds)
                else:
                    duration = epoch_duration
                
                ann_stage_events.append({'Stage': stage,
                                                'Start': start,
                                                'Duration': duration})

        return ann_stage_events, ann_Startdatetime

    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if ('allscore.txt' in ann_fname):
            if len(signals) > len(labels):
                return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels) 
            if len(labels) > len(signals):
                return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)

        if ('stg.txt' in ann_fname) and len(signals) == len(labels) + 1:
            return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)        
    
