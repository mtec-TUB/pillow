import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("CAP")
class CAP(BaseDataset):
    """CAPSLPDB - CAP Sleep Database dataset."""
    
    def __init__(self):
        super().__init__("CAP","CAPSLPDB - CAP Sleep Database")
  
    def _setup_dataset_config(self):
        self.ann2label = {
                            "W":0,
                            "S1":1,
                            "S2":2,
                            "S3":3,
                            "1S3":3,    # occurs in file n12.txt
                            "S4":3,
                            "REM":4,
                            "R":4,
                            "MT": 5
                            }
        
        self.intra_dataset_mapping = {
            "Abdomen": ['ADDDOME', 'ADDOME', 'Abdo','abdomen',],
            "Cannula": [ 'Canula', 'cannula',],
            "C3-A2": ['C3-A2','C3A2',],
            "C4-A1": ['C4-A1','C4A1',],
            "DX1-DX2": ['DX1-DX2','Dx1-DX2',],
            "Fp2-F4": ['FP2-F4','Fp2-F4',],
            "ECG": ['ECG','EKG','ekg',],
            "Chin1": ['CHIN1','EMG1'],
            "Chin2": ['CHIN2','EMG2'],
            "Flow": ['Flow','flow', ],
            "LOC": ['EOG-L', 'LOC','LOC-A1','EOG sin'],
            "O2-A1": ['O2-A1','O2A1',],
            "Pleth": ['PLETH', 'Pleth'],
            "Position": ['Posizione','Position',],
            "ROC": ['EOG-R','ROC','ROC-A2', 'EOG dx'],
            "SAO2": ["SpO2", "SAO2"],
            "Thorax": ['Torace','toracico', 'thorax','TORACE',],
            "Tib_sx": ['Tib sx','TIB Sx','tib sin'],
            "Tib_dx": ['TIB Dx','Tib dx','tib dx',],
            }
        
        self.inter_dataset_mapping = {
            "Abdomen": self.Mapping(self.TTRef.ABDOMINAL, None),
            "C3-A2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "C3": self.Mapping(self.TTRef.C3, None),
            "C4-A1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "C4": self.Mapping(self.TTRef.C4, None),
            "C3-P3": self.Mapping(self.TTRef.C3, self.TTRef.P3),
            'C4-P4': self.Mapping(self.TTRef.C4, self.TTRef.P4),
            "F1-F3": self.Mapping(self.TTRef.F1, self.TTRef.F3),
            "F2-F4": self.Mapping(self.TTRef.F2, self.TTRef.F4),
            "F3": self.Mapping(self.TTRef.F3, None),
            "F3-C3": self.Mapping(self.TTRef.F3, self.TTRef.C3),
            "F4": self.Mapping(self.TTRef.F4, None),
            "F3A2": self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            "F4A1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            "F4-C4": self.Mapping(self.TTRef.F4, self.TTRef.C4),
            "F7": self.Mapping(self.TTRef.F7, None),
            "F7-T3": self.Mapping(self.TTRef.F7, self.TTRef.T7),
            "F8": self.Mapping(self.TTRef.F8, None),
            "F8-T4": self.Mapping(self.TTRef.F8, self.TTRef.T8),
            "FP1": self.Mapping(self.TTRef.Fp1, None),
            "FP1-F3": self.Mapping(self.TTRef.Fp1, self.TTRef.F3),
            "Fp2-F4": self.Mapping(self.TTRef.Fp2, self.TTRef.F4),
            "Fp2": self.Mapping(self.TTRef.Fp2, None),
            "O1": self.Mapping(self.TTRef.O1, None),
            "O1A2": self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            "O2": self.Mapping(self.TTRef.O2, None),
            "P3": self.Mapping(self.TTRef.P3, None),
            "P3-O1": self.Mapping(self.TTRef.P3, self.TTRef.O1),
            "P4": self.Mapping(self.TTRef.P4, None),
            "P4-O2": self.Mapping(self.TTRef.P4, self.TTRef.O2),
            "ECG": self.Mapping(self.TTRef.ECG, None),
            "flow": self.Mapping(self.TTRef.AIRFLOW, None),
            "LOC": self.Mapping(self.TTRef.EL, self.TTRef.LPA),
            "O2-A1": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "T3": self.Mapping(self.TTRef.T7, None),
            "T3-T5": self.Mapping(self.TTRef.T7, self.TTRef.P7),
            "T4": self.Mapping(self.TTRef.T8, None),
            "T4-T6": self.Mapping(self.TTRef.T8, self.TTRef.P8),
            "T5": self.Mapping(self.TTRef.P7, None),
            "T6": self.Mapping(self.TTRef.P8, None),
            "Position": self.Mapping(self.TTRef.POSITION, None),
            "ROC": self.Mapping(self.TTRef.ER, self.TTRef.RPA),
            "SAO2": self.Mapping(self.TTRef.SPO2, None),
            "Thorax": self.Mapping(self.TTRef.THORACIC, None),
            "Tib_sx": self.Mapping(self.TTRef.EMG_LLEG, None),
            "Tib_dx": self.Mapping(self.TTRef.EMG_RLEG, None),
            "EMG1-EMG2": self.Mapping(self.TTRef.EMG_CHIN, None),   # is the chin EMG channel that appears most frequently
            "LOC-ROC": self.Mapping(self.TTRef.EL, self.TTRef.ER),
            "ROC-LOC": self.Mapping(self.TTRef.ER, self.TTRef.EL),

        }
        
        self.channel_names = ['A1', 'A2', 'ADDDOME', 'ADDOME', 'Abdo', 'C3', 'C3-A2', 'C3-P3', 'C3A2', 'C4', 'C4-A1',
                              'C4-P4', 'C4A1', 'CHIN1', 'CHIN2', 'Canula', 'DX1', 'DX1-DX2', 'DX2', 'Dx1-DX2', 'ECG',
                              'ECG1', 'ECG1-ECG2', 'ECG2', 'EKG', 'EMG', 'EMG-EMG', 'EMG1', 'EMG1-EMG2', 'EMG2', 'EOG dx',
                              'EOG sin', 'EOG-L', 'EOG-R', 'F1-F3', 'F2-F4', 'F3', 'F3-C3', 'F3A2', 'F4', 'F4-C4', 'F4A1',
                              'F7', 'F7-T3', 'F8', 'F8-T4', 'FP1', 'FP1-F3', 'FP2-F4', 'Flattening', 'Flow', 'Flusso',
                              'Fp2', 'Fp2-F4', 'HR', 'Heart Rate Varia', 'LOC', 'LOC-A1', 'LOC-ROC', 'MIC', 'O1', 'O1A2',
                              'O2', 'O2-A1', 'O2A1', 'Ox Status', 'P3', 'P3-O1', 'P4', 'P4-O2', 'PLETH', 'Pleth', 'Position',
                              'Posizione', 'ROC', 'ROC-A2', 'ROC-LOC', 'SAO2', 'STAT', 'SX1', 'SX1-SX2', 'SX2', 'Sound',
                              'SpO2', 'T3', 'T3-T5', 'T4', 'T4-T6', 'T5', 'T6', 'TAG', 'TERMISTORE', 'THE', 'TIB Dx', 'TIB Sx', 
                              'TORACE', 'Tib dx', 'Tib sx', 'Torace', 'abdomen', 'cannula', 'deltoide', 'ekg', 'flow', 'milo',
                              'thorax', 'tib dx', 'tib sin', 'toracico']
        
        
        self.channel_types = {'analog': ['T3-T5', 'Tib dx', 'EOG sin', 'F3', 'PLETH', 'Torace', 'abdomen', 'cannula', 
                                         'CHIN1', 'F3A2', 'EMG1-EMG2', 'Flusso', 'Fp2', 'thorax', 'P3-O1', 'EMG', 'Fp2-F4',
                                         'ROC-A2', 'DX1-DX2', 'C4', 'EMG1', 'F3-C3', 'Pleth', 'Heart Rate Varia', 'SX1-SX2',
                                         'C3-A2', 'FP2-F4', 'P3', 'ROC', 'Dx1-DX2', 'F1-F3', 'O2-A1', 'P4', 'EMG-EMG', 'Sound',
                                         'F4A1', 'DX1', 'C4A1', 'C3', 'C4-P4', 'F2-F4', 'ROC-LOC', 'LOC-ROC', 'C3-P3',
                                         'Flattening', 'Canula', 'F8-T4', 'DX2', 'TERMISTORE', 'EOG-L', 'SX2', 'LOC-A1', 'A2', 
                                         'TAG', 'ECG2', 'EKG', 'T4', 'THE', 'O2A1', 'F7', 'EOG dx', 'T4-T6', 'SX1', 'EMG2', 'T3',
                                         'ADDDOME', 'F4-C4', 'TORACE', 'Abdo', 'milo', 'ECG', 'deltoide', 'MIC', 'C4-A1', 'LOC', 
                                         'ekg', 'O2', 'FP1', 'Tib sx', 'Flow', 'TIB Sx', 'toracico', 'TIB Dx', 'ECG1', 'CHIN2',
                                         'tib sin', 'P4-O2', 'F8', 'ADDOME', 'A1', 'T6', 'ECG1-ECG2', 'O1', 'flow', 'F4', 'C3A2',
                                         'T5', 'tib dx', 'EOG-R', 'F7-T3', 'O1A2', 'FP1-F3'],
                              'digital': ['SpO2', 'STAT', 'SAO2', 'Ox Status', 'HR', 'Posizione', 'Position']}
        
        self.channel_groups = {
            'eeg_eog': ['A1', 'A2','C3', 'C3-A2', 'C3-P3', 'C3A2', 'C4', 'C4-A1','Fp2', 'Fp2-F4','LOC', 'LOC-A1', 'LOC-ROC',
                                  'C4-P4', 'C4A1','DX1', 'DX1-DX2', 'DX2', 'Dx1-DX2', 'EOG dx', 'EOG sin', 'EOG-L', 'EOG-R', 'F1-F3', 'F2-F4', 'F3', 'F3-C3', 'F3A2', 'F4', 'F4-C4', 'F4A1',
                                   'F7', 'F7-T3', 'F8', 'F8-T4', 'FP1', 'FP1-F3', 'FP2-F4','O1', 'O1A2', 'SX1', 'SX1-SX2', 'SX2',
                                  'O2', 'O2-A1', 'O2A1','P3', 'P3-O1', 'P4', 'P4-O2',  'ROC', 'ROC-A2', 'ROC-LOC', 'T3', 'T3-T5', 'T4', 'T4-T6', 'T5', 'T6', ],
            'emg': ['CHIN1', 'CHIN2', 'EMG', 'EMG-EMG', 'EMG1', 'EMG1-EMG2', 'EMG2','Tib dx', 'Tib sx','TIB Dx', 'TIB Sx', 'deltoide', 'tib dx', 'tib sin' ],
            'ecg': ['ECG', 'ECG1', 'ECG1-ECG2', 'ECG2', 'EKG','ekg',],
            'thoraco_abdo_resp':[ 'ADDDOME', 'ADDOME', 'Abdo','Flow','Torace', 'abdomen', 'toracico', 'thorax','TORACE','flow', 'TERMISTORE'],
            'snoring': ['MIC','Sound',]
        }
        
        
        self.file_extensions = {
            'psg_ext': '*.edf',
            'ann_ext': '*.txt'
        }
    
    def dataset_paths(self) -> Tuple[str, str]:
        return [
            "1.0.0",
            "1.0.0"
        ]
    
    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        Parse CAP txt annotation files.
        """
        ann_stage_events = []
        
        with open(ann_fname) as file:
            for i, line in enumerate(file):
                if line.startswith("Recording Date"):
                    date_str = line.split(':')[1].strip()
                    ann_start_datetime = datetime.strptime(date_str, "%d/%m/%Y")
                if line.startswith("Sleep Stage"):
                    start_idx = i
                    break
        
        df = pd.read_csv(ann_fname,header=start_idx,sep='\t',skip_blank_lines=False)

        start_time_label = None
        
        for i, row in df.iterrows():
            if not row.isna().any() and 'MCAP' not in row['Event']:
                label = row['Sleep Stage']
                try:
                    start = datetime.strptime(row['Time [hh:mm:ss]'],"%H.%M.%S")
                except ValueError:
                    start = datetime.strptime(row['Time [hh:mm:ss]'],"%H:%M:%S")
                try:
                    duration = row['Duration[s]']
                except KeyError:
                    duration = row['Duration [s]']
                
                if start_time_label == None:
                    start_time_label = start
                    
                start = int((start - start_time_label).seconds)
    
                ann_stage_events.append({'Stage': label,
                                            'Start': start,
                                            'Duration': duration})        #place holder
                
        for i, event in enumerate(ann_stage_events[:-1]):
            ann_stage_events[i]['Duration'] = ann_stage_events[i+1]['Start'] - event['Start']

        ann_start_datetime = datetime.combine(ann_start_datetime.date(), start_time_label.time())

        return ann_stage_events, ann_start_datetime, None, None
    
        
    def align_front(self, logger, alignment, pad_values, epoch_duration, delay_sec, signal: np.ndarray, labels, fs
                  ) -> Tuple[np.ndarray, np.ndarray]:

        return self.base_align_front(logger, delay_sec, alignment, pad_values, epoch_duration, signal, labels,fs)

    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(labels) > len(signals):
            return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)

        if len(signals) > len(labels):
            return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)        
    