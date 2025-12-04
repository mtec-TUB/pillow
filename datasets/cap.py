import pandas as pd
import numpy as np
import pyedflib
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from .base import BaseDataset
from .registry import register_dataset

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
                            "S4":3,
                            "REM":4,
                            "R":4,
                            "MT": 5
                            }
        
        self.alias_mapping = {
            "Abdomen": ['ADDDOME', 'ADDOME', 'Abdo','abdomen',],
            "Cannula": [ 'Canula', 'cannula',],
            "C3-A2": ['C3-A2','C3A2',],
            "C4-A1": ['C4-A1','C4A1',],
            "DX1-DX2": ['DX1-DX2','Dx1-DX2',],
            "ECG": ['ECG','EKG','ekg',],
            "Flow": ['Flow','flow', ],
            "LOC": ['EOG-L', 'LOC','LOC-A1',],
            "O2-A1": ['O2-A1','O2A1',],
            "Pleth": ['PLETH', 'Pleth'],
            "Position": ['Posizione','Position',],
            "ROC": ['EOG-R','ROC','ROC-A2',],
            "SAO2": ["SpO2", "SAO2"],
            "Thorax": ['Torace','toracico', 'thorax','TORACE',],
            "Tib_sx": ['Tib sx','TIB Sx',],
            "Tib_dx": ['TIB Dx','Tib dx','tib dx',],
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
            'thoraco_abdo_resp':[ 'ADDDOME', 'ADDOME', 'Abdo','Flow','Torace', 'abdomen', 'toracico', 'thorax','TORACE','flow', 'Canula', 'cannula','TERMISTORE', 'THE',  ],
            'snoring': ['MIC','Sound',]
        }
        
        
        self.file_extensions = {
            'psg_ext': '*.edf',
            'ann_ext': '*.txt'
        }
    
    def dataset_paths(self) -> Tuple[str, str]:
        """
        MIT-BIH dataset paths.
        """
        data_dir = "CAPSLPDB - CAP Sleep Database/1.0.0"
        ann_dir = "CAPSLPDB - CAP Sleep Database/1.0.0"
        return data_dir, ann_dir
    
    def ann_parse(self, ann_fname: str, epoch_duration: Optional[int] = None) -> Tuple[List[Dict], datetime]:
        """
        Parse MIT-BIH .st annotation files.
        """
        ann_stage_events = []
        print(ann_fname)
        
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
        return ann_stage_events, ann_start_datetime
    
        
    def align_front(self, logger, ann_Startdatetime, psg_fname:str, ann_fname: str, signal: np.ndarray, labels, fs
                  ) -> Tuple[np.ndarray, np.ndarray]:

        psg_f = pyedflib.EdfReader(psg_fname)
        psg_start_datetime = psg_f.getStartdatetime()

        start_seconds= (ann_Startdatetime - psg_start_datetime).total_seconds()

        if start_seconds > 0:
            logger.info(f"Labeling started {start_seconds/60:.2f} min after signal start, signal will be shortened at the front to match")
            signal = signal[int(start_seconds*fs):]

        return True, signal, labels

    def align_end(self, logger, psg_fname: str, ann_fname: str, signals: np.ndarray,
                  labels: np.ndarray,
                  ) -> Tuple[np.ndarray, np.ndarray]:

        if len(signals) > len(labels):
            logger.info(f"Signal (len: {len(signals)}) is shortend to match label length (len: {len(labels)})")
            signals = signals[:len(labels)]
        
        if len(labels) > len(signals):
            logger.info(f"Labels (len: {len(labels)}) are shortend to match signal length ({len(signals)})")
            labels = labels[:len(signals)]
        
        assert len(signals) == len(labels), f"Length mismatch: signal={len(signals)}, labels={len(labels)} \n TODO: implement alignment function"
        
        return signals, labels
    