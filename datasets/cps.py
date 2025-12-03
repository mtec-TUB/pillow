import os
import wfdb
from typing import Dict, List, Tuple
from datetime import datetime, date
import numpy as np
import pandas as pd

from .base import BaseDataset
from .registry import register_dataset


@register_dataset("CPS")
class CPS(BaseDataset):
    """CPS (Comprehensive Polysomnography Dataset) dataset"""

    def __init__(self):
        super().__init__("CPS","CPS - Comprehensive Polysomnography Dataset (A Resource for Sleep-Related Arousal Research)", keep_folder_structure=False)

    def _setup_dataset_config(self):
        self.ann2label = {
            "Wach": 0,
            "N1": 1,
            "N2": 2,
            "N3": 3,
            "Rem": 4,
            "A": 6,
            "Artefakt": 6
        }
        
        
        self.channel_names = ['Akku', 'Licht', 'O2', 'EMG', 'C3:A2', 'A2', 'EOGr', 'C3', 'O2:A1', 'A1', 
                'C4:A1', 'EOGl:A2', 'C4', 'F4', 'F4:A1', 'EOGl:A1', 'EMG+', 
                'A1:O2', 'EOGr:A1', 'EOGl', 'EOGr:A2', 'ECG 2', 'EMG-',
                'PLMr', 'PLMl', 'Druck Snore', 'Pleth', 'RIP.Thrx', 'Summe RIPs', 'Pulse', 
                'Druck Flow', 'RIP.Abdom', 'Flow Th', 'Beweg.', 'Schnarc', 'SPO2', 'Pos.']
        
        
        self.channel_types = {
            'analog': ['Akku', 'Licht', 'O2', 'EMG', 'C3:A2', 'A2', 'EOGr', 'C3', 'O2:A1', 'A1', 
                       'C4:A1', 'EOGl:A2', 'C4', 'F4', 'F4:A1', 'EOGl:A1', 'EMG+', 
                       'A1:O2', 'EOGr:A1', 'EOGl', 'EOGr:A2', 'ECG 2', 'EMG-',
                       'PLMr', 'PLMl', 'Druck Snore', 'Pleth', 'RIP.Thrx', 'Summe RIPs', 'Pulse', 
                       'Druck Flow', 'RIP.Abdom', 'Flow Th', 'Beweg.', 'Schnarc', 'SPO2', 'Pos.'],
            'digital': []
        }

        
        
        self.channel_groups = {
            'eeg_eog': ['O2', 'C3:A2', 'A2', 'EOGr', 'C3', 'O2:A1', 'A1', 'C4:A1', 'EOGl:A2', 'C4', 'F4', 'F4:A1', 'EOGl:A1', 'A1:O2', 'EOGr:A1', 'EOGl', 'EOGr:A2'],
            'emg': ['EMG', 'EMG+', 'EMG-', 'PLMr', 'PLMl'],
            'ecg': ['ECG 2'],
            'thoraco_abdo_resp': ['RIP.Thrx', 'Flow Th', 'RIP.Abdom', 'Summe RIPs'],
            'nasal_pressure': ['Druck Snore'],
            'snoring': ['Schnarc']
        }
        
                
        self.file_extensions = {
            'psg_ext': '*/PSG/*.hea',
            'ann_ext': '*/PSG/Analysedaten/Schlafprofil.txt'
        }


    def dataset_paths(self) -> Tuple[str, str]:
        """Dataset paths for CPS dataset"""
        return (
            "CPS - Comprehensive Polysomnography Dataset (A Resource for Sleep-Related Arousal Research)/1.0.0/data",
            "CPS - Comprehensive Polysomnography Dataset (A Resource for Sleep-Related Arousal Research)/1.0.0/data"
        )
    
    def ann_parse(self, ann_fname: str, epoch_duration: int = 30) -> Tuple[List[Dict], datetime]:
        """
        Parse CPS annotation files.
        CPS uses semicolon-separated CSV files with German sleep stage names.
        
        Args:
            ann_fname: Path to annotation file (Schlafprofil.txt)
            epoch_duration: Duration of each epoch in seconds (default: 30)
            
        Returns:
            Tuple of (sleep_stage_events, start_datetime)
        """
        ann_stage_events = []
        ann_startdatetime = None
        
        data = pd.read_csv(ann_fname, sep=";", names=['Timestamp', 'Stage'], 
                            skipinitialspace=True, skiprows=6)
        

        for i, row in data.iterrows():
            stage = row['Stage']
            start = datetime.combine(date(1970,1,1),datetime.strptime(row['Timestamp'], '%H:%M:%S,%f').time())

            if ann_startdatetime == None and stage != "A":
                ann_startdatetime = start
            elif ann_startdatetime == None and stage == "A":
                continue

            ann_stage_events.append({
                'Stage': stage,
                'Start': int((start - ann_startdatetime).seconds),
                'Duration': epoch_duration
            })
        
        return ann_stage_events, ann_startdatetime
    
    def align_front(self, logger, ann_Startdatetime, psg_fname, ann_fname, signal, labels, fs):

        psg_fname, ext = os.path.splitext(psg_fname)
        record = wfdb.rdheader(psg_fname)
        psg_start_datetime = datetime.combine(record.base_date,record.base_time)

        print(f"Start time in signal file: {psg_start_datetime}")
        print(f"Start time in annot file: {ann_Startdatetime}")

        start_seconds= (ann_Startdatetime - psg_start_datetime).total_seconds()
      
        if start_seconds%(1/fs) != 0:
            raise Exception("Annotations start at timestamp outside of sample rate")

        if start_seconds < 0:
            if start_seconds < -30:
                raise Exception
            logger.info(f"Signal started {-start_seconds/60:.2f} min after label start, signal and label will be shortened ({30+int(start_seconds)}sec) at the front to match")
            signal = signal[30*fs+int(start_seconds*fs):]
            labels = labels[1:]
        elif start_seconds > 0:
            logger.info(f"Labeling started {start_seconds/60:.2f} min after signal start, signal will be shortened at the front to match")
            signal = signal[int(start_seconds*fs):]

        return True, signal,labels
    
    def align_end(self, logger, psg_fname: str, ann_fname: str, signals: np.ndarray,
                  labels: np.ndarray,
                  ) -> Tuple[np.ndarray, np.ndarray]:

        # if len(signals) > len(labels):
        #     logger.info(f"Signal (len: {len(signals)}) is shortend to match label length (len: {len(labels)})")
        #     signals = signals[:len(labels)]
        
        if len(labels) == len(signals) +1:
            logger.info(f"Labels (len: {len(labels)}) are shortend to match signal length ({len(signals)})")
            labels = labels[:len(signals)]
        
        assert len(signals) == len(labels), f"Length mismatch: signal={len(signals)}, labels={len(labels)} \n TODO: implement alignment function"
        
        return signals, labels
    