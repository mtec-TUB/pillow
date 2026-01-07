import os
import wfdb
from typing import Dict, List, Tuple
from datetime import datetime, date
import numpy as np
import pandas as pd

from datasets.base import BaseDataset
from psg_processing.utils import Alignment
from datasets.registry import register_dataset


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

    def get_file_identifier(self, psg_fname, ann_fname):
        psg_id = psg_fname.split('/PSG/')[0]
        ann_id = ann_fname.split('/PSG/')[0]
        return psg_id, ann_id


    def dataset_paths(self) -> Tuple[str, str]:
        """Dataset paths for CPS dataset"""
        return (
            "CPS - Comprehensive Polysomnography Dataset (A Resource for Sleep-Related Arousal Research)/1.0.0/data",
            "CPS - Comprehensive Polysomnography Dataset (A Resource for Sleep-Related Arousal Research)/1.0.0/data"
        )
    
    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        Parse CPS annotation files.
        CPS uses semicolon-separated CSV files with German sleep stage names.
        
        Args:
            ann_fname: Path to annotation file (Schlafprofil.txt)
            
        Returns:
            Tuple of (sleep_stage_events, start_datetime)
        """
        ann_stage_events = []
        ann_startdatetime = None

        epoch_duration = 30  # CPS uses 30-second epochs
        
        data = pd.read_csv(ann_fname, sep=";", names=['Timestamp', 'Stage'], 
                            skipinitialspace=True, skiprows=6)
        

        for i, row in data.iterrows():
            stage = row['Stage']
            start = datetime.combine(date(1970,1,1),datetime.strptime(row['Timestamp'], '%H:%M:%S,%f').time())

            if ann_startdatetime == None:# and stage != "A":
                ann_startdatetime = start
            # elif ann_startdatetime == None and stage == "A":
            #     continue

            ann_stage_events.append({
                'Stage': stage,
                'Start': int((start - ann_startdatetime).seconds),
                'Duration': epoch_duration
            })

        
        return ann_stage_events, ann_startdatetime
    
    def align_front(self, logger, alignment, pad_values, epoch_duration, delay_samples, signal, labels, fs):

        if delay_samples < 0:
            if delay_samples < -30:
                raise Exception("Annotation without corresponding signal data is not supported")    # should not happen in CPS, just as safety check
            if alignment == Alignment.MATCH_SHORTER.value or alignment == Alignment.MATCH_SIGNAL.value:
                logger.info(f"Signal started {-delay_samples/60:.2f} min after label start, signal and label will be shortened ({30+int(delay_samples)}sec) at the front to match")
                signal = signal[30*fs+int(delay_samples*fs):]
                labels = labels[1:]
                return signal, labels
            elif alignment == Alignment.MATCH_LONGER.value or alignment == Alignment.MATCH_ANNOT.value:
                logger.info(f"Signal started {-delay_samples/60:.2f} min after label start, signal will be padded with constant value:{pad_values[0]} at the front to match")
                n_pad_samples = int(-delay_samples*fs)
                signal = np.hstack((np.full((n_pad_samples,), pad_values[0]), signal))
                return signal, labels
        elif delay_samples > 0:
            return self.base_align_front(logger, delay_samples, alignment, pad_values, epoch_duration, signal, labels,fs)

    
    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(labels) == len(signals) +1:
            return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)
