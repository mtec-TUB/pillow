import os
import wfdb
from typing import Any, Dict, List, Optional, Tuple
from datetime import _Time, datetime, date
import numpy as np
import pandas as pd

from datasets.base import BaseDataset
from psg_processing.utils import Alignment
from datasets.registry import register_dataset

from datasets.file_handlers import WFDBHandler


@register_dataset("CPS")
class CPS(BaseDataset):
    """CPS (Comprehensive Polysomnography Dataset) dataset"""

    def __init__(self):
        super().__init__("CPS","CPS - Comprehensive Polysomnography Dataset (A Resource for Sleep-Related Arousal Research)", keep_folder_structure=False)

        self._file_handler = WFDBHandler()

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

        # https://physionet.org/content/cps-dataset-sleep/1.0.0/
        self.inter_dataset_mapping = {
            "O2": self.Mapping(self.TTRef.O2, None),
            "C3:A2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "A2": self.Mapping(self.TTRef.RPA, None),
            "EOGr": self.Mapping(self.TTRef.ER, None),
            "C3": self.Mapping(self.TTRef.C3, None),
            "O2:A1": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "A1": self.Mapping(self.TTRef.LPA, None),
            "C4:A1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "EOGl:A2": self.Mapping(self.TTRef.EL, self.TTRef.RPA),
            "C4": self.Mapping(self.TTRef.C4, None),
            "F4": self.Mapping(self.TTRef.F4, None),
            "F4:A1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            "EOGl:A1": self.Mapping(self.TTRef.EL, self.TTRef.LPA),
            "EMG": self.Mapping(self.TTRef.EMG_CHIN, None),
            "A1:O2": self.Mapping(self.TTRef.LPA, self.TTRef.O2),
            "EOGr:A1": self.Mapping(self.TTRef.ER, self.TTRef.LPA),
            "EOGl": self.Mapping(self.TTRef.EL, None),
            "EOGr:A2": self.Mapping(self.TTRef.ER, self.TTRef.RPA),
            "ECG 2": self.Mapping(self.TTRef.ECG,None),
            "PLMr": self.Mapping(self.TTRef.EMG_RLEG, None),
            "PLMl": self.Mapping(self.TTRef.EMG_LLEG, None),
            "Schnarc": self.Mapping(self.TTRef.SNORE, None),
            "RIP.Thrx": self.Mapping(self.TTRef.THORACIC, None),
            "RIP.Abdom": self.Mapping(self.TTRef.ABDOMINAL, None),
            "Flow Th": self.Mapping(self.TTRef.AIRFLOW, None),
            "SPO2": self.Mapping(self.TTRef.SPO2, None),
            "Pos.": self.Mapping(self.TTRef.POSITION, None),
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

    def get_file_identifier(self, psg_fname=None, ann_fname=None):
        psg_id, ann_id = None, None
        if psg_fname:
            psg_id = psg_fname.split('/PSG/')[0]
        if ann_fname:
            ann_id = ann_fname.split('/PSG/')[0]
        return psg_id, ann_id


    def dataset_paths(self):
        """Dataset paths for CPS dataset"""
        return [
            os.path.join("1.0.0", "data"),
            os.path.join("1.0.0", "data")
        ]
    
    def ann_parse(self, ann_fname: str):
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
            # use 1970 instead of actual date because edf stores 1970-01-01 for all files (easier comparison)
            start = datetime.combine(date(1970,1,1),datetime.strptime(row['Timestamp'], '%H:%M:%S,%f').time())

            if ann_startdatetime == None:
                ann_startdatetime = start

            ann_stage_events.append({
                'Stage': stage,
                'Start': int((start - ann_startdatetime).seconds),
                'Duration': epoch_duration
            })

        marker_file = ann_fname.replace("Schlafprofil.txt", "Marker.txt")
        lights_off, lights_on = None, None
        if os.path.exists(marker_file):
            marker_df = pd.read_csv(marker_file, sep=";", names=['Timestamp', 'Event'], skipinitialspace=True,skip_blank_lines=True)
            lights_off = marker_df.loc[marker_df['Event'] == "Licht aus", "Timestamp"]
            if len(lights_off) >= 1:
                # take first of available lights off markers (in some recorings, two markers occur, reason unknown)
                lights_off = datetime.strptime(lights_off.iloc[0], '%H:%M:%S,%f').time()
            
            lights_on = marker_df.loc[marker_df['Event'] == "Licht an", "Timestamp"]
            if len(lights_on) == 1:
                lights_on = datetime.strptime(lights_on.iloc[0], '%H:%M:%S,%f').time()
            else:
                lights_on = None
        else:
            raise Exception(f"Marker file not found: {marker_file}")    # should not occur
        
        return ann_stage_events, ann_startdatetime, lights_off, lights_on
    
    def align_front(self, logger, alignment, pad_values, epoch_duration, delay_sec, signal, labels, fs):

        return self.base_align_front(logger, delay_sec, alignment, pad_values, epoch_duration, signal, labels,fs)

    
    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(labels) == len(signals) +1:
            return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)
