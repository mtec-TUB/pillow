import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("HMC")
class HMC(BaseDataset):
    """HMC (Haaglanden Medisch Centrum sleep staging database) dataset."""
    
    def __init__(self):
        super().__init__("HMC","HMC - Haaglanden Medisch Centrum sleep staging database")

    def _setup_dataset_config(self):
        self.ann2label = {
            "Sleep stage W": 0,   # Wake
            "Sleep stage N1": 1,  # NREM Stage 1
            "Sleep stage N2": 2,  # NREM Stage 2
            "Sleep stage N3": 3,  # NREM Stage 3
            "Sleep stage R": 4,   # REM sleep
        }

        self.inter_dataset_mapping = {
            "EEG C4-M1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "EEG F4-M1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            "EEG O2-M1": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "EEG C3-M2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "EOG E1-M2": self.Mapping(self.TTRef.EL, self.TTRef.RPA),
            "EOG E2-M2": self.Mapping(self.TTRef.ER, self.TTRef.RPA),
            "EMG chin": self.Mapping(self.TTRef.EMG_CHIN, None),
            "ECG": self.Mapping(self.TTRef.ECG, None)
        }
        
        self.channel_names = [
            'EEG C4-M1', 'EEG F4-M1', 'EEG O2-M1', 'EEG C3-M2',
            'EOG E1-M2', 'EOG E2-M2',
            'EMG chin',
            'ECG'
        ]
        
        
        self.channel_types = {
            'analog': [
                'EOG E1-M2', 'EEG C4-M1', 'EEG F4-M1', 'EEG O2-M1', 'EOG E2-M2', 
                'EEG C3-M2', 'EMG chin', 'ECG'
            ],
            'digital': []
        }
        
        
        self.channel_groups = {
            'eeg_eog': ['EOG E1-M2', 'EEG C4-M1', 'EEG F4-M1', 'EEG O2-M1', 'EOG E2-M2', 'EEG C3-M2'],
            'emg': ['EMG chin'],
            'ecg': ['ECG']
        }
        
        
        self.file_extensions = {
            'psg_ext': '*[!sleepscoring].edf',
            'ann_ext': '*_sleepscoring.txt'
        }

    def get_file_identifier(self, psg_fname=None, ann_fname=None):
        psg_id, ann_id = None, None
        if psg_fname:
            psg_ext = ".edf"
            psg_id = psg_fname.split(psg_ext)[0]
        if ann_fname:
            ann_ext = "_sleepscoring.txt"
            ann_id = ann_fname.split(ann_ext)[0]
        return psg_id, ann_id
    
    def dataset_paths(self):
        return [
            os.path.join('1.1', 'recordings'),
            os.path.join('1.1', 'recordings')
        ]
    
    def ann_parse(self, ann_fname: str):
        """
        Parse HMC CSV annotation files.
        
        Args:
            ann_fname: Path to CSV annotation file
            
        Returns:
            Tuple of (sleep_stage_events, start_datetime)
        """
        annot = pd.read_csv(ann_fname,sep=',', header=0,skipinitialspace=True)

        ann_stage_events = []
        first_row = annot.iloc[0]
        ann_Startdatetime = datetime.strptime(first_row['Date'] + ' ' + first_row['Time'],'%d.%m.%y %H.%M.%S')
        
        for i, row in annot.iterrows():
            if row['Annotation'] == 'Lights off':
                lights_off = ann_Startdatetime + timedelta(seconds=row['Recording onset'])
            elif row['Annotation'] == 'Lights on':
                lights_on = ann_Startdatetime + timedelta(seconds=row['Recording onset'])
            else:
                start = row['Recording onset']
                duration = row['Duration']
                stage = row['Annotation']
                ann_stage_events.append({'Stage': stage,
                                            'Start': start,
                                            'Duration': duration})

        return ann_stage_events, ann_Startdatetime, lights_off, lights_on

    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):
        print("Aligning signals and labels for HMC dataset...")
        if len(signals) == len(labels) + 1:
            return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)
