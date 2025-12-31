import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
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
            'psg_ext': '*.edf',
            'ann_ext': '*_sleepscoring.txt'
        }
    
    def dataset_paths(self) -> Tuple[str, str]:
        """
        HMC dataset paths.
        """
        data_dir = "HMC - Haaglanden Medisch Centrum sleep staging database/1.1/recordings"
        ann_dir = "HMC - Haaglanden Medisch Centrum sleep staging database/1.1/recordings"
        return data_dir, ann_dir
    
    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
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
            if row['Annotation'] not in ['Lights off','Lights on']:
                start = row['Recording onset']
                duration = row['Duration']
                stage = row['Annotation']
                ann_stage_events.append({'Stage': stage,
                                            'Start': start,
                                            'Duration': duration})

        return ann_stage_events, ann_Startdatetime
    
    def align_end(self, logger, psg_fname, ann_fname, signals, labels):

        # all signals are one epoch longer than the labels
        if len(signals) == len(labels) +1:
            signals = signals[:-1]

        assert len(signals) == len(labels), f"Length mismatch: signal={len(signals)}, labels={len(labels)} \n TODO: implement alignment function"

        return signals, labels
