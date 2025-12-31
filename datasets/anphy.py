import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("ANPHY")
class ANPHY(BaseDataset):
    """ANPHY dataset"""
    
    def __init__(self):
        super().__init__("ANPHY","ANPHY", keep_folder_structure=False)

    def _setup_dataset_config(self):
        self.ann2label = {
            "W": 0,   # Wake
            "N1": 1,  # NREM Stage 1
            "N2": 2,  # NREM Stage 2
            "N3": 3,  # NREM Stage 3
            "R": 4,   # REM sleep
            "L": 6    # Prepare, lights on/off time
        }
        
        self.channel_names = ['AF3', 'AF3-Ref', 'AF4', 'AF4-Ref', 'AF7', 'AF7-Ref', 'AF8', 'AF8-Ref', 'AFZ', 'AFZ-Ref', 'C1', 'C1-', 'C1-Ref', 'C2', 
                              'C2-Ref', 'C3', 'C3-Ref', 'C4', 'C4-Ref', 'C5', 'C5-Ref', 'C6', 'C6-Ref', 'CP1', 'CP1-Ref', 'CP2', 'CP2-Ref', 'CP3', 
                              'CP3-Ref', 'CP4', 'CP4-Ref', 'CP5', 'CP6', 'CP6-Ref', 'CPZ', 'CPZ-Ref', 'CZ', 'CZ-Ref', 'ChEMG1', 'ChEMG2', 'ECG1', 
                              'ECG2', 'EOG1', 'EOG2', 'F1', 'F1-Ref', 'F10', 'F10-Ref', 'F11', 'F12', 'F2', 'F2-Ref', 'F3', 'F3-Ref', 'F4', 'F4-', 
                              'F4-Ref', 'F5', 'F5-Ref', 'F6', 'F6-Ref', 'F7', 'F7-Ref', 'F8', 'F8-Ref', 'F9', 'F9-Ref', 'FC1', 'FC1-Ref', 'FC2', 
                              'FC2-Ref', 'FC3', 'FC3-Ref', 'FC4', 'FC4-Ref', 'FC5', 'FC5-Ref', 'FC6', 'FC6-Ref', 'FCZ', 'FCZ-Ref', 'FPZ', 'FPZ-Ref', 
                              'FT10', 'FT10-Ref', 'FT11', 'FT12', 'FT7', 'FT7-Ref', 'FT8', 'FT8-Ref', 'FT9', 'FT9-Ref', 'FZ', 'FZ-Ref', 'Fp1', 
                              'Fp1-Ref', 'Fp2', 'Fp2-Ref', 'LLEG+', 'LLEG-', 'O1', 'O1-Ref', 'O2', 'O2-Ref', 'OZ', 'OZ-Ref', 'P1', 'P1-Ref', 'P10', 
                              'P10-Ref', 'P11', 'P12', 'P2', 'P2-Ref', 'P3', 'P3-Ref', 'P4', 'P4-Ref', 'P5', 'P5-Ref', 'P6', 'P6-Ref', 'P9', 'P9-Ref', 
                              'PO3', 'PO3-Ref', 'PO4', 'PO4-Ref', 'PO7', 'PO7-Ref', 'PO8', 'PO8-Ref', 'POZ', 'POZ-Ref', 'PZ', 'PZ-Ref', 'RLEG+', 
                              'RLEG-', 'SO1', 'SO1-Ref', 'SO2', 'SO2-Ref', 'T10', 'T10-Ref', 'T3', 'T3-Ref', 'T4', 'T4-Ref', 'T5', 'T5-Ref', 'T6', 
                              'T6-Ref', 'T9', 'T9-Ref', 'TP10', 'TP10-Ref', 'TP11', 'TP12', 'TP7', 'TP7-Ref', 'TP8', 'TP8-Ref', 'TP9', 'TP9-Ref', 
                              'ZY1', 'ZY2']        
        
        self.channel_types = {'analog':['AF3', 'AF3-Ref', 'AF4', 'AF4-Ref', 'AF7', 'AF7-Ref', 'AF8', 'AF8-Ref', 'AFZ', 'AFZ-Ref', 'C1', 'C1-', 'C1-Ref', 'C2', 
                              'C2-Ref', 'C3', 'C3-Ref', 'C4', 'C4-Ref', 'C5', 'C5-Ref', 'C6', 'C6-Ref', 'CP1', 'CP1-Ref', 'CP2', 'CP2-Ref', 'CP3', 
                              'CP3-Ref', 'CP4', 'CP4-Ref', 'CP5', 'CP6', 'CP6-Ref', 'CPZ', 'CPZ-Ref', 'CZ', 'CZ-Ref', 'ChEMG1', 'ChEMG2', 'ECG1', 
                              'ECG2', 'EOG1', 'EOG2', 'F1', 'F1-Ref', 'F10', 'F10-Ref', 'F11', 'F12', 'F2', 'F2-Ref', 'F3', 'F3-Ref', 'F4', 'F4-', 
                              'F4-Ref', 'F5', 'F5-Ref', 'F6', 'F6-Ref', 'F7', 'F7-Ref', 'F8', 'F8-Ref', 'F9', 'F9-Ref', 'FC1', 'FC1-Ref', 'FC2', 
                              'FC2-Ref', 'FC3', 'FC3-Ref', 'FC4', 'FC4-Ref', 'FC5', 'FC5-Ref', 'FC6', 'FC6-Ref', 'FCZ', 'FCZ-Ref', 'FPZ', 'FPZ-Ref', 
                              'FT10', 'FT10-Ref', 'FT11', 'FT12', 'FT7', 'FT7-Ref', 'FT8', 'FT8-Ref', 'FT9', 'FT9-Ref', 'FZ', 'FZ-Ref', 'Fp1', 
                              'Fp1-Ref', 'Fp2', 'Fp2-Ref', 'LLEG+', 'LLEG-', 'O1', 'O1-Ref', 'O2', 'O2-Ref', 'OZ', 'OZ-Ref', 'P1', 'P1-Ref', 'P10', 
                              'P10-Ref', 'P11', 'P12', 'P2', 'P2-Ref', 'P3', 'P3-Ref', 'P4', 'P4-Ref', 'P5', 'P5-Ref', 'P6', 'P6-Ref', 'P9', 'P9-Ref', 
                              'PO3', 'PO3-Ref', 'PO4', 'PO4-Ref', 'PO7', 'PO7-Ref', 'PO8', 'PO8-Ref', 'POZ', 'POZ-Ref', 'PZ', 'PZ-Ref', 'RLEG+', 
                              'RLEG-', 'SO1', 'SO1-Ref', 'SO2', 'SO2-Ref', 'T10', 'T10-Ref', 'T3', 'T3-Ref', 'T4', 'T4-Ref', 'T5', 'T5-Ref', 'T6', 
                              'T6-Ref', 'T9', 'T9-Ref', 'TP10', 'TP10-Ref', 'TP11', 'TP12', 'TP7', 'TP7-Ref', 'TP8', 'TP8-Ref', 'TP9', 'TP9-Ref', 
                              'ZY1', 'ZY2'],
                              'digital': []
        }
        
        self.channel_groups = {
            'eeg_eog': ['AF3', 'AF3-Ref', 'AF4', 'AF4-Ref', 'AF7', 'AF7-Ref', 'AF8', 'AF8-Ref', 'AFZ', 'AFZ-Ref', 'C1', 'C1-', 'C1-Ref', 'C2', 
                              'C2-Ref', 'C3', 'C3-Ref', 'C4', 'C4-Ref', 'C5', 'C5-Ref', 'C6', 'C6-Ref', 'CP1', 'CP1-Ref', 'CP2', 'CP2-Ref', 'CP3', 
                              'CP3-Ref', 'CP4', 'CP4-Ref', 'CP5', 'CP6', 'CP6-Ref', 'CPZ', 'CPZ-Ref', 'CZ', 'CZ-Ref',  'EOG1', 'EOG2', 'F1', 'F1-Ref', 
                              'F10', 'F10-Ref', 'F11', 'F12', 'F2', 'F2-Ref', 'F3', 'F3-Ref', 'F4', 'F4-', 
                              'F4-Ref', 'F5', 'F5-Ref', 'F6', 'F6-Ref', 'F7', 'F7-Ref', 'F8', 'F8-Ref', 'F9', 'F9-Ref', 'FC1', 'FC1-Ref', 'FC2', 
                              'FC2-Ref', 'FC3', 'FC3-Ref', 'FC4', 'FC4-Ref', 'FC5', 'FC5-Ref', 'FC6', 'FC6-Ref', 'FCZ', 'FCZ-Ref', 'FPZ', 'FPZ-Ref', 
                              'FT10', 'FT10-Ref', 'FT11', 'FT12', 'FT7', 'FT7-Ref', 'FT8', 'FT8-Ref', 'FT9', 'FT9-Ref', 'FZ', 'FZ-Ref', 'Fp1', 
                              'Fp1-Ref', 'Fp2', 'Fp2-Ref',  'O1', 'O1-Ref', 'O2', 'O2-Ref', 'OZ', 'OZ-Ref', 'P1', 'P1-Ref', 'P10', 
                              'P10-Ref', 'P11', 'P12', 'P2', 'P2-Ref', 'P3', 'P3-Ref', 'P4', 'P4-Ref', 'P5', 'P5-Ref', 'P6', 'P6-Ref', 'P9', 'P9-Ref', 
                              'PO3', 'PO3-Ref', 'PO4', 'PO4-Ref', 'PO7', 'PO7-Ref', 'PO8', 'PO8-Ref', 'POZ', 'POZ-Ref', 'PZ', 'PZ-Ref', 
                               'SO1', 'SO1-Ref', 'T10', 'T10-Ref', 'T3', 'T3-Ref', 'T4', 'T4-Ref', 'T5', 'T5-Ref', 'T6', 
                              'T6-Ref', 'T9', 'T9-Ref', 'TP10', 'TP10-Ref', 'TP11', 'TP12', 'TP7', 'TP7-Ref', 'TP8', 'TP8-Ref', 'TP9', 'TP9-Ref', 
                              'ZY1', 'ZY2'],
            'emg': ['ChEMG1', 'ChEMG2','LLEG+', 'LLEG-','RLEG+','RLEG-'],
            'ecg': ['ECG1', 'ECG2']
        }
        
        
        self.file_extensions = {
            'psg_ext': '**/*.edf',
            'ann_ext': '**/*.txt'
        }

    def get_file_identifier(self, psg_fname, ann_fname):
        psg_id = Path(psg_fname).parent
        ann_id = Path(ann_fname).parent
        return psg_id, ann_id
        
    
    def dataset_paths(self) -> Tuple[str, str]:
        """
        HMC dataset paths.
        """
        data_dir = "ANPHY/osfstorage-archive"
        ann_dir = "ANPHY/osfstorage-archive"
        return data_dir, ann_dir
    
    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        Parse ANPHY annotation files.
        """
        annot = pd.read_csv(ann_fname,sep='\t', names=['Event', 'Start Time', 'Duration'])

        ann_stage_events = []
        start_time_label = None

        for i, row in annot.iterrows():
            start = row['Start Time']

            if start_time_label == None:
                start_time_label = start

            duration = row['Duration']
            stage = row['Event']
            ann_stage_events.append({'Stage': stage,
                                        'Start': start - start_time_label,
                                        'Duration': duration})

        return ann_stage_events, start_time_label

    
    def align_front(self, logger, start_time, psg_fname, ann_fname, signal, labels, fs):

        if start_time > 0:
            logger.info(f"Labeling started {start_time/60:.2f} min after signal start, signal will be shortened at the front to match")
            signal = signal[int(start_time*fs):]

        return True, signal, labels

    
    def align_end(self, logger, psg_fname, ann_fname, signals, labels):

        if len(labels) > len(signals):
            logger.info(f"Labels (len: {len(labels)}) are shortend to match signal ({len(signals)})")
            labels = labels[:len(signals)]

        if len(signals) > len(labels):
            logger.info(f"Signal (len: {len(signals)}) is shortend to match label (len: {len(labels)})")
            signals = signals[:len(labels)]

        assert len(signals) == len(labels), f"Length mismatch: signal ({os.path.basename(psg_fname)})={len(signals)}, labels({os.path.basename(ann_fname)})={len(labels)} TODO: implement alignment function"

        return signals, labels
