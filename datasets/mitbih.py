import os
import numpy as np
import wfdb
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from datasets.base import BaseDataset
from datasets.registry import register_dataset

from datasets.file_handlers import WFDBHandler

@register_dataset("MIT-BIH")
class MITBIH(BaseDataset):
    """MIT-BIH - Polysomnographic Database dataset."""
    
    def __init__(self):
        super().__init__("MIT-BIH","MIT-BIH - Polysomnographic Database")

        self._file_handler = WFDBHandler()
  
    def _setup_dataset_config(self):
        self.ann2label = {
                "W": 0,
                "1": 1,
                "2": 2,
                "3": 3,
                "4": 3,
                "R": 4,
                "M": 5,
                }
        
        self.intra_dataset_mapping = {
            "Resp (abdomen)": ["Resp (abdomen)","Resp (abdominal)"],
            "rEOG": ["EOG (right)","EOG"],
            }
        
        # https://physionet.org/files/slpdb/1.0.0/slpdb.html
        self.inter_dataset_mapping = {
            "ECG": self.Mapping(self.TTRef.ECG, None),
            "EEG (C3-O1)": self.Mapping(self.TTRef.C3, self.TTRef.O1),
            "EEG (C4-A1)": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "EEG (O2-A1)": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "EMG": self.Mapping(self.TTRef.EMG_CHIN, None),
            "rEOG": self.Mapping(self.TTRef.ER, None),
            "SO2": self.Mapping(self.TTRef.SPO2, None),
            "Resp (abdomen)": self.Mapping(self.TTRef.ABDOMINAL, None),
            "Resp (chest)": self.Mapping(self.TTRef.THORACIC, None),
        }
        
        self.channel_names = ['BP', 'ECG', 'EEG (C3-O1)', 'EEG (C4-A1)',
                              'EEG (O2-A1)', 'EMG', 'EOG', 'EOG (right)', 
                              'Resp (abdomen)', 'Resp (abdominal)', 'Resp (chest)',
                              'Resp (nasal)', 'Resp (sum)', 'SO2', 'SV']
        
        
        self.channel_types = {'analog': ['Resp (abdomen)', 'Resp (nasal)', 'EMG',
                                         'SV', 'EOG', 'ECG', 'EEG (C3-O1)', 'EEG (C4-A1)',
                                         'SO2', 'BP', 'Resp (chest)', 'Resp (sum)', 
                                         'Resp (abdominal)', 'EOG (right)', 'EEG (O2-A1)'],
                              'digital': []}

        
        self.channel_groups = {
            'eeg_eog': ['EEG (C3-O1)', 'EEG (C4-A1)','EEG (O2-A1)','EOG', 'EOG (right)', ],
            'emg': ['EMG'],
            'ecg': ['ECG'],
            'thoraco_abdo_resp':[ 'Resp (abdomen)', 'Resp (abdominal)', 'Resp (chest)',
             'Resp (nasal)', 'Resp (sum)',]
        }
        
        # dataset contains two annotation files per psg file: '.st' and '.st-',
        # we use only '.st-' files because it's labels seem to be more accurat
        self.file_extensions = {
            'psg_ext': '*.hea',
            'ann_ext': 'slp*.st-'   
        }
    
    def dataset_paths(self) -> Tuple[str, str]:
        return [
            '',
            ''
        ]
    
    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        Parse MIT-BIH .st annotation files.
        """
        ann_stage_events = []
        
        record_name, extension = os.path.splitext(ann_fname)
        annot = wfdb.rdann(record_name, extension.strip('.'))
        
        fs = 250

        start_time_label = None
        
        for i, (sample, aux_note) in enumerate(zip(annot.sample, annot.aux_note)):
            label = aux_note.strip('\x00')
                            
            if start_time_label==None:
                if sample == 1:
                        sample = 0
                start_time_label = sample
                
            start = float(sample - start_time_label)/fs

            if "slp32.st-" in ann_fname and (start - 2)%30==0:
                # this file has some weird stages in between that are shifted by 2 seconds
                start = start-2

            ann_stage_events.append({'Stage': label,
                                        'Start': start,
                                        'Duration': 30})        #place holder, is calculated afterwards

        for i, event in enumerate(ann_stage_events[:-1]):
            ann_stage_events[i]['Duration'] = ann_stage_events[i+1]['Start'] - event['Start']

        return ann_stage_events, float(start_time_label)/fs
    
    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(signals) > len(labels):
            return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)
    