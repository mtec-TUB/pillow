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
                }
        
        self.intra_dataset_mapping = {
            "Resp (abdomen)": ["Resp (abdomen)","Resp (abdominal)"]
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
        
        
        self.file_extensions = {
            'psg_ext': '*.hea',
            'ann_ext': 'slp*.st'
        }
    
    def dataset_paths(self) -> Tuple[str, str]:
        """
        MIT-BIH dataset paths.
        """
        data_dir = "MIT-BIH - Polysomnographic Database"
        ann_dir = "MIT-BIH - Polysomnographic Database"
        return data_dir, ann_dir
    
    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        Parse MIT-BIH .st annotation files.
        """
        ann_stage_events = []
        
        record_name, extension = os.path.splitext(ann_fname)
        annot = wfdb.rdann(record_name, extension.strip('.'))
        
        fs = annot.fs

        epoch_duration = 30  # default epoch duration, is calculated afterwards

        start_time_label = None
        
        for i, (sample, aux_note) in enumerate(zip(annot.sample, annot.aux_note)):
            aux_note = aux_note.strip('\x00')
            
            parts = aux_note.split()
            
            # often several annotations in one string
            if parts and parts[0] in self.ann2label:
                label = parts[0]      
            else:
                label = aux_note
            
            if not any(note in label for note in ['LA','HA','MT','CA','X','H','L','A','M']):
                if label not in self.ann2label:
                    print(label)
                    raise Exception
                
                if start_time_label==None:
                    if sample == 1:
                        sample = 0
                    start_time_label = sample
                    
                start = float(sample - start_time_label)/fs

                ann_stage_events.append({'Stage': label,
                                            'Start': start,
                                            'Duration': epoch_duration})        #place holder

        for i, event in enumerate(ann_stage_events[:-1]):
            ann_stage_events[i]['Duration'] = ann_stage_events[i+1]['Start'] - event['Start']

        return ann_stage_events, float(start_time_label)
    
    def align_front(self, logger, alignment, pad_values, epoch_duration, delay_samples, signal, labels, fs):

        delay_sec = delay_samples/fs

        return self.base_align_front(logger, delay_sec, alignment, pad_values, epoch_duration, signal, labels,fs)

    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if 'slp66.st' in ann_fname:
            return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)
    