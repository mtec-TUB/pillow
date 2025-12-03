import os
import numpy as np
import wfdb
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from .base import BaseDataset
from .registry import register_dataset

@register_dataset("SLEEPBRL")
class SLEEPBRL(BaseDataset):
    """SLEEPBRL - Sleep Bioradiolocation Database dataset."""
    
    def __init__(self):
        super().__init__("SLEEPBRL","SLEEPBRL - Sleep Bioradiolocation Database")
  
    def _setup_dataset_config(self):
        self.ann2label = {
                "W": 0,
                "1": 1,
                "2": 2,
                "3": 3,
                "R": 4,
                }
        
        self.alias_mapping = {
            "Resp (abdomen)": ["Resp (abdomen)","Resp (abdominal)"]
            }
        
        self.channel_names = ['S9', 'S14', 'S1', 'S10', 'S5', 'S6', 'S3',
                              'S15', 'S4', 'S13', 'S2', 'S11', 'S8', 'S7', 'S12', 'S16']
        
        
        self.channel_types = {'analog': ['S9', 'S14', 'S1', 'S10', 'S5', 'S6', 'S3',
                                          'S15', 'S4', 'S13', 'S2', 'S11', 'S8', 'S7', 'S12', 'S16'],
                             'digital': []}

        
        self.channel_groups = {}
        
        
        self.file_extensions = {
            'psg_ext': '*.edf',
            'ann_ext': '*.edf.atr'
        }

    
    def dataset_paths(self) -> Tuple[str, str]:
        """
        SLEEPBRL dataset paths.
        """
        data_dir = "SLEEPBRL - Sleep Bioradiolocation Database"
        ann_dir = "SLEEPBRL - Sleep Bioradiolocation Database"
        return data_dir, ann_dir
    
    def ann_parse(self, ann_fname: str, epoch_duration: Optional[int] = None) -> Tuple[List[Dict], datetime]:
        """
        Parse SLEEPBRL .atr annotation files.
        """
        ann_stage_events = []
        
        record_name, extension = os.path.splitext(ann_fname)
        annot = wfdb.rdann(record_name, extension.strip('.'))

        fs = 50 # 50 Hz for all channels

        for i, (sample, aux_note) in enumerate(zip(annot.sample, annot.aux_note)):

            start = sample/fs 

            ann_stage_events.append({'Stage': aux_note,
                                        'Start': start,
                                        'Duration': epoch_duration})        #place holder

        for i, event in enumerate(ann_stage_events[:-1]):
            ann_stage_events[i]['Duration'] = ann_stage_events[i+1]['Start'] - event['Start']

        return ann_stage_events, None

    