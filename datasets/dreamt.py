"""
DREAMT (Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("DREAMT")
class DREAMT(BaseDataset):
    """DREAMT dataset."""
    
    def __init__(self):
        super().__init__("DREAMT","DREAMT - Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology")

    def _setup_dataset_config(self):
        self.ann2label = {
            "W": 0,      # Wake
            "N1": 1,     # NREM Stage 1
            "N2": 2,     # NREM Stage 2
            "N3": 3,     # NREM Stage 3
            "R": 4,      # REM sleep
            "Missing": 6 # Unscored/Missing
        }        
        
        self.channel_names = [
            'BVP','ACC_X','ACC_Y','ACC_Z','TEMP','EDA','HR'
        ]
        
        
        self.channel_types = {
            'analog': ['BVP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'TEMP', 'EDA', 'HR'],
            'digital': []
        }
        
        self.channel_groups = {}
        
        self.file_extensions = {
            'psg_ext': '*.csv',
            'ann_ext': '*.csv',  # Annotations are embedded in data CSV files
        }        
        

    
    def dataset_paths(self) -> Tuple[str, str]:
        """
        DREAMT dataset paths.
        """
        data_dir = "DREAMT - Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology/data"
        ann_dir = "DREAMT - Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology/data"
        return data_dir, ann_dir
    
    def ann_parse(self, ann_fname: str, epoch_duration: Optional[int] = None) -> Tuple[np.ndarray, int, List[Dict]]:
        """
        DREAMT annotation parsing.
        """
        sampling_rate = 64
        epoch_duration = 30
        dataset = pd.read_csv(ann_fname, sep=",", header=0)
        
        # Prepare dataset to get labels:
        # - starting after Preparation Stage 'P'
        # - filter out labels only after a full epoch (30 seconds)
        dataset = dataset[dataset['Sleep_Stage'] != 'P'].reset_index()
        dataset = dataset.iloc[((dataset.index == 0) | (dataset.index + 1) % (sampling_rate * epoch_duration) == 0)]
        
        ann_stage_events = []
        start_time = None
        
        for i, row in dataset.iterrows():
            stage = row['Sleep_Stage']
            epoch_start = float(row['TIMESTAMP'])
            if start_time == None:
                start_time = epoch_start
            duration = epoch_duration
            ann_stage_events.append({
                'Stage': stage,
                'Start': epoch_start - start_time,
                'Duration': duration
            })
        
        return ann_stage_events, None
