"""
DREAMT (Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology)
"""
import os
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

        self._file_handler = None # DREAMT uses custom CSV handling directly implemented here


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
        return [
            'data',
            'data'
        ]
    
    def get_channels(self, logger, filepath):
        """Extract column names from DREAMT CSV files."""
        try:
            dataset = pd.read_csv(filepath, sep=",", header=0, nrows=1)
            # Exclude non-signal columns specific to DREAMT
            signal_columns = [col for col in dataset.columns 
                            if col not in ['TIMESTAMP', 'Sleep_Stage']]
            return signal_columns
        except Exception as e:
            logger.error(f"Error reading DREAMT CSV file {filepath}: {e}")
            return []

    def read_signal(self, logger, filepath, channel):
        """Read signal from DREAMT CSV file for specific channel."""
        try:
            dataset = pd.read_csv(filepath, sep=",", header=0)
            if channel in dataset.columns:
                # DREAMT-specific: Remove preparation stage data
                dataset = dataset[dataset["Sleep_Stage"] != "P"].reset_index()
                return dataset[channel].to_numpy()
        except Exception as e:
            logger.error(f"Error reading DREAMT CSV signal from {filepath}: {e}")
        return None

    def get_signal_data(self, logger, filepath, channel):
        """Get complete DREAMT CSV signal information for processing."""
        try:
            # DREAMT-specific sampling rate
            sampling_rate = 64
            dataset = pd.read_csv(filepath, sep=",", header=0)

            # DREAMT-specific preprocessing:
            # - Remove preparation stage 'P'
            dataset = dataset[dataset["Sleep_Stage"] != "P"].reset_index()
            signal = dataset[channel].to_numpy()

            logger.info(f"Channel selected: {channel}")
            logger.info(f"Select channel samples: {len(signal)}")

            file_duration = len(signal) / sampling_rate

            return {
                "signal": signal,
                "sampling_rate": sampling_rate,
                "start_datetime": None,
                "file_duration": file_duration,
            }
        except Exception as e:
            logger.error(f"Error processing DREAMT CSV file {filepath}: {e}")
            raise
    
    def ann_parse(self, ann_fname: str) -> Tuple[np.ndarray, int, List[Dict]]:
        """
        DREAMT annotation parsing.
        """
        sampling_rate = 64
        epoch_duration = 30 # DREAMT uses 30-second epochs
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
