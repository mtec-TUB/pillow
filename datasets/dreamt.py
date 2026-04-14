"""
DREAMT (Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology)
"""
import os
import numpy as np
import pandas as pd
import polars as pl
import csv
from typing import Dict, List, Optional, Tuple
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("DREAMT")
class DREAMT(BaseDataset):
    """DREAMT dataset."""
    
    def __init__(self):
        super().__init__("DREAMT","DREAMT - Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology")

        self.use_sample_rate = 100

        self._file_handler = None # DREAMT uses custom CSV handling directly implemented here

        self.unit_dict = {
            'C4-M1': 'uV',
            'F4-M1': 'uV',
            'O2-M1': 'uV',
            'Fp1-O2': 'uV',
            'T3 - CZ': 'uV',
            'CZ - T4': 'uV',
            'CHIN': 'uV',
            'E1': 'uV',
            'E2': 'uV',
            'ECG': 'uV',
            'LAT': 'uV',
            'RAT': 'uV',
            'SNORE': 'uV',
            'PTAF': 'uV',
            'FLOW': 'uV',
            'THORAX': 'uV',
            'ABDOMEN': 'uV',
            'SAO2': '%',
            'BVP': 'a.u.',
            'IBI': 'ms',
            'ACC_X': '1/64g',
            'ACC_Y': '1/64g',
            'ACC_Z': '1/64g',
            'TEMP': '°C',
            'EDA': 'us',
            'HR': 'bpm'
        }

    def _setup_dataset_config(self):
        self.ann2label = {
            "W": 0,      # Wake
            "N1": 1,     # NREM Stage 1
            "N2": 2,     # NREM Stage 2
            "N3": 3,     # NREM Stage 3
            "R": 4,      # REM sleep
            "Missing": 6, # Unscored/Missing
            "P": 0     # Preparation stage, labeled as Wake as in described in https://physionet.org/content/dreamt/2.1.0/
        }        
        
        self.channel_names = ['C4-M1', 'F4-M1', 'O2-M1', 'Fp1-O2', 'T3 - CZ', 'CZ - T4', 'CHIN', 'E1', 'E2', 'ECG', 'LAT', 'RAT', 'SNORE', 'PTAF', 'FLOW', 
                              'THORAX', 'ABDOMEN', 'SAO2', 'BVP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'TEMP', 'EDA', 'HR', 'IBI']
        
        self.inter_dataset_mapping = {
            "C4-M1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "F4-M1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            "O2-M1": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "Fp1-O2": self.Mapping(self.TTRef.Fp1, self.TTRef.O2),
            "T3 - CZ": self.Mapping(self.TTRef.T7, self.TTRef.Cz),
            "CZ - T4": self.Mapping(self.TTRef.Cz, self.TTRef.T8),
            "CHIN": self.Mapping(self.TTRef.EMG_CHIN, None),
            "E1": self.Mapping(self.TTRef.EL, None),
            "E2": self.Mapping(self.TTRef.ER, None),
            "ECG": self.Mapping(self.TTRef.ECG, None),
            "LAT": self.Mapping(self.TTRef.EMG_LLEG, None),
            "RAT": self.Mapping(self.TTRef.EMG_RLEG, None),
            "SNORE": self.Mapping(self.TTRef.SNORE, None),
            "THORAX": self.Mapping(self.TTRef.THORACIC, None),
            "ABDOMEN": self.Mapping(self.TTRef.ABDOMINAL, None),
            "SAO2": self.Mapping(self.TTRef.SPO2, None),
            "HR": self.Mapping(self.TTRef.HR, None),
            "FLOW": self.Mapping(self.TTRef.AIRFLOW, None),
        }
        
        self.channel_types = {'analog': ['C4-M1', 'F4-M1', 'O2-M1', 'Fp1-O2', 'T3 - CZ', 'CZ - T4', 'CHIN', 'E1', 'E2', 'ECG', 'LAT', 'RAT', 'SNORE', 
                                         'PTAF', 'FLOW', 'THORAX', 'ABDOMEN', 'SAO2', 'BVP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'TEMP', 'EDA', 'HR'], 
                              'digital': ['IBI']}
        
        self.channel_groups = {}
        
        self.file_extensions = {
            'psg_ext': '*.csv',
            'ann_ext': '*.csv',  # Annotations are embedded in data CSV files
        }        
    
    def dataset_paths(self):
        if self.use_sample_rate == 100:
            return [
                '2.1.0/data_100Hz',
                '2.1.0/data_100Hz'
            ]
        elif self.use_sample_rate == 64:
            return [
                '2.1.0/data_64Hz',
                '2.1.0/data_64Hz'
            ]
        else:
            raise ValueError(f"Unsupported sample rate: {self.use_sample_rate}. Existing rates in dataset are 100Hz and 64Hz.")
    
    def get_channels(self, logger, filepath):
        """Extract column names from DREAMT CSV files."""
        try:
            dataset = pd.read_csv(filepath, sep=",", header=0, nrows=1)
            # Exclude non-signal columns specific to DREAMT
            signal_columns = [col for col in dataset.columns 
                            if col not in ['TIMESTAMP', 'Sleep_Stage','Obstructive_Apnea','Central_Apnea','Hypopnea','Multiple_Events']]
            return signal_columns
        except Exception as e:
            logger.error(f"Error reading DREAMT CSV file {filepath}: {e}")
            return []

    def read_signal(self, logger, filepath, channel):
        """Read signal from DREAMT CSV file for specific channel."""
        try:
            dataset = pl.read_csv(filepath, null_values=[""])
            if channel in dataset.columns:
                return dataset[channel].to_numpy()
        except Exception as e:
            logger.error(f"Error reading DREAMT CSV signal from {filepath}: {e}")
        return None
    

    def get_signal_data(self, logger, filepath, channel):
        """Get complete DREAMT CSV signal information for processing."""
        try:
            sampling_rate = self.use_sample_rate

            df = pl.read_csv(filepath, columns=[channel], null_values=[""], schema_overrides={channel: pl.Float64})
            signal = df[channel].to_numpy()
            missing_count = int(np.isnan(signal).sum())

            if missing_count == len(signal):
                # occurs for channel T3-Cz in file S062_PSG_df_updated.csv
                logger.warning(f"All values for channel {channel} in file {os.path.basename(filepath)} are missing. Skipping this channel")
                return {}

            file_duration = len(signal) / sampling_rate

            return {
                "signal": signal,
                "sampling_rate": sampling_rate,
                "unit": self.unit_dict.get(channel,'a.u.'),
                "start_datetime": None,
                "file_duration": file_duration,
            }

        except Exception as e:
            logger.error(f"Error processing DREAMT CSV file {filepath}: {e}")
            raise

    def ann_parse(self, ann_fname: str):
        """
        DREAMT annotation parsing.
        """
        sampling_rate = self.use_sample_rate
        epoch_duration = 30
        rows_per_epoch = sampling_rate * epoch_duration

        ann_stage_events = []
        start_time = None

        with open(ann_fname, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)

            header = next(reader)
            stage_idx = header.index("Sleep_Stage")
            time_idx = header.index("TIMESTAMP")


            for row_idx, row in enumerate(reader):
                # keep only the first sample/annotation of each epoch
                if row_idx % rows_per_epoch != 0:
                    continue

                stage = row[stage_idx]
                epoch_start = float(row[time_idx])

                if start_time is None:
                    start_time = epoch_start

                ann_stage_events.append({
                    "Stage": stage,
                    "Start": epoch_start - start_time,
                    "Duration": epoch_duration,
                })

        lights_off, lights_on = None, None
        return ann_stage_events, start_time, lights_off, lights_on
    
    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):
        # Labels can be one epoch longer than signal because signal gets cropped to full epochs and label exist for the last partial epoch
        return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)
