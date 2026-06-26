import os
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("MHRW")
class MHRW(BaseDataset):
    """MHRW - Motion and heart rate from a wrist-worn wearable dataset."""
    
    def __init__(self):
        super().__init__("MHRW","MHRW - Motion and heart rate from a wrist-worn wearable")

    unit_mapping = {
        'hr': 'bpm',
        'acc_x': 'g',
        'acc_y': 'g',
        'acc_z': 'g',
        'steps': 'count',}


    def get_channels(self, logger, filepath):
        """available channels from file."""
        channels = []

        channels.append('heartrate')

        motion_file = filepath.replace('heart_rate', 'motion').replace('heartrate', 'acceleration')
        if os.path.exists(motion_file):
            channels.extend(['acc_x', 'acc_y', 'acc_z'])

        steps_file = filepath.replace('heart_rate', 'steps').replace('heartrate', 'steps')
        if os.path.exists(steps_file):
            channels.append('steps')

        return channels

    def read_signal(self, logger, filepath, channel):
        """Read signal data for a specific channel.
        Sampling rates from doi:10.1093/sleep/zsz180"""
        file_info = self.get_file_info(logger, filepath)
        start_time = file_info["start_datetime"]
        duration = file_info["file_duration"]

        if channel == 'heartrate':
            df = pd.read_csv(filepath, names=['timestamp', 'heartrate'],sep=',')
            aligned_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= start_time + duration)]
            sampling_rate = 1
        elif channel in ['acc_x', 'acc_y', 'acc_z']:
            motion_file = filepath.replace('heart_rate', 'motion').replace('heartrate', 'acceleration')
            df = pd.read_csv(motion_file, names=['timestamp', 'acc_x', 'acc_y', 'acc_z'], sep=' ')
            aligned_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= start_time + duration)]
            sampling_rate = 50
        elif channel == 'steps':
            steps_file = filepath.replace('heart_rate', 'steps').replace('heartrate', 'steps')
            df = pd.read_csv(steps_file, names=['timestamp', 'steps'], sep=',')
            aligned_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= start_time + duration)]
            sampling_rate = 1


        f = interp1d(aligned_df['timestamp'], aligned_df[channel], kind='nearest', fill_value='extrapolate')
        signal_interpl = f(np.arange(start_time, start_time + duration, 1/sampling_rate))

        return signal_interpl
    
    def get_time_range(self, logger, filepath, sep=','):
        times = pd.read_csv(filepath, sep=sep).values[:,0]
        return float(times[0]), float(times[-1])
    
    def get_file_info(self, logger, filepath):
        """Get start datetime and file duration."""

        hr_start_time, hr_end_time = self.get_time_range(logger, filepath)

        motion_file = filepath.replace('heart_rate', 'motion').replace('heartrate', 'acceleration')
        motion_start_time, motion_end_time = self.get_time_range(logger, motion_file, sep=' ') if os.path.exists(motion_file) else (np.nan, np.nan)

        # steps_file = filepath.replace('heart_rate', 'steps').replace('heartrate', 'steps')
        # steps_start_time, steps_end_time = self.get_time_range(logger, steps_file) if os.path.exists(steps_file) else (np.nan, np.nan)

        overlap = motion_start_time and (hr_start_time <= motion_end_time) and (motion_start_time <= hr_end_time)
        
        if (overlap): 
            start_time = max([hr_start_time, motion_start_time])
            end_time = min([hr_end_time, motion_end_time])
        else:
            return {}

        file_duration = (end_time - start_time)

        return {"start_datetime": start_time,
                "file_duration": file_duration}
    
    def get_signal_data(self, logger, filepath, channel):
        """Get complete signal information for specific channel.
        Sampling rates from doi:10.1093/sleep/zsz180"""
        # Get start time to align signals from different files
        file_info = self.get_file_info(logger, filepath)
        start_time = file_info["start_datetime"]
        duration = file_info["file_duration"]

        if channel == 'heartrate':
            df = pd.read_csv(filepath, names=['timestamp', 'heartrate'], sep=',')
            aligned_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= start_time + duration)]
            sampling_rate = 1
        elif channel in ['acc_x', 'acc_y', 'acc_z']:
            motion_file = filepath.replace('heart_rate', 'motion').replace('heartrate', 'acceleration')
            df = pd.read_csv(motion_file, names=['timestamp', 'acc_x', 'acc_y', 'acc_z'], sep=' ')
            aligned_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= start_time + duration)]
            sampling_rate = 50
        elif channel == 'steps':
            steps_file = filepath.replace('heart_rate', 'steps').replace('heartrate', 'steps')
            df = pd.read_csv(steps_file, names=['timestamp', 'steps'], sep=',')
            aligned_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= start_time + duration)]
            sampling_rate = 1

        if aligned_df.empty:
            logger.warning(f"Channel \"{channel}\" has no data in the common time range of heartrate and acceleration for file {filepath}. Will be ignored.")
            return {}
        
        f = interp1d(aligned_df['timestamp'], aligned_df[channel], kind='nearest', fill_value='extrapolate')
        signal_interpl = f(np.arange(start_time, start_time + duration, 1/sampling_rate))
        unit = self.unit_mapping.get(channel, 'u.a.')        

        return {
            "signal": signal_interpl,
            "sampling_rate": sampling_rate,
            "unit": unit
        }


    def _setup_dataset_config(self):
        self.ann2label = {0: "W",
                          1: "N1",
                          2: "N2",
                          3: "N3",
                          4: "N3", # According to AASM
                          5: "REM",
                          -1: "UNK"}

        
        self.channel_names =  ['heartrate', 'acc_x', 'acc_y', 'acc_z', 'steps']
        
        
        self.channel_types ={'analog': ['acc_x', 'acc_y', 'acc_z'], 
                             'digital': ['heartrate', 'steps']} 
        
        self.channel_groups = {}
        
        self.file_extensions = {
                                'psg_ext': '**/*_heartrate.txt',    # used as common reference per patient (channels are derived from _heartrate.txt, _motion.txt and _steps.txt)
                                'ann_ext': '**/*_labeled_sleep.txt'
                                }
        

    def dataset_paths(self):
        return ['', '']
    
    def get_file_identifier(self, psg_fname=None, ann_fname=None):
        """Used to find corresponding PSG and annotation files based on filename patterns
        """
        psg_id, ann_id = None, None
        if psg_fname:
            psg_ext = self.file_extensions['psg_ext'].split('*')[-1]
            psg_id = os.path.basename(psg_fname).split(psg_ext)[0]
        if ann_fname:
            ann_ext = self.file_extensions['ann_ext'].split('*')[-1]
            ann_id = os.path.basename(ann_fname).split(ann_ext)[0]
        return psg_id, ann_id
    
    def ann_parse(self, ann_fname):
        """
        function to parse the annotation file of the dataset into sleep stage events with start and duration

        """
        ann_stage_events = []
        ann_df = pd.read_csv(ann_fname, sep=' ', names=["Timestamp", "Stage"])
        start_time = self.get_file_info(None, ann_fname.replace('labels','heart_rate').replace('_labeled_sleep.txt', '_heartrate.txt'))["start_datetime"]
        lights_off, lights_on = abs(start_time), abs(start_time) + int(ann_df.iloc[-1]["Timestamp"])

        epoch_duration = 30  # seconds

        ann_stage_events = []
        for i,row in ann_df.iterrows():
            stage = row['Stage']
            start = row['Timestamp']
            ann_stage_events.append({'Stage': stage,
                                    'Start': start,
                                    'Duration': epoch_duration})

        return ann_stage_events, abs(start_time), lights_off, lights_on
    

    def align_front(self, logger, alignment, pad_values, epoch_duration, delay_sec, signal, labels, fs):
        """ Align front part of signals and labels, in some datasets annotations start after signal recording"""

        return self.base_align_front(logger, delay_sec, alignment, pad_values, epoch_duration, signal, labels,fs)

    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(labels) > len(signals):
            return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)

        if len(signals) > len(labels):
            return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)   
