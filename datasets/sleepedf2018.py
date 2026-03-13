import os
import numpy as np
import pyedflib
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("SLEEP-EDF")
class SleepEDF2018(BaseDataset):
    """Sleep-EDF-2018 dataset."""
    
    def __init__(self):
        super().__init__("SLEEP-EDF","Sleep-EDFX - Sleep-EDF Expanded")
        
    def _setup_dataset_config(self):
        self.ann2label = {
            "Sleep stage W": 0,      # Wake
            "Sleep stage 1": 1,      # NREM Stage 1
            "Sleep stage 2": 2,      # NREM Stage 2
            "Sleep stage 3": 3,      # NREM Stage 3
            "Sleep stage 4": 3,      # NREM Stage 4 (Follow AASM Manual)
            "Sleep stage R": 4,      # REM sleep
            "Sleep stage ?": 6,      # Unknown/Unscored
            "Movement time": 5       # Movement
        }

        self.inter_dataset_mapping = {
            "EOG horizontal": self.Mapping(self.TTRef.EL, self.TTRef.ER), 
            "EEG Fpz-Cz": self.Mapping(self.TTRef.Fpz, self.TTRef.Cz),
            "EEG Pz-Oz": self.Mapping(self.TTRef.Pz, self.TTRef.Oz),
            'EMG submental': self.Mapping(self.TTRef.EMG_CHIN, None),
        }

        self.channel_names =  [
            'EMG submental', 'Resp oro-nasal', 'EOG horizontal', 'Temp rectal', 
            'EEG Pz-Oz', 'Event marker', 'EEG Fpz-Cz', 'Marker'
        ]
        
        self.channel_types = {
            'analog': [
                'Resp oro-nasal', 'EEG Fpz-Cz', 'Temp rectal', 'EOG horizontal', 
                'EMG submental', 'EEG Pz-Oz', 'Event marker'
            ], 
            'digital': ['Marker']
        }
        
        self.channel_groups = {
            'eeg_eog': ['EOG horizontal', 'EEG Fpz-Cz', 'EEG Pz-Oz'],
            'emg': ['EMG submental'],
            'thoraco_abdo_resp': ['Resp oro-nasal']
        }
    
        self.file_extensions = {
            'psg_ext': '**/*0-PSG.edf',
            'ann_ext': '**/*-Hypnogram.edf'
        }
        
    def get_file_identifier(self, psg_fname=None, ann_fname=None):
        psg_id, ann_id = None, None
        if psg_fname:
            psg_ext = self.file_extensions['psg_ext'].split('*')[-1]
            psg_id = psg_fname.split(psg_ext)[0]
        if ann_fname:
            ann_ext = self.file_extensions['ann_ext'].split('*')[-1]
            ann_id = ann_fname.split(ann_ext)[0][:-1]
        return psg_id, ann_id
    
    def dataset_paths(self) -> Tuple[str, str]:
        return [
            '1.0.0',
            '1.0.0'
        ]
    
    def get_light_times(self, logger, psg_fname):

        psg_fname = os.path.basename(psg_fname)
        subject_id = int(psg_fname[3:5])
        subject_night = int(psg_fname[5])
        # print(subject_id, subject_night)
        if "SC4" in psg_fname:
            # Sleep-Cassette
            subjects = pd.read_excel(os.path.join(self.dset_dir,'1.0.0','SC-subjects.xls'))
            lights_off = subjects.loc[(subjects['subject'] == subject_id) & (subjects['night'] == subject_night), 'LightsOff'].values[0]
        elif "ST7" in psg_fname:
            # Sleep-Telemetry
            subjects = pd.read_excel(os.path.join(self.dset_dir,'1.0.0','ST-subjects.xls'), skiprows=1,names=["subject","Age", "Gender","Placebo_night_nr","Placebo_lights_off","Temazepam_night_nr","Temazepam_lights_off"])
            if subjects.loc[(subjects['subject'] == subject_id),"Placebo_night_nr"].values[0] == subject_night:
                lights_off = subjects.loc[(subjects['subject'] == subject_id), 'Placebo_lights_off'].values[0]
            elif subjects.loc[(subjects['subject'] == subject_id),"Temazepam_night_nr"].values[0] == subject_night:
                lights_off = subjects.loc[(subjects['subject'] == subject_id), 'Temazepam_lights_off'].values[0]
            else:
                raise Exception
        else:
            raise Exception
        
        return lights_off, None
    
    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        Parse Sleep-EDF-2018 EDF annotation files using PyEDF.
        
        Args:
            ann_fname: Path to EDF hypnogram file
            
        Returns:
            Tuple of (sleep_stage_events, start_datetime)
        """
        try:
            ann_f = pyedflib.EdfReader(ann_fname)
        except Exception as e:
            return [], None,None,None
        ann_onsets, ann_durations, ann_stages = ann_f.readAnnotations()
        ann_startdatetime = ann_f.getStartdatetime()
        start_offset = 0
        
        ann_stage_events = []
        
        for a in range(len(ann_stages)):
            onset_sec = int(ann_onsets[a])
            
            # Handle data gaps at the beginning
            if a == 0 and onset_sec != 0:
                ann_startdatetime = ann_startdatetime + timedelta(seconds=onset_sec)
                start_offset = onset_sec
            
            # Special handling for specific files with known gaps
            if 'ST7121JE-Hypnogram' in ann_fname and onset_sec == 30840:
                ann_stage_events.append({
                    'Stage': "Sleep stage ?",
                    'Start': 30810,
                    'Duration': 30
                })
                
            if 'ST7221JA-Hypnogram' in ann_fname and onset_sec == 32820:
                ann_stage_events.append({
                    'Stage': "Sleep stage ?",
                    'Start': 30870,
                    'Duration': 1950
                })
            
            duration_sec = int(ann_durations[a])
            ann_str = "".join(ann_stages[a])
            
            ann_stage_events.append({
                'Stage': ann_str,
                'Start': onset_sec - start_offset,
                'Duration': duration_sec
            })
        
        ann_f.close()
        return ann_stage_events, ann_startdatetime, None, None

    def align_front(self, logger, alignment, pad_values, epoch_duration, delay_sec, signal, labels, fs):

        return self.base_align_front(logger, delay_sec, alignment, pad_values, epoch_duration, signal, labels,fs)

    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(labels) > len(signals):
            return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)

        if ('telemetry' in psg_fname) and (len(signals) > len(labels)):
            return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)        
    
        
        