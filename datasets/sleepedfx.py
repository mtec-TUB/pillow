import os
from mne import read_annotations
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("SLEEP-EDFX")
class SleepEDFX(BaseDataset):
    """Sleep-EDFX dataset."""
    
    def __init__(self):
        super().__init__("SLEEP-EDFX","Sleep-EDFX - Sleep-EDF Expanded")
        self.has_front_alignment = True
        self.has_end_alignment = True

    def _setup_dataset_config(self):
        self.ann2label = {
            "Sleep stage W": "W",     # Wake
            "Sleep stage 1": "N1",      # NREM Stage 1
            "Sleep stage 2": "N2",      # NREM Stage 2
            "Sleep stage 3": "N3",      # NREM Stage 3
            "Sleep stage 4": "N3",      # NREM Stage 4 (Follow AASM Manual)
            "Sleep stage R": "REM",      # REM sleep
            "Sleep stage ?": "UNK",      # Unknown/Unscored
            "Movement time": "MOVE",       # Movement
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
    
    def dataset_paths(self):
        return [
            '1.0.0',
            '1.0.0'
        ]
    
    def get_light_times(self, logger, psg_fname):

        psg_fname = os.path.basename(psg_fname)
        subject_id = int(psg_fname[3:5])
        subject_night = int(psg_fname[5])
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
    
    def ann_parse(self, ann_fname: str):
        """
        Parse Sleep-EDF-2018 EDF annotation files using MNE.
        
        Args:
            ann_fname: Path to EDF hypnogram file
            
        Returns:
            Tuple of (sleep_stage_events, start_datetime)
        """
        ann_f = read_annotations(ann_fname)

        ann_onsets = ann_f.onset
        ann_durations = ann_f.duration
        ann_stages = ann_f.description

        ann_start_time = ann_onsets[0]
        
        ann_stage_events = []
        
        for i in range(len(ann_onsets)-1):

            ann_stage_event = {
                'Stage': ann_stages[i],
                'Start': ann_onsets[i] - ann_start_time,
                'Duration': ann_durations[i]
            }
            ann_stage_events.append(ann_stage_event)
            # Fill out missing annotations with 'Sleep stage ?' (happens only one time in 'ST7121JE-Hypnogram' and 'ST7221JA-Hypnogram')
            if ann_onsets[i] + ann_durations[i] != ann_onsets[i+1]:
                ann_stage_event = {
                    'Stage': "Sleep stage ?",
                    'Start': (ann_onsets[i] + ann_durations[i]) - ann_start_time,
                    'Duration': ann_onsets[i+1] - (ann_onsets[i] + ann_durations[i])
                }
                ann_stage_events.append(ann_stage_event)
        
        # add last ann_stage_event because loop stopped before
        ann_stage_events.append({
                'Stage': ann_stages[-1],
                'Start': ann_onsets[-1] - ann_start_time,
                'Duration': ann_durations[-1]
        })
        return ann_stage_events, ann_start_time, None, None


