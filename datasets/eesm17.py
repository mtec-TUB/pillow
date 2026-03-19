import os
import pandas as pd
from pathlib import Path
import glob

from datasets.base import BaseDataset
from datasets.registry import register_dataset

from datasets.file_handlers import EEGLABHandler


@register_dataset("EESM17")
class EESM17(BaseDataset):
    """Ear-EEG Sleep Monitoring 2017 (EESM17) dataset"""
    
    def __init__(self):
        super().__init__("EESM17","Ear-EEG Sleep Monitoring 2017 (EESM17)", keep_folder_structure=False)

        self._file_handler = EEGLABHandler()
    
    def _setup_dataset_config(self):
        self.ann2label = {
                        1: 0,   # Wake
                        3: 1,   # NREM Stage 1
                        4: 2,   # NREM Stage 2
                        5: 3,   # NREM Stage 3
                        2: 4,   # REM sleep
                        7: 6,   # Artefact
                        8: 6,   # Unknown
                        }
        
        
        self.channel_names = ['LOC', 'OSAT', 'DIF3', 'DIF2', 'CHIN12', 'ERB1', 'ERK', 'O1', 'A2', 'ELE', 'DC1', 'ELA', 'ELB', 'PR', 'O2', 'ERE', 
                              'ROC', 'ELG', 'ERI', 'C4', 'ERB', 'ELK', 'DIF4', 'ERA', 'A1', 'DC3', 'C3', 'ELI', 'F3', 'Event', 'ERG', 'ELB1', 
                              'F4', 'DC4', 'DC2']
        
        self.inter_dataset_mapping = {
            "F3": self.Mapping(self.TTRef.F3, None),
            "F4": self.Mapping(self.TTRef.F4, None),
            "O1": self.Mapping(self.TTRef.O1, None),
            "O2": self.Mapping(self.TTRef.O2, None),
            "C3": self.Mapping(self.TTRef.C3, None),
            "C4": self.Mapping(self.TTRef.C4, None),
            "A1": self.Mapping(self.TTRef.LPA, None),
            "A2": self.Mapping(self.TTRef.RPA, None),
            "LOC": self.Mapping(self.TTRef.EL, None),
            "ROC": self.Mapping(self.TTRef.ER, None),
            "OSAT": self.Mapping(self.TTRef.SPO2, None),
            "CHIN12": self.Mapping(self.TTRef.EMG_LCHIN, self.TTRef.EMG_RCHIN),
            "PR": self.Mapping(self.TTRef.HR, None),
        }
        
        self.channel_types = {'analog': ['C4', 'ELG', 'DC3', 'ERI', 'DC2', 'F4', 'ELK', 'O2', 'ERA', 'ELB', 'ERB', 'O1', 'A1', 'C3', 'ELA', 'ROC', 'DC1', 
                                         'ERK', 'ERG', 'LOC', 'DIF4', 'ELI', 'ELB1', 'DC4', 'ELE', 'ERE', 'DIF3', 'A2', 'F3', 'DIF2', 'CHIN12', 'ERB1'], 
                              'digital': ['OSAT', 'Event', 'PR']}
        
        
        self.channel_groups = {'eeg_eog': ['C4', 'ELG', 'ERI', 'F4', 'ELK', 'O2', 'ERA', 'ELB', 'ERB', 'O1', 'A1', 'C3', 'ELA', 'ROC', 
                                         'ERK', 'ERG', 'LOC', 'ELI', 'ELB1', 'ELE', 'ERE',  'A2', 'F3', 'ERB1'],
                                'emg': ['CHIN12',],
                                }
                
        
        self.file_extensions = {'psg_ext': '**/*_eeg.set',
                                'ann_ext': '**/*_acq-scoring_events.tsv'} 
        
    def dataset_paths(self) -> tuple[str, str]:
        return ['', '']
    
    def ann_parse(self, ann_fname):
        annot = pd.read_csv(ann_fname,sep='\t', header=0)

        ann_stage_events = []

        # Start time of labels is "Lights Off" marker in sleep events file
        try:
            base_folder = Path(ann_fname).parent 
            events_file = glob.glob(os.path.join(base_folder, "*sleep_events.tsv"))[0]
            events = pd.read_csv(events_file,sep='\t', header=0)
            start_time = events["onset"][events["trial_type"] == "Lights Off"].iloc[0]
        except:
            print(f"Error while trying to retrieve Lights Off marker for {ann_fname}")


        for i, row in annot.iterrows():
            start = row['onset']

            duration = 30 # default epoch-duration, correct duration is calculated afterwards
            stage = row['staging']
            ann_stage_events.append({'Stage': stage,
                                        'Start': start,
                                        'Duration': duration})
        
        for i, event in enumerate(ann_stage_events[:-1]):
            ann_stage_events[i]['Duration'] = ann_stage_events[i+1]['Start'] - event['Start']

        events_file = ann_fname.replace("acq-scoring_events", "events")
        events = pd.read_csv(events_file,sep='\t', header=0)

        lights_off = events.loc[events['trial_type'] == 'Lights Off', 'onset']
        if len(lights_off) == 1:
            lights_off = lights_off.iloc[0]
        else:
            raise Exception(f"Expected exactly one 'Lights Off' event in {events_file}, but found {len(lights_off)}.")
        
        lights_on = events.loc[events['trial_type'] == 'Lights On', 'onset']
        if len(lights_on) == 1:
            lights_on = lights_on.iloc[0]
        else:
            raise Exception(f"Expected exactly one 'Lights On' event in {events_file}, but found {len(lights_on)}.")

        return ann_stage_events, start_time, lights_off, lights_on
    
    def align_front(self, logger, alignment, pad_values, epoch_duration, delay_sec, signal, labels, fs):

        return self.base_align_front(logger, delay_sec, alignment, pad_values, epoch_duration, signal, labels,fs) 
    
    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(signals) > len(labels):
            return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)
        elif len(labels) == len(signals) + 1:
            return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)
