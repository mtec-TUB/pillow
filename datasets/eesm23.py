from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from decimal import Decimal

from datasets.base import BaseDataset
from datasets.registry import register_dataset

from datasets.file_handlers import EEGLABHandler

from datasets.eesm19 import EESM_Preprocessor


@register_dataset("EESM23")
class EESM23(BaseDataset):
    """Ear-EEG Sleep Monitoring 2023 (EESM23) dataset"""
    
    def __init__(self):
        super().__init__("EESM23","Ear-EEG Sleep Monitoring 2023 (EESM23)", keep_folder_structure=False)

        self._file_handler = EEGLABHandler()
    
    def _setup_dataset_config(self):
        self.ann2label = {
                        "Wake": 0,   # Wake
                        "N1": 1,  # NREM Stage 1
                        "N2": 2,  # NREM Stage 2
                        "N3": 3,  # NREM Stage 3
                        "REM": 4,   # REM sleep
                        "Artefact": 6
                        }
        
        
        self.channel_names = ['EOGr', 'EMGc', 'EMGl', 'C4', 'F3', 'EMGr', 'C3', 'M1', 'F4', 'EOGl', 'O2', 'RT', 'LT', 'LB', 'RB', 'O1', 'ELE', 'M2']

        self.inter_dataset_mapping = {
            "F3": self.Mapping(self.TTRef.F3, None),
            "F4": self.Mapping(self.TTRef.F4, None),
            "O1": self.Mapping(self.TTRef.O1, None),
            "O2": self.Mapping(self.TTRef.O2, None),
            "C3": self.Mapping(self.TTRef.C3, None),
            "C4": self.Mapping(self.TTRef.C4, None),
            "M1": self.Mapping(self.TTRef.LPA, None),
            "M2": self.Mapping(self.TTRef.RPA, None),
            "EOGl": self.Mapping(self.TTRef.EL, None),
            "EOGr": self.Mapping(self.TTRef.ER, None),
            "EMGc": self.Mapping(self.TTRef.EMG_CHIN, None),
            "EMGl": self.Mapping(self.TTRef.EMG_LCHIN, None),
            "EMGr": self.Mapping(self.TTRef.EMG_RCHIN, None),
        }
        
        
        self.channel_types = {'analog': ['C3', 'F4', 'F3', 'O2', 'RT', 'O1', 'LB', 'LT', 'EMGr', 'EOGl', 'EOGr', 'EMGl', 'EMGc', 'C4', 'RB', 'M1', 'M2','ELE'], 
                              'digital': []}
        
        self.channel_groups = {'eeg_eog': ['C3', 'F4', 'F3', 'O2', 'RT', 'O1', 'LB', 'LT', 'EOGl', 'EOGr', 'C4', 'RB', 'M1', 'M2','ELE'],
                                'emg': ['EMGl', 'EMGr', 'EMGc'],
                                }
                
        
        self.file_extensions = {'psg_ext': '**/*_eeg.set',
                                'ann_ext': '**/*_task-sleep_acq-scoring_events.tsv'}
        
    def get_file_identifier(self, psg_fname=None, ann_fname=None):
        psg_id, ann_id = None, None
        if psg_fname:
            psg_id = Path(psg_fname).parent
        if ann_fname:
            ann_id = Path(ann_fname).parent
        return psg_id, ann_id
    
    def dataset_paths(self):
        return [
            '',
            ''
        ]
    
    def ann_parse(self, ann_fname):
        annot = pd.read_csv(ann_fname,sep='\t', header=0)

        ann_stage_events = []
        start_time_label = None

        start_time_label = None
        for i, row in annot.iterrows():
            start = round(row['onset'])

            if start_time_label == None:
                start_time_label = Decimal(str(start))

            duration = row['duration']
            stage = row['scoring']
            ann_stage_events.append({'Stage': stage,
                                        'Start': float(Decimal(str(start)) - start_time_label),
                                        'Duration': duration})
            
        lights_off = float(start_time_label)        # see EESM23/code/dataset_preparation/BIDS_unittests_EESM23.py

        return ann_stage_events, float(start_time_label), lights_off, None
    
    def align_front(self, logger, alignment, pad_values, epoch_duration, delay_sec, signal, labels, fs):
        
        return self.base_align_front(logger, delay_sec, alignment, pad_values, epoch_duration, signal, labels,fs) 

    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(labels) > len(signals):
            if len(labels) > len(signals) + 1:
                logger.warning(
                    f"Labels longer than signals by {len(labels) - len(signals)} epochs in {psg_fname}, truncating.")
            return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)

        if len(signals) > len(labels):
            return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)

        logger.error(
            f"Unexpected alignment case in {psg_fname}: len(signals)={len(signals)}, len(labels)={len(labels)}")
        raise ValueError(f"Unexpected signal/label length combination: {len(signals)} signals, {len(labels)} labels")
    
    def preprocess(self, data_dir, ann_dir, output_dir):
        return EESM_Preprocessor(self).preprocess(data_dir, ann_dir, output_dir)
