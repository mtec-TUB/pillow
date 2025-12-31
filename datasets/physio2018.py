import os
import numpy as np
import wfdb
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("PHYSIO2018")
class Physio2018(BaseDataset):
    """Physio2018 (PhysioNet Challenge 2018) dataset."""
    
    def __init__(self):
        super().__init__("PHYSIO2018","Physio2018 - PysioNet Challenge 2018", keep_folder_structure = False)
  
    def _setup_dataset_config(self):
        self.ann2label = {
                "W": 0,
                "N1": 1,
                "N2": 2,
                "N3": 3,
                "R": 4,
                }
        
        
        self.channel_names = ['ABD', 'E1-M2', 'C3-M2', 'ECG', 'O2-M1', 'O1-M2', 'F3-M2', 'C4-M1', 
                'F4-M1', 'SaO2', 'AIRFLOW', 'Chin1-Chin2', 'CHEST']
        
        
        self.channel_types = {'analog': ['ABD', 'E1-M2', 'C3-M2', 'ECG', 'O2-M1', 'O1-M2', 'F3-M2', 'C4-M1', 'F4-M1', 'AIRFLOW', 'Chin1-Chin2', 'CHEST'],
                'digital': ['SaO2']}
        
        
        self.channel_groups = {
            'eeg_eog': ['E1-M2', 'C3-M2','O2-M1', 'O1-M2', 'F3-M2', 'C4-M1', 'F4-M1'],
            'emg': ['Chin1-Chin2'],
            'ecg': ['ECG'],
            'thoraco_abdo_resp': ['ABD', 'AIRFLOW', 'CHEST']
        }
        
        
        self.file_extensions = {
            'psg_ext': '**/*.hea',
            'ann_ext': '**/*.arousal'
        }
    
    def dataset_paths(self) -> Tuple[str, str]:
        """
        Physio2018 dataset paths.
        """
        data_dir = "Physio2018 - PysioNet Challenge 2018/1.0.0/training"
        ann_dir = "Physio2018 - PysioNet Challenge 2018/1.0.0/training"
        return data_dir, ann_dir
    
    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        Parse Physio2018 annotation files.
        """
        ann_stage_events = []
        
        record_name, extension = os.path.splitext(ann_fname)
        annot = wfdb.rdann(record_name, extension.strip('.'))
        
        fs = annot.fs

        epoch_duration = 30  # Physio2018 uses default 30-second epochs, is calculated afterwards
        
        start_time_label = None

        for i, (sample, aux_note) in enumerate(zip(annot.sample, annot.aux_note)):
            print(sample,aux_note)
            if not any(note in aux_note for note in ['resp_hypoventilation','resp_cheynestokesbreath','arousal_bruxism','arousal_noise','arousal_plm','arousal_snore','arousal_rera','arousal_spontaneous','resp_partialobstructive','resp_centralapnea','resp_mixedapnea','resp_obstructiveapnea','resp_hypopnea']):
                if aux_note not in self.ann2label:
                       print(aux_note)
                       raise Exception
                    
                start = sample
                if start_time_label == None:
                    start_time_label = start
                ann_stage_events.append({'Stage': aux_note,
                                            'Start': float(start - start_time_label)/fs,
                                            'Duration': epoch_duration})    # place holder

        for i, event in enumerate(ann_stage_events[:-1]):
            ann_stage_events[i]['Duration'] = ann_stage_events[i+1]['Start'] - event['Start']

        return ann_stage_events, float(start_time_label)
    
    def align_front(self, logger, start_time, psg_fname, ann_fname, signal, labels, fs):
    
        start_seconds = start_time/fs

        if start_seconds > 0:
            logger.info(f"Labeling started {start_seconds/60:.2f} min after signal start, signal will be shortened at the front to match")
            signal = signal[int(start_seconds*fs):]

        return True, signal,labels

    def align_end(self, logger, psg_fname: str, ann_fname: str, signals: np.ndarray,
                  labels: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
        
        if len(signals) > len(labels):
            logger.info(f"Signal (len: {len(signals)}) is shortend to match label (len: {len(labels)})")
            signals = signals[:len(labels)]
        
        assert len(signals) == len(labels), f"Length mismatch: signal={len(signals)}, labels={len(labels)} \n TODO: implement alignment function"
        
        return signals, labels
    