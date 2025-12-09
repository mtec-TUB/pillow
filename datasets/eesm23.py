import os
import pandas as pd
from decimal import Decimal

from .base import BaseDataset
from .registry import register_dataset


@register_dataset("EESM23")
class EESM23(BaseDataset):
    """Ear-EEG Sleep Monitoring 2023 (EESM23) dataset"""
    
    def __init__(self):
        super().__init__("EESM23","Ear-EEG Sleep Monitoring 2023 (EESM23)", keep_folder_structure=False)
    
    def _setup_dataset_config(self):
        self.ann2label = {
                        "Wake": 0,   # Wake
                        "N1": 1,  # NREM Stage 1
                        "N2": 2,  # NREM Stage 2
                        "N3": 3,  # NREM Stage 3
                        "REM": 4,   # REM sleep
                        "Artefact": 6
                        }
        
        
        self.channel_names = ['F3', 'EMGl', 'F4', 'O2', 'O1', 'EMGr', 'EOGr', 'EMGc', 'C4', 'C3', 'M2', 'M1', 'EOGl']
        
        
        self.channel_types = {'analog': ['F3', 'EMGl', 'F4', 'O2', 'O1', 'EMGr', 'EOGr', 'EMGc', 'C4', 'C3', 'M2', 'M1', 'EOGl'], 'digital': []}
        
        
        self.channel_groups = {'eeg_eog': ['F3', 'F4', 'O2', 'O1', 'C4', 'C3', 'EOGr', 'EOGl','M2', 'M1'],
                                'emg': ['EMGl', 'EMGr', 'EMGc'],
                                }
                
        
        self.file_extensions = {'psg_ext': '**/*_task-sleep_acq-PSG_eeg.set',
                                'ann_ext': '**/*_task-sleep_acq-scoring_events.tsv'}
        
    def dataset_paths(self) -> tuple[str, str]:
        """
        EESM23 dataset paths.
        """
        data_dir = "Ear-EEG Sleep Monitoring 2023 (EESM23)"
        ann_dir = "Ear-EEG Sleep Monitoring 2023 (EESM23)"
        return data_dir, ann_dir
    
    def ann_parse(self, ann_fname, epoch_duration = None):
        annot = pd.read_csv(ann_fname,sep='\t', header=0)

        ann_stage_events = []
        start_time_label = None

        for i, row in annot.iterrows():
            start = row['onset']

            if start_time_label == None:
                start_time_label = Decimal(str(start))

            duration = row['duration']
            stage = row['scoring']
            ann_stage_events.append({'Stage': stage,
                                        'Start': float(Decimal(str(start)) - start_time_label),
                                        'Duration': duration})
            

        return ann_stage_events, float(start_time_label)
    
    def align_front(self, logger, start_time, psg_fname, ann_fname, signal, labels, fs):
        if not (start_time*fs).is_integer():
            raise Exception("Annotations start at timestamp outside of sample rate")

        if start_time > 0:
            logger.info(f"Labeling started {start_time/60:.2f} min after signal start, signal will be shortened at the front to match")
            signal = signal[int(start_time*fs):]

        return True, signal, labels
    
    def align_end(self, logger, psg_fname, ann_fname, signals, labels):

        # if len(labels) > len(signals):
        #     logger.info(f"Labels (len: {len(labels)}) are shortend to match signal ({len(signals)})")
        #     labels = labels[:len(signals)]

        if len(signals) == len(labels) + 1:
            logger.info(f"Signal (len: {len(signals)}) is shortend to match label (len: {len(labels)})")
            signals = signals[:len(labels)]

        assert len(signals) == len(labels), f"Length mismatch: signal ({os.path.basename(psg_fname)})={len(signals)}, labels({os.path.basename(ann_fname)})={len(labels)} TODO: implement alignment function"

        return signals, labels