import pandas as pd

from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("BDSP")
class BDSP(BaseDataset):
    """BDSP dataset"""

    def __init__(self):
        super().__init__("BDSP","bdsp", keep_folder_structure=False)

    
    def _setup_dataset_config(self):
        self.ann2label = {
                "W": "W",
                "N1": "N1",
                "N2": "N2",
                "N3": "N3",
                "R": "REM",
                "UNSCORED": "UNK",
                "L": "UNK", # all epochs before lights off and after lights on
                }
        
        # only for the first 7 subjects (have to run again when all files are downloaded)
        self.channel_names = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1', 'E2', 'CHIN', 'SNORE', 'NPT', 'C-FLOW', 'CHEST', 'ABDOMINAL', 
                              'LAT', 'RAT', 'EKG', 'RR', 'SaO2', 'Pleth', 'Sentec-TC', 'C PRESS', 'LEAK', 'ETCO2', 'THERM', 'PTAF', 'CPAP', 'THORACIC', 
                              'F7-A2', 'F8-A1', 'T3-A2', 'T4-A1', 'T5-A2', 'T6-A1', 'Flexor', 'Extensor', 'F3-C3', 'F4-C4', 'C3-P3', 'C4-P4', 'T3-T5', 
                              'T4-T6', 'T5-O1', 'T6-02', 'P3-01', 'P4-02']
        
        # only for the first 7 subjects (have to run again when all files are downloaded)
        self.channel_types = {'analog': ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1', 'E2', 'CHIN', 'SNORE', 'NPT', 'C-FLOW', 'CHEST', 
                                         'ABDOMINAL', 'LAT', 'RAT', 'EKG', 'RR', 'Pleth', 'C PRESS', 'LEAK', 'THERM', 'PTAF', 'CPAP', 'THORACIC', 'F7-A2', 
                                         'F8-A1', 'T3-A2', 'T4-A1', 'T5-A2', 'T6-A1', 'Flexor', 'Extensor', 'F3-C3', 'F4-C4', 'C3-P3', 'C4-P4', 'T3-T5', 
                                         'T4-T6', 'T5-O1', 'T6-02', 'P3-01', 'P4-02'], 
                              'digital': ['SaO2', 'Sentec-TC', 'ETCO2']}
        
        # have to be filled when all files are downloaded
        self.channel_groups = {'eeg_eog': [],
                                'emg': [],
                                'ecg': [],
                                'thoraco_abdo_resp': [],
                                'nasal_pressure': [],
                                'snoring': []
                                }
        
        self.file_extensions = {'psg_ext': '**/*-PSG_eeg.edf',
                                'ann_ext': '**/*-psg_sleep_annotations.csv'}
        

    def dataset_paths(self):
        return ['', '']
    
    def ann_parse(self, ann_fname):

        ann_df = pd.read_csv(ann_fname, sep=',', header=0)

        ann_stage_events = []
        start_time = None
        epoch_duration = 30

        for i, row in ann_df.iterrows():

            start = (row["Epoch"] - 1) * epoch_duration     # 1-based indexing
            stage = row['Stage']
            if pd.isna(stage):
                stage = "UNSCORED"

            if start_time is None:
                start_time = start

            ann_stage_events.append({'Stage': stage,
                                        'Start': start - start_time,
                                        'Duration': epoch_duration})

        sleep_epochs = ann_df[ann_df['Stage'] != 'L']
        lights_off = int((sleep_epochs.iloc[0]['Epoch'] - 1) * epoch_duration)
        lights_on = int(sleep_epochs.iloc[-1]['Epoch'] * epoch_duration)

        if lights_off is None:
            raise Exception(f"Could not determine lights off time for {ann_fname}")
        if lights_on is None:
            raise Exception(f"Could not determine lights on time for {ann_fname}")
        
        # events_file = ann_fname.replace('_sleep_annotations.csv', '_events_annotations.csv')

        return ann_stage_events, start_time, lights_off, lights_on
    
    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(labels) == len(signals) + 1:
            return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)
