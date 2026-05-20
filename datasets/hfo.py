import pandas as pd
from pathlib import Path

from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("HFO")
class HFO(BaseDataset):
    """HFO - Dataset of EEG recordings containing HFO markings for 30 pediatric patients with epilepsy (ds003555) dataset"""

    def __init__(self):
        super().__init__("HFO","HFO - Dataset of EEG recordings containing HFO markings for 30 pediatric patients with epilepsy (ds003555)", keep_folder_structure=False)
    
    def _setup_dataset_config(self):
        self.ann2label = {"N2": "N2",
                          "N3": "N3",
                          "REM": "REM",
        }        
        
        self.channel_names = ['Fp1', 'A2', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 
                              'O1', 'A1', 'O2', 'T1', 'T2', 'Cz2']
        
        # see README
        self.inter_dataset_mapping = {
            'PSG_F3': self.Mapping(self.TTRef.F3, None),
            'PSG_F4': self.Mapping(self.TTRef.F4, None),
            'PSG_C3': self.Mapping(self.TTRef.C3, None),
            'PSG_C4': self.Mapping(self.TTRef.C4, None),
            'PSG_O1': self.Mapping(self.TTRef.O1, None),
            'PSG_O2': self.Mapping(self.TTRef.O2, None),
            'PSG_EOG': self.Mapping(self.TTRef.EL, self.TTRef.ER),
            'PSG_EOGL': self.Mapping(self.TTRef.EL, self.TTRef.LPA),
            'PSG_EOGR': self.Mapping(self.TTRef.ER, self.TTRef.LPA),
            'PSG_EMG': self.Mapping(self.TTRef.EMG_CHIN, None),
            'PSG_THER': self.Mapping(self.TTRef.AIRFLOW, None), # both PSG_THER and PSG_CAN measure AIRFLOW, which to choose?
            'PSG_THOR': self.Mapping(self.TTRef.THORACIC, None),
            'PSG_ABD': self.Mapping(self.TTRef.ABDOMINAL, None),
            'PSG_BEAT': self.Mapping(self.TTRef.HR, None),
            'PSG_SPO2': self.Mapping(self.TTRef.SPO2, None),
            'HB_1': self.Mapping(self.TTRef.AF7, None),
            'HB_2': self.Mapping(self.TTRef.AF8, None),
        }
        
        self.channel_types = {'analog': ['Fp1', 'A2', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 
                              'O1', 'A1', 'O2', 'T1', 'T2', 'Cz2'], 
                              'digital': []}
        
        
        self.channel_groups = {'eeg_eog': ['Fp1', 'A2', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 
                              'O1', 'A1', 'O2', 'T1', 'T2', 'Cz2'],
                                }
                
        self.file_extensions = {'psg_ext': '**/*hfo_eeg.edf',
                                'ann_ext': '**/DataIntervals.tsv'} 
        
    def get_file_identifier(self, psg_fname=None, ann_fname=None):
        psg_id, ann_id = None, None
        if psg_fname:
            psg_id = Path(psg_fname).parent.parts[-3:]
        if ann_fname:
            ann_id = Path(ann_fname).parent.parts[-3:]
        return psg_id, ann_id

    def dataset_paths(self):
        return ['', 'derivatives/']
    
    def ann_parse(self, ann_fname):
        annot = pd.read_csv(ann_fname,sep='\t', header=0)

        fs = 1024
        ann_stage_events = []

        start_time = None

        for i, row in annot.iterrows():
            if row["RunNb"] == 0:
                continue
            start = int(row['StartInd'])
            if start_time is None:
                start_time = start

            duration = int(row['EndInd']) - start

            stage = row['SleepStage']
            ann_stage_events.append({'Stage': stage,
                                    'Start': (start-start_time)/fs,
                                    'Duration': duration/fs})


        return ann_stage_events, start_time, None, None
    
    def align_front(self, logger, alignment, pad_values, epoch_duration, delay_sec, signal, labels, fs):

        return self.base_align_front(logger, delay_sec, alignment, pad_values, epoch_duration, signal, labels,fs) 
    
    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        # if len(signals) > len(labels):
        #     return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)
        if len(labels) > len(signals):
            return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)


#####
# DateInterval hat keine aneinandergereihten Scorings
# Entweder nur Run files nehmen oder raw files ohne Annotations?