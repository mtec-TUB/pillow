import os

import pandas as pd
from datetime import datetime, timedelta

from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("PSG_IPA")
class PSG_IPA(BaseDataset):
    """PSG-IPA - A PolySomnoGraphic Inter-scorer Performance Assessment database"""
    
    def __init__(self):
        super().__init__("PSG_IPA","PSG-IPA - A PolySomnoGraphic Inter-scorer Performance Assessment database", keep_folder_structure=False)
  
    def _setup_dataset_config(self):
        self.ann2label = {
            "Sleep stage W": "W",
            "Sleep stage N1": "N1",
            "Sleep stage N2": "N2",
            "Sleep stage N3": "N3",
            "Sleep stage R": "REM",
            "Unscored": "UNK",
        }

        self.inter_dataset_mapping = {
            'EEG F4-M1': self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            'EEG C4-M1': self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            'EEG O2-M1': self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            'EMG chin': self.Mapping(self.TTRef.EMG_CHIN, None),
            'EOG E1-M2': self.Mapping(self.TTRef.EL, self.TTRef.RPA),
            'EOG E2-M2': self.Mapping(self.TTRef.ER, self.TTRef.RPA),
            'EEG Cz-M1': self.Mapping(self.TTRef.Cz, self.TTRef.LPA),
            'ECG': self.Mapping(self.TTRef.ECG, None),
            'EMG LAT': self.Mapping(self.TTRef.EMG_LLEG, None),
            'EMG RAT': self.Mapping(self.TTRef.EMG_RLEG, None),
            'SaO2': self.Mapping(self.TTRef.SPO2, None),
            'Resp abdomen': self.Mapping(self.TTRef.ABDOMINAL, None),
            'Resp chest': self.Mapping(self.TTRef.THORACIC, None),
        }
        

        self.channel_names = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EMG chin', 'EOG E1-M2', 'EOG E2-M2', 'ECG', 'EMG LAT', 'EMG RAT', 'Resp nasal', 
                              'Resp abdomen', 'Resp chest', 'SaO2', 'EEG Cz-M1']
        
        
        self.channel_types = {'analog': ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EMG chin', 'EOG E1-M2', 'EOG E2-M2', 'ECG', 'EMG LAT', 'EMG RAT', 
                                         'Resp nasal', 'Resp abdomen', 'Resp chest', 'EEG Cz-M1'],
                              'digital': ['SaO2']}
    
        
        self.channel_groups = {
            'eeg_eog': ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EOG E1-M2', 'EOG E2-M2', 'EEG Cz-M1'],
            'emg': ['EMG chin', 'EMG LAT', 'EMG RAT'],
            'ecg': ['ECG'],
            'thoraco_abdo_resp': ['Resp nasal', 'Resp abdomen', 'Resp chest'],
        }        
        
        
        self.file_extensions = {
            'psg_ext': '**/PSG/*.edf',
            'ann_ext': '**/Annotations/semiauto/*_semiauto_scorer1.txt' # use Scorer 1 with semiautomatic annotations (computer aided)
        }

    def dataset_paths(self):
        return ['', '']
    
    def get_file_identifier(self, psg_fname=None, ann_fname=None):
        psg_id, ann_id = None, None
        if psg_fname:
            psg_ext = self.file_extensions['psg_ext'].split('*')[-1]
            psg_id = os.path.basename(psg_fname).split(psg_ext)[0]
        if ann_fname:
            ann_ext = self.file_extensions['ann_ext'].split('*')[-1]
            ann_id = os.path.basename(ann_fname).split(ann_ext)[0]
        return psg_id, ann_id
    
    def ann_parse(self, ann_fname):
        annot = pd.read_csv(ann_fname,sep=',', header=0,skipinitialspace=True)

        ann_stage_events = []
        first_row = annot.iloc[0]
        ann_Startdatetime = datetime.strptime(first_row['Date'] + ' ' + first_row['Time'],'%d.%m.%y %H.%M.%S')
        
        for i, row in annot.iterrows():
            label = str(row['Annotation'])
            if label == 'Lights off':
                lights_off = ann_Startdatetime + timedelta(seconds=row['Recording onset'])
            elif label == 'Lights on':
                lights_on = ann_Startdatetime + timedelta(seconds=row['Recording onset'])
            elif not any(stage in label.lower() for stage in ['eeg arousal', 'hypopnea','limb movement', 'central apnea','obstructive apnea','mixed apnea']):
                start = row['Recording onset']
                duration = row['Duration']
                ann_stage_events.append({'Stage': label,
                                            'Start': start,
                                            'Duration': duration})

        return ann_stage_events, ann_Startdatetime, lights_off, lights_on
    
    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(signals) > len(labels):
            return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)