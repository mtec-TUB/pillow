"""
NCHSDB - NCH Sleep DataBank
"""
import pandas as pd
from decimal import Decimal
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("NCHSDB")
class NCHSDB(BaseDataset):
    """NCHSDB - NCH Sleep DataBank dataset."""
    
    def __init__(self):
        super().__init__("NCHSDB","NCHSDB - NCH Sleep DataBank")
    
    def _setup_dataset_config(self):
        self.ann2label = {
                'Sleep stage W': 0,
                'Sleep stage N1': 1,
                'Sleep stage N2': 2,
                'Sleep stage N3': 3,
                'Sleep stage R': 4,
                'Sleep stage ?': 6,
                'Sleep stage 2': 2,
                'Sleep stage 1': 1,
                'Sleep stage 3' : 3,
        }

        self.alias_mapping = {'Abdomen': ['Abdominal','EEG Abd'],
                              'Resp Abdomen': ['Resp Abdominal', 'Resp Abdomen'],
                              'C-flow': ['C-flow','C-Flow'],
                              'Cz-O1': ['EEG Cz-O1','EEG CZ-O1'],
                              'ECG2': ['EEG EKG2','EKG2'],
                              'ECG2-ECG': ['ECG EKG2-EKG','EEG EKG2-EKG'],
                              'EMG LLeg-RLeg': ['EMG LLEG-RLEG','EMG LLeg-RLeg'],
                              'F3': ['EEG F3','F3'],
                              'M1': ['EEG M1','M1'],
                              'M2': ['EEG M2','M2'],
                              'F4': ['EEG F4','F4'],
                              'C4': ['EEG C4','C4'],
                              'C3': ['EEG C3','C3'],
                              'O1': ['EEG O1','O1'],
                              'O2': ['EEG O2','O2'],
                              'LOC-M2': ['EOG LOC-M2','EEG LOC-M2'],
                              'ROC-M1': ['EOG ROC-M1','EEG ROC-M1'],
                              'Chin1-Chin2': ['EMG Chin1-Chin2','EMG CHIN1-CHIN2','EEG Chin1-Chin2'],
                              'Chin1-Chin3': ['EEG Chin1-Chin3','EMG Chin1-Chin3','EMG CHIN1-CHIN3'],
                              'Snore':['EEG Snore','Snore'],
                              'SnoreDR': ['Snore_DR','SNORE_DR'],
                              'SpO2': ['SpO2','OSAT','Osat'],
                              'Resp Chest': ['Resp Chest','Resp Thoracic'],

        }

        self.channel_names =  ['EEG F3', 'EEG 22', 'EEG M1', 'EEG 28', 'EEG 24', 'EEG Cz-O1', 'C-Flow', 'EEG 31', 'LLeg', 'Abdominal', 'EEG 29', 
                               'EEG Chest', 'Fp2', 'EMG LLEG-LLEG2', 'EOG LOC-M2', 'OZ', 'DC8', 'EKG', 'C-Pressure', 'EMG Chin1-Chin2', 'EEG 27', 
                               'Resp Abdominal', 'FPZ', 'P3', 'Pressure', 'EEG EKG2', 'EEG Chin3-Chin2', 'Resp PTAF', 'Resp Airflow', 'Resp Rate', 
                               'Position', 'EEG Press', 'EEG ROC-M1', 'EKG2', 'EEG Chin1-Chin3', 'EEG 32', 'PTAF', 'OSAT', 'EMG LAT1-LAT2', 
                               'Snore_DR', 'EEG 25', 'Capno', 'EMG Chin3-Chin2', 'EMG Chin2-Chin1', 'Rate', 'Tidal Vol', 'Snore', 'EEG M2', 
                               'EEG Chin2', 'EEG 40', 'EMG CHIN1-CHIN2', 'F4', 'PPG', 'M2', 'EEG O2', 'EEG O1', 'EMG RLEG-RLEG2', 'Pleth', 
                               'EEG O1-M2', 'EEG F4-M1', 'EEG CZ-O1', 'Chin3', 'ECG LA-RA', 'Fp1', 'EEG C4', 'EEG E2', '39', 'F8', 'OSat', 
                               'EEG RLeg2', 'EEG EKG-RLeg', 'EEG Spare', 'EEG RLeg1', 'EMG LLEG-RLEG', 'EEG LOC-M2', 'EMG Chin1-Chin3', 'DC3', 
                               'EEG ROC-M2', 'P4', 'EEG EKG1', 'Resp Flow', 'XFlow', 'EOG ROC-M1', 'Flow_DR', 'EEG Chin1-Chin2', 'EEG 23', 'M1', 
                               '40', 'Chin2', 'O2', 'EEG E1', 'LOC', 'T6', 'EEG O2-M1', 'EEG Abd', 'T3', 'EEG C3', 'RLeg', 'F3', 
                               'EMG LLeg-RLeg', 'EEG 26', 'EEG 20', 'EEG EKG2-EKG', 'CZ', 'TcCO2', 'EEG F4', 'C-flow', 'ECG EKG2-EKG', 'EEG 30', 
                               'DC4', 'EEG 33', 'C4', 'Resp Airflow+-Re', 'EEG Chin1', 'EEG Snore', 'EMG CHIN1-CHIN3', 'Chin1', 'EMG RAT1-RAT2', 
                               'EEG LLeg1', 'EEG C3-M2', 'T4', 'Resp Chest', 'EEG F4-M2', 'SNORE_DR', 'EEG C4-M1', 'Resp Thoracic', 'EMG RLEG+-RLEG-', 
                               'PZ', 'Airflow', 'SpO2', 'EEG LLeg2', 'Resp FLOW-Ref', 'EEG Chin3', 'FZ', 'EEG Therm', 'EEG C4-M2', 'ROC', 'Thoracic', 
                               'F7', 'EtCO2', 'ECG ECGL-ECGR', 'T5', 'EEG F3-M2', 'Resp Abdomen', 'EMG LLEG+-LLEG-', 'PR', 'C3', 'O1', '38', 'EEG 21']
        
        
        self.channel_types = {'analog': ['Resp Abdominal', 'EEG 21', 'EEG C4', 'T4', 'C-Flow', 'EEG 23', 'EEG 28', 'EOG LOC-M2', 'EEG M1', 'EEG 33', 
                                         'CZ', 'EtCO2', 'F8', 'EEG EKG2', 'Resp Rate', 'T6', 'FZ', 'EEG EKG1', 'EEG F4-M2', 'C-Pressure', 'EEG E1', 
                                         'Fp2', 'EEG LLeg1', 'EEG O2', 'LLeg', 'EMG CHIN1-CHIN2', 'EEG F3-M2', 'EEG 24', 'EMG Chin3-Chin2', 'TcCO2', 
                                         'Thoracic', 'P4', 'EEG O1-M2', 'FPZ', 'Chin3', 'Resp Thoracic', 'F7', 'T3', 'EEG 31', 'Chin1', 'O1', 
                                         'EEG Chin3-Chin2', 'C-flow', 'Resp Abdomen', 'EEG O1', 'ECG EKG2-EKG', 'C3', 'EEG F4-M1', 'EEG ROC-M2', 'P3', 
                                         'EEG F4', 'Tidal Vol', 'EEG C3-M2', 'EMG RLEG-RLEG2', 'EEG E2', 'EEG LOC-M2', 'Resp Airflow', 'EEG LLeg2', 
                                         'EEG CZ-O1', 'EMG CHIN1-CHIN3', 'F3', 'EEG C4-M1', 'SNORE_DR', '38', 'F4', 'Resp PTAF', 'EEG Chin1-Chin3', 
                                         'ROC', 'EKG', 'EEG EKG2-EKG', '40', 'XFlow', 'DC8', 'EMG Chin1-Chin2', 'EEG Chin1', 'T5', 'Resp Flow', 
                                         'EMG RLEG+-RLEG-', 'EEG 29', 'EKG2', 'EEG 32', 'OZ', 'EEG Snore', 'EEG Chin3', 'EEG 20', 'EOG ROC-M1', 
                                         'Flow_DR', 'M1', '39', 'Airflow', 'PTAF', 'EEG Cz-O1', 'EEG 26', 'LOC', 'Resp Chest', 'EEG 22', 'Fp1', 
                                         'EEG 40', 'EEG Chin2', 'Pressure', 'EEG 25', 'Resp FLOW-Ref', 'EMG Chin2-Chin1', 'M2', 'EMG Chin1-Chin3', 
                                         'RLeg', 'EEG ROC-M1', 'Resp Airflow+-Re', 'EEG O2-M1', 'PZ', 'Abdominal', 'EEG EKG-RLeg', 'EEG Chin1-Chin2', 
                                         'EEG RLeg2', 'EMG RAT1-RAT2', 'EEG F3', 'Capno', 'Chin2', 'EMG LLeg-RLeg', 'O2', 'EEG C4-M2', 'ECG ECGL-ECGR', 
                                         'EMG LLEG+-LLEG-', 'EEG Press', 'EEG Therm', 'C4', 'EEG Spare', 'EMG LAT1-LAT2', 'EEG 30', 'EMG LLEG-LLEG2', 
                                         'Snore', 'ECG LA-RA', 'EEG M2', 'EEG Chest', 'EEG RLeg1', 'EEG 27', 'EEG Abd', 'EMG LLEG-RLEG', 'Snore_DR', 
                                         'EEG C3'], 
                              'digital': ['SpO2', 'Position', 'Pleth', 'DC3', 'OSat', 'PR', 'DC4', 'PPG', 'OSAT', 'Rate']}
        
        # Not known: 'EEG 21', 'EEG 33','EEG 30', 'EEG 20','EEG 26', 'EEG 23','EEG 40','EEG 25','EEG 27','EEG 32','EEG 22','EEG 29','EEG 31','EEG 24','EEG 28',

        self.channel_groups = {'eeg_eog': ['Fp2', 'EEG M1','EEG F3','EOG LOC-M2','EEG O2-M1', 'EOG ROC-M1', 'EEG Cz-O1', 'EEG C3-M2', 'EEG O1-M2', 
                                           'EEG F4-M1', 'EEG C4-M1', 'EEG F3-M2', 'EEG CZ-O1','OZ', 'FPZ', 'P3', 'EEG ROC-M1', 'EEG M2','F4', 'M2',
                                           'EEG O2', 'EEG O1','Fp1','EEG C4','EEG E2', 'F8', 'EEG LOC-M2','EEG ROC-M2', 'P4','M1', 'O2', 'EEG E1',
                                           'LOC', 'T6','T3', 'EEG C3','F3', 'CZ',  'EEG F4','C4','T4', 'EEG F4-M2','PZ', 'FZ','EEG C4-M2','ROC', 
                                           'F7', 'T5','C3', 'O1'],
                                'emg': ['EMG CHIN1-CHIN2', 'EMG LLEG+-LLEG-', 'EMG RLEG+-RLEG-', 'EMG Chin1-Chin2', 'EMG LLeg-RLeg','LLeg','EMG LLEG-LLEG2',
                                        'Chin2','Chin1','EMG RAT1-RAT2','EMG LAT1-LAT2', 'EMG Chin3-Chin2', 'EMG Chin2-Chin1','EMG RLEG-RLEG2','Chin3',
                                        'EMG LLEG-RLEG','EMG Chin1-Chin3','RLeg','EMG CHIN1-CHIN3','EEG Chin2', 'EEG RLeg2','EEG Chin1-Chin3','EEG RLeg1',
                                        'EEG Chin1-Chin2','EEG LLeg1','EEG LLeg2','EEG Chin3','EEG Chin1',],
                                'ecg': ['ECG EKG2-EKG','ECG LA-RA','EKG', 'EEG EKG2', 'EKG2', 'ECG ECGL-ECGR','EEG EKG-RLeg','EEG EKG1','EEG EKG2-EKG',],
                                'thoraco_abdo_resp': ['Resp Abdomen', 'Resp Chest','Resp Flow', 'Resp Airflow','Resp Thoracic', 'Resp Abdominal','Abdominal','Resp PTAF',  'Thoracic', 'Resp Airflow+-Re', 
                                                      'EEG Chest','Resp Airflow','EEG Abd',],
                                'snoring': ['Snore', 'Snore_DR','EEG Snore',]
                                }
        
        
        self.file_extensions = {
                                'psg_ext': '*.edf',
                                'ann_ext': '*.tsv'
                                }
        

    def dataset_paths(self) -> Tuple[str, str]:
        """
        NCHSDB dataset paths.
        """
        data_dir = "NCHSDB - NCH Sleep DataBank/sleep_data"
        ann_dir = "NCHSDB - NCH Sleep DataBank/sleep_data"
        return data_dir, ann_dir
        
    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        function to parse the annotation file of the dataset into sleep stage events with start and duration

        """

        ann_stage_events = []
        ann_df = pd.read_csv(ann_fname,header = 0, sep='\t')
        start_time_label = None

        epoch_duration = 30  # NCHSDB uses 30-second epochs

        for i,row in ann_df.iterrows():
            event = row['description']
            if 'Sleep stage' in event:
                if event not in self.ann2label:
                    print(event)
                    raise Exception
                start = Decimal(str(row['onset']))
                if start_time_label == None:
                    start_time_label = start
                duration =  row['duration']

                if i == len(ann_df)-1 and duration != epoch_duration:
                    break
                
                start = start - start_time_label
                if '4480_5926.tsv' in ann_fname:
                    if start==2250:
                         ann_stage_events.append({'Stage':'Sleep stage ?','Start':2190,'Duration':60})  
                ann_stage_events.append({'Stage': event,
                                        'Start': start,
                                        'Duration': duration})
                
        if ann_stage_events[-1]['Duration'] != epoch_duration:
            ann_stage_events.pop()

        return ann_stage_events,start_time_label
    
    def align_front(self, logger, start_seconds, psg_fname, ann_fname: str,signal: np.ndarray, labels, fs
                  ) -> Tuple[np.ndarray, np.ndarray]:

        if not (float(start_seconds*Decimal(str(fs)))).is_integer():
            print(fs)
            print(start_seconds%(1/Decimal(str(fs))))
            raise Exception("Annotations start at timestamp outside of sample rate")

        if start_seconds != 0:
            logger.info(f"Labeling started {start_seconds/60:.2f} min after signal start, signal will be shortened at the front to match")
            signal = signal[int(start_seconds*Decimal(str(fs))):] 

        return True, signal, labels