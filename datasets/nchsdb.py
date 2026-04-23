"""
NCHSDB - NCH Sleep DataBank
"""
import os
import pandas as pd
from decimal import Decimal
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("NCHSDB")
class NCHSDB(BaseDataset):
    """NCHSDB - NCH Sleep DataBank dataset."""
    
    def __init__(self):
        super().__init__("NCHSDB","NCHSDB - NCH Sleep DataBank")

    def get_signal_data(self, logger, filepath, channel):
        signal_data = super().get_signal_data(logger, filepath, channel)
        if '5053_2167.edf' in filepath and not signal_data["sampling_rate"].is_integer():
            logger.warning(f"Found non-integer sampling rate {signal_data['sampling_rate']} in file {os.path.basename(filepath)}. Rounding to nearest integer.")
            signal_data["sampling_rate"] = round(signal_data["sampling_rate"])
        return signal_data
    
    def _setup_dataset_config(self):
        self.ann2label = {
                'Sleep stage W': "W",
                'Sleep stage N1': "N1",
                'Sleep stage N2': "N2",
                'Sleep stage N3': "N3",
                'Sleep stage R': "REM",
                'Sleep stage ?': "UNK",
                'Sleep stage 2': "N2",
                'Sleep stage 1': "N1",
                'Sleep stage 3' : "N3",
        }

        self.intra_dataset_mapping = {'Abdomen': ['Abdominal','EEG Abd'],
                              'Resp Abdomen': ['Resp Abdominal', 'Resp Abdomen'],
                              'C-flow': ['C-flow','C-Flow'],
                              'Cz-O1': ['EEG Cz-O1','EEG CZ-O1'],
                              'ECG2': ['EEG EKG2','EKG2'],
                              'ECG2-ECG': ['ECG EKG2-EKG','EEG EKG2-EKG'],
                              'EMG LLeg-RLeg': ['EMG LLEG-RLEG','EMG LLeg-RLeg'],
                              'EMG LLEG+-LLEG-': ['EMG LLEG+-LLEG-', 'EMG LLEG-LLEG2'],
                              'EMG RLEG+-RLEG-': ['EMG RLEG+-RLEG-','EMG RLEG-RLEG2'],
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
                              'Chin3-Chin2' : ['EEG Chin3-Chin2','EMG Chin3-Chin2'],
                              'Chin1-Chin2': ['EMG Chin1-Chin2','EMG CHIN1-CHIN2','EEG Chin1-Chin2'],
                              'Chin1-Chin3': ['EEG Chin1-Chin3','EMG Chin1-Chin3','EMG CHIN1-CHIN3'],
                              'Snore':['EEG Snore','Snore'],
                              'SnoreDR': ['Snore_DR','SNORE_DR'],
                              'SpO2': ['OSAT',"SpO2"],  # channel "OSat" is not listed because only one file has this channel and it also contains SpO2, so we can use SpO2 for mapping
                              'Resp Chest': ['Resp Chest','Resp Thoracic'],
        }

        # chose mostly channels that appear often (more than 28% of files) in dataset (doi: 10.1038/s41597-022-01545-6)
        self.inter_dataset_mapping = {
            "Resp Abdomen": self.Mapping(self.TTRef.ABDOMINAL, None),
            "Cz-O1": self.Mapping(self.TTRef.Cz, self.TTRef.O1),
            "ECG2-ECG": self.Mapping(self.TTRef.ECG, None),
            "EMG LLeg-RLeg": self.Mapping(self.TTRef.EMG_LLEG, self.TTRef.EMG_RLEG),
            "LOC-M2": self.Mapping(self.TTRef.EL, self.TTRef.RPA),
            "ROC-M1": self.Mapping(self.TTRef.ER, self.TTRef.LPA),
            "Chin1-Chin2": self.Mapping(self.TTRef.EMG_LCHIN, self.TTRef.EMG_RCHIN),    # not sure
            "Snore": self.Mapping(self.TTRef.SNORE, None),
            "SpO2": self.Mapping(self.TTRef.SPO2, None),
            "EEG C3-M2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "EEG C4-M1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "EEG O1-M2": self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            "EEG O2-M1": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "EEG F3-M2": self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            "EEG F4-M1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            "C-flow": self.Mapping(self.TTRef.CPAP, None),
            "Resp Airflow": self.Mapping(self.TTRef.AIRFLOW, None),
            "Resp Chest": self.Mapping(self.TTRef.THORACIC, None),
            "EMG LLEG+-LLEG-": self.Mapping(self.TTRef.EMG_LLEG, None),
            "EMG RLEG+-RLEG-": self.Mapping(self.TTRef.EMG_RLEG, None),
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
                                        'EEG Chin1-Chin2','EEG LLeg1','EEG LLeg2','EEG Chin3','EEG Chin1','EEG Chin3-Chin2'],
                                'ecg': ['ECG EKG2-EKG','ECG LA-RA','EKG', 'EEG EKG2', 'EKG2', 'ECG ECGL-ECGR','EEG EKG-RLeg','EEG EKG1','EEG EKG2-EKG',],
                                'thoraco_abdo_resp': ['Resp Abdomen', 'Resp Chest','Resp Flow', 'Resp Airflow','Resp Thoracic', 'Resp Abdominal','Abdominal','Resp PTAF',  'Thoracic', 'Resp Airflow+-Re', 
                                                      'EEG Chest','Resp Airflow','EEG Abd',],
                                'snoring': ['Snore', 'Snore_DR','EEG Snore',]
                                }
        
        
        self.file_extensions = {
                                'psg_ext': '*.edf',
                                'ann_ext': '*.tsv'
                                }
        

    def dataset_paths(self):
        return [
            'sleep_data',
            'sleep_data'
        ]
        
    def ann_parse(self, ann_fname: str):
        """
        function to parse the annotation file of the dataset into sleep stage events with start and duration

        """

        ann_stage_events = []
        ann_df = pd.read_csv(ann_fname,header = 0, sep='\t')
        start_time_label = None
        lights_off, lights_on = [], []

        epoch_duration = 30  # NCHSDB uses 30-second epochs

        for i,row in ann_df.iterrows():
            event = row['description']
            if re.search(r'(?i)(?:Lights Off)(?=\s|$)', event):
                lights_off.append(Decimal(str(row['onset'])))
            elif re.search(r'(?i)(?:Lights On)(?=\s|$)', event):
                lights_on.append(Decimal(str(row['onset'])))
            elif 'Sleep stage' in event:
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
                if '4480_5926.tsv' in ann_fname and start==2250:
                    ann_stage_events.append({'Stage':'Sleep stage ?','Start':2190,'Duration':60})  
                ann_stage_events.append({'Stage': event,
                                        'Start': start,
                                        'Duration': duration})
                
        if len(ann_stage_events) == 0:
            return ann_stage_events, None, None, None
                
        if ann_stage_events[-1]['Duration'] < epoch_duration:
            ann_stage_events.pop() # drop last epoch as it corresponds to the last unfilled epoch of signal data which is dropped aswell
        
        if len(lights_off) == 1:
            lights_off = float(lights_off[0])
        elif len(lights_off) == 0:
            lights_off = None
        elif any(name in ann_fname for name in ['7681_21013', '17950_18358', '10798_556', '5299_9865']):
            lights_off = float(lights_off[-1])  # manually checked and in these files the last lights off event seems to be the correct one
        elif len(lights_off) > 1:  
            lights_off = float(lights_off[0])
        else:
            raise Exception(f"Found {len(lights_off)} lights off events in annotation file {os.path.basename(ann_fname)}")
        
        if len(lights_on) == 1:
            lights_on = float(lights_on[0])
        elif len(lights_on) == 0:
            lights_on = None
        elif lights_on[0] == lights_on[1]:  # if there are multiple lights on events but they have the same timestamp, take the first one
            lights_on = float(lights_on[0])
        elif any(name in ann_fname for name in ['10480_2032', '8608_18625', '11341_10369']):
            lights_on = float(lights_on[0])     # manually checked and in these files the first lights on event seems to be the correct one
        elif len(lights_on) > 1:
            lights_on = float(lights_on[-1])
        else:
            raise Exception(f"Found {len(lights_on)} lights on events in annotation file {os.path.basename(ann_fname)}")

        return ann_stage_events, float(start_time_label), lights_off, lights_on
    
    def align_front(self, logger, alignment, pad_values, epoch_duration,delay_sec,signal: np.ndarray, labels, fs
                  ):

        # if not (float(delay_sec*Decimal(str(fs)))).is_integer():
        #     print(fs)
        #     print(delay_sec%(1/Decimal(str(fs))))
        #     raise Exception("Annotations start at timestamp outside of sample rate")

        return self.base_align_front(logger, delay_sec, alignment, pad_values, epoch_duration, signal, labels,fs)