import numpy as np
from pyedflib import EdfReader
from typing import Dict, List, Tuple
from datetime import datetime, date
from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("PHYSIO2026")
class Physio2026(BaseDataset):
    """Physio2026 (PhysioNet Challenge 2026) dataset."""
    
    def __init__(self):
        super().__init__("PHYSIO2026","Physio2026 - PhysioNet Challenge 2026", keep_folder_structure = False)

    def _setup_dataset_config(self):
        self.ann2label = {
                9: "UNK", # Stage 'L' in BDSP
                0: "UNK", # corresponds to empty rows in BDSP
                5: "W",
                3: "N1",
                2: "N2",
                1: "N3",
                4: "REM",
                }
        
        self.intra_dataset_mapping = {
            'ECG': ['EKG', 'ECG', 'EKG-E1','ECG-LA'],
            'SpO2': ['SaO2', 'SpO2', 'SPO2'],
            'C3': ['C3', 'C3-AVG'],
            'C4': ['C4', 'C4-AVG'],
            'O1': ['O1', 'O1-AVG'],
            'O2': ['O2', 'O2-AVG'],
            'F3': ['F3', 'F3-AVG'],
            'F4': ['F4', 'F4-AVG'],
            'E1': ['E1', 'E1-AVG'],
            'E2': ['E2', 'E2-AVG'],
            'Chin1-Chin2': ['CHIN1-CHIN2', 'Chin1-Chin2'],
            'Chin1-Chin3': ['CHIN1-CHIN3', 'Chin1-Chin3'],
            'Chin2': ['CHIN2', 'Chin2','ChinR'],
            'Chin1': ['CHIN','CHIN1', 'ChinL'],
            'Right Leg': ['Right Leg', 'R LEG','R EMG','RAT', 'RAT-E1'],
            'Left Leg': ['Left Leg', 'L LEG','L EMG','LAT', 'LAT-E1'],
            'Thorax': ['THORACIC', 'Thorax', 'THORAX','Chest', 'CHEST', 'CHEST-E1'],
            'Abdomen': ['ABDOMINAL', 'Abdomen', 'ABDOMEN', 'ABD', 'ABD-E1'],
            'Nasal Pressure': ['NPT', 'Nasal Pressure', 'PTAF'],
            'Airflow': ['AIRFLOW', 'AirFlow', 'AIRFLOW-E1','Flow_DR'],
            'Thermistor': ['THERM', 'Thermistor', 'THERMISTOR'],
            'CPAP Pressure': ['C PRESS', 'CPAP Pressure', 'CPAP Press', 'Cpress',
                            'CPAP PRESSURE', 'CPRES',],
            'CPAP Flow': ['C-FLOW', 'CFLOW', 'CFlow'],
        }

        self.inter_dataset_mapping = {
            'ECG': self.Mapping(self.TTRef.ECG, None),
            'SpO2': self.Mapping(self.TTRef.SPO2, None),
            'C3': self.Mapping(self.TTRef.C3, None),
            'C3-M2': self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            'C4': self.Mapping(self.TTRef.C4, None),
            'C4-M1': self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            'O1': self.Mapping(self.TTRef.O1, None),
            'O1-M2': self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            'O2': self.Mapping(self.TTRef.O2, None),
            'O2-M1': self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            'F3': self.Mapping(self.TTRef.F3, None),
            'F3-M2': self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            'F4': self.Mapping(self.TTRef.F4, None),
            'F4-M1': self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            'E1': self.Mapping(self.TTRef.EL, None),
            'E1-M2': self.Mapping(self.TTRef.EL, self.TTRef.RPA),
            'E1-M1': self.Mapping(self.TTRef.EL, self.TTRef.LPA),
            'E2': self.Mapping(self.TTRef.ER, None),
            'E2-M1': self.Mapping(self.TTRef.ER, self.TTRef.LPA),
            'E2-M2': self.Mapping(self.TTRef.ER, self.TTRef.RPA),
            'M1': self.Mapping(self.TTRef.LPA, None),
            'M2': self.Mapping(self.TTRef.RPA, None),
            'Chin1-Chin2': self.Mapping(self.TTRef.EMG_LCHIN, self.TTRef.EMG_RCHIN),
            'Chin1': self.Mapping(self.TTRef.EMG_LCHIN, None),
            'Chin2': self.Mapping(self.TTRef.EMG_RCHIN, None),
            'Right Leg': self.Mapping(self.TTRef.EMG_RLEG, None),
            'Left Leg': self.Mapping(self.TTRef.EMG_LLEG, None),
            'Thorax': self.Mapping(self.TTRef.THORACIC, None),
            'Abdomen': self.Mapping(self.TTRef.ABDOMINAL, None),
            'Airflow': self.Mapping(self.TTRef.AIRFLOW, None),
            'CPAP Pressure': self.Mapping(self.TTRef.CPAP, None),
        }
        
        self.channel_names = ['F3-M2', 'F4-M1', 'C4-M1', 'O2-M1', 'E1', 'E2', 'CHIN', 'EKG', 'LAT', 'RAT', 'PTAF', 'THORACIC', 'ABDOMINAL', 'SaO2', 
                              'THERM', 'C PRESS', 'C3-M2', 'O1-M2', 'NPT', 'CHEST', 'C-FLOW', 'Right Leg', 'Left Leg', 'M1', 'C3', 'M2', 'C4', 'O2', 
                              'O1', 'Nasal Pressure', 'ChinA', 'ChinR', 'ChinL', 'F3', 'Thorax', 'Abdomen', 'Thermistor', 'F4', 'CFLOW', 
                              'CPAP Pressure', 'SpO2', 'Flow_DR', 'CPAP Press', 'Cpress', 'CPAP Pressure 1', 'Thermistor 2', 'R EMG', 'L EMG', 
                              'CPAP PRESSURE', 'Pressure', 'E1-M2', 'E2-M1', 'CHIN1-CHIN2', 'AIRFLOW', 'ABD', 'CHIN2', 'RLEG+', 'RLEG-', 'LLEG+', 
                              'LLEG-', 'AirFlow', 'Airflow2', 'CFlow', 'ECG-LA', 'ECG-RA', 'ECG-LL', 'ECG-V1', 'ECG-V2', 'E2-M2', 'Chin1-Chin2', 
                              'CPRES', 'CHIN1-CHIN3', 'F3-AVG', 'F4-AVG', 'C3-AVG', 'C4-AVG', 'O1-AVG', 'O2-AVG', 'E1-AVG', 'E2-AVG', 'Chin1-Chin3', 
                              'CHIN1', 'Flow', 'Chest', 'Chin2', 'LAT-E1', 'RAT-E1', 'AIRFLOW-E1', 'CHEST-E1', 'ABD-E1', 'EKG-E1', 'EMG', 'R LEG', 
                              'L LEG', 'ABDOMEN', 'THORAX', 'THERMISTOR', 'SPO2', 'ECG', 'E1-M1']
        
        self.channel_types = {'analog': ['F3-M2', 'F4-M1', 'C4-M1', 'O2-M1', 'E1', 'E2', 'CHIN', 'EKG', 'LAT', 'RAT', 'PTAF', 'THORACIC', 'ABDOMINAL', 'SaO2', 'THERM', 
                                'C PRESS', 'C3-M2', 'O1-M2', 'NPT', 'CHEST', 'C-FLOW', 'Right Leg', 'Left Leg', 'M1', 'C3', 'M2', 'C4', 'O2', 'O1', 'Nasal Pressure',
                                'ChinA', 'ChinR', 'ChinL', 'F3', 'Thorax', 'Abdomen', 'Thermistor', 'F4', 'CFLOW', 'CPAP Pressure', 'Flow_DR', 'CPAP Press', 'Cpress', 
                                'CPAP Pressure 1', 'Thermistor 2', 'R EMG', 'L EMG', 'CPAP PRESSURE', 'Pressure', 'E1-M2', 'E2-M1', 'CHIN1-CHIN2', 'AIRFLOW', 'ABD', 
                                'CHIN2', 'RLEG+', 'RLEG-', 'LLEG+', 'LLEG-', 'AirFlow', 'Airflow2', 'CFlow', 'ECG-LA', 'ECG-RA', 'ECG-LL', 'ECG-V1', 'ECG-V2', 'E2-M2', 
                                'Chin1-Chin2', 'CPRES', 'CHIN1-CHIN3', 'F3-AVG', 'F4-AVG', 'C3-AVG', 'C4-AVG', 'O1-AVG', 'O2-AVG', 'E1-AVG', 'E2-AVG', 'Chin1-Chin3', 
                                'CHIN1', 'Chest', 'Chin2', 'LAT-E1', 'RAT-E1', 'AIRFLOW-E1', 'CHEST-E1', 'ABD-E1', 'EKG-E1', 'EMG', 'R LEG', 'L LEG', 'ABDOMEN', 
                                'THORAX', 'THERMISTOR', 'ECG', 'E1-M1'], 
                            'digital': ['SpO2', 'Flow', 'SPO2']}
        
        
        self.channel_groups = {
            'eeg_eog': ['F3-M2', 'F4-M1', 'C4-M1', 'O2-M1', 'E1', 'E2','C3-M2', 'O1-M2','M1', 'C3', 'M2', 'C4', 'O2', 'F3','F4','E1-M2','E2-M1',
                              'O1', 'E2-M2', 'F3-AVG', 'F4-AVG', 'C3-AVG', 'C4-AVG', 'O1-AVG', 'O2-AVG', 'E1-AVG', 'E2-AVG','E1-M1'],
            'emg': ['CHIN', 'LAT', 'RAT','Right Leg', 'Left Leg','ChinA', 'ChinR', 'ChinL', 'R EMG', 'L EMG', 'CHIN1-CHIN2', 'CHIN2', 'RLEG+','RLEG-', 'LLEG+', 
                    'LLEG-', 'Chin1-Chin2', 'CHIN1-CHIN3', 'Chin1-Chin3', 'CHIN1','Chin2', 'R LEG', 'L LEG', 'EMG', 'LAT-E1', 'RAT-E1',],
            'ecg': ['EKG', 'ECG-LA', 'ECG-RA', 'ECG-LL', 'ECG-V1', 'ECG-V2','EKG-E1', 'ECG',],
            'nasal_pressure':  ['Nasal Pressure'],
            'thoraco_abdo_resp': ['THORACIC', 'ABDOMINAL','CHEST','Thorax', 'Abdomen','ABD','AirFlow', 'Airflow2', 'Chest',  'AIRFLOW-E1', 'CHEST-E1', 'ABD-E1',
                                    'ABDOMEN', 'THORAX']
        }
        
        self.file_extensions = {
            'psg_ext': '**/*.edf',
            'ann_ext': '**/*_expert_annotations.edf'
        }
    
    def dataset_paths(self):
        return [
            'physiological_data',
            'human_annotations'
        ]
    
    def ann_parse(self, ann_fname: str):
        """
        Parse Physio2026 annotation files.
        """
        ann_stage_events = []
        
        ann_f = EdfReader(ann_fname)
        try:
            ch_idx = ann_f.getSignalLabels().index('stage_expert')
        except ValueError:
            return ann_stage_events, None, None, None
        
        annotations = np.array(ann_f.readSignal(ch_idx), dtype=int)

        ann_stage_events = [{"Stage": annotations[idx], "Start": idx * 30, "Duration": 30} for idx in range(len(annotations))]

        return ann_stage_events, None, None, None
    

