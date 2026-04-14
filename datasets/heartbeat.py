import os
from typing import Tuple

from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("HEARTBEAT")
class HEARTBEAT(BaseDataset):
    """HEARTBEAT (Heart Biomarker Evaluation in Apnea Treatment) dataset"""
    
    def __init__(self):
        super().__init__("HEARTBEAT","HEARTBEAT - Heart Biomarker Evaluation in Apnea Treatment")
  
    def _setup_dataset_config(self):
        self.ann2label = {} # No sleep stage annotations available for this dataset

        self.intra_dataset_mapping = {
            "ECG": ['EKG','EKG_EG'],
            "SpO2": ['SpO2', 'SpO2_EG'],
            "SpO2-Quality": ['SpO2-Quality', 'SpO2-Quality_EG',],
            "Thermistor": ['Thermistor', 'Thermistor_EG'],
            "Abdomen": ['Abdomen', 'Abdomen_EG'],
            "Activity": ['Activity', 'Activity_EG'],
            "Thorax": ["Thorax", 'Thorax_EG'],
            "Nasal": ['Nasal', 'Nasal_EG'],
            "Position": ['Position', 'Position_EG'],
            "Pulse": ['Pulse', 'Pulse_EG'],
            "Flow": ['Flow', 'Flow_EG'],
            "Snore": ['Snore', 'Snore_EG'],
            "XSum": ['XSum', 'XSum_EG'],
            "XFlow": ['XFlow', 'XFlow_EG'],
            "RD-Pleth": ['RD-Pleth', 'RD-Pleth_EG'],
        }

        self.inter_dataset_mapping = {
            "Abdomen": self.Mapping(self.TTRef.ABDOMINAL, None),
            "Thorax": self.Mapping(self.TTRef.THORACIC, None),
            "ECG": self.Mapping(self.TTRef.ECG, None),
            "Snore": self.Mapping(self.TTRef.SNORE, None),
            "Position": self.Mapping(self.TTRef.POSITION, None),
            "SpO2": self.Mapping(self.TTRef.SPO2, None),
        }

        self.channel_names = ['0-1v DC', 'Abdomen', 'Abdomen_EG', 'Activity', 'Activity_EG', 'Battery', 'CPAP_PDS', 'Differential Pre', 'EKG', 'EKG_EG', 'Elevation', 
                              'Flattening', 'Flow', 'Flow_EG', 'Gravity X', 'Gravity Y', 'Nasal', 'Nasal_EG', 'Phase', 'Position', 'Position_EG', 'Pulse', 'Pulse_EG', 
                              'RD-Pleth', 'RD-Pleth_EG', 'RD-Quality', 'RMI', 'RR', 'Snore', 'Snore_EG', 'SpO2', 'SpO2-BB', 'SpO2-Quality', 'SpO2-Quality_EG', 'SpO2_EG', 
                              'Thermistor', 'Thermistor_EG', 'Thorax', 'Thorax_EG', 'Tidal Volume', 'XFlow', 'XFlow_EG', 'XSum', 'XSum_EG']
        
        
        self.channel_types = {'analog': ['0-1v DC', 'Abdomen', 'Abdomen_EG', 'Activity', 'Activity_EG', 'Battery', 'CPAP_PDS', 'Differential Pre', 'EKG', 'EKG_EG', 'Elevation', 
                                         'Flattening', 'Flow', 'Flow_EG', 'Gravity X', 'Gravity Y', 'Nasal', 'Nasal_EG', 'Phase', 'Position', 'Position_EG', 'Pulse', 
                                         'RD-Pleth', 'RMI', 'RR', 'Snore', 'Thermistor', 'Thermistor_EG', 'Thorax', 'Thorax_EG', 'Tidal Volume', 'XFlow', 'XFlow_EG', 'XSum', 
                                         'XSum_EG'], 
                              'digital': ['Pulse_EG', 'RD-Pleth_EG', 'RD-Quality', 'Snore_EG', 'SpO2', 'SpO2-BB', 'SpO2-Quality', 'SpO2-Quality_EG', 'SpO2_EG']}
    
        
        self.channel_groups = {
            'eeg_eog': [],
            'emg': [ ],
            'ecg': ['EKG','EKG_EG'],
            'thoraco_abdo_resp': ['Flow', 'Flow_EG', 'Thorax','Thorax_EG','Abdomen','Abdomen_EG',],
            'nasal_pressure': ['Nasal','Nasal_EG'],
            'snoring': ['Snore','Snore_EG',]
        }        

        
        self.file_extensions = {
            'psg_ext': '**/*.edf',
            'ann_ext': '*-nsrr.xml' # not available for this dataset, just for consistency
        }

    def dataset_paths(self):
        """Dataset paths for HEARTBEAT dataset"""
        return [
            os.path.join("polysomnography", "edfs"),
            ''   # not available for this dataset, just for consistency
        ]