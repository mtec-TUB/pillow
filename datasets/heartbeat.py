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


        self.inter_dataset_mapping = {
        }
        

        self.channel_names = ['0-1v DC', 'Abdomen', 'Abdomen_EG', 'Activity', 'Activity_EG', 'Battery', 'CPAP_PDS', 'Differential Pre', 'EKG', 'EKG_EG', 'Elevation', 
                              'Flattening', 'Flow', 'Flow_EG', 'Gravity X', 'Gravity Y', 'Nasal', 'Nasal_EG', 'Phase', 'Position', 'Position_EG', 'Pulse', 'Pulse_EG', 
                              'RD-Pleth', 'RD-Pleth_EG', 'RD-Quality', 'RMI', 'RR', 'Snore', 'Snore_EG', 'SpO2', 'SpO2-BB', 'SpO2-Quality', 'SpO2-Quality_EG', 'SpO2_EG', 
                              'Thermistor', 'Thermistor_EG', 'Thorax', 'Thorax_EG', 'Tidal Volume', 'XFlow', 'XFlow_EG', 'XSum', 'XSum_EG']
        
        
        self.channel_types = {'analog': ['Position', 'EKG', 'Gravity X', 'Battery', 'RR', 'Tidal Volume', 'Abdomen', 'Phase', 'XSum', 'Nasal', 'Flow', 'Thorax', 
                                         'Thermistor', '0-1v DC', 'Differential Pre', 'Snore','RMI', 'Gravity Y', 'Elevation', 'Activity', 'XFlow'], 
                            'digital': ['SpO2', 'SpO2-Quality_EG', 'CPAP_PDS', 'Thermistor_EG', 'Pulse_EG', 'RD-Pleth', 'Flow_EG', 'SpO2_EG', 'SpO2-BB', 'XSum_EG', 
                                        'SpO2-Quality', 'XFlow_EG', 'Nasal_EG', 'Position_EG', 'Flattening', 'Pulse', 'Activity_EG', 'EKG_EG', 'Abdomen_EG', 'RD-Quality', 
                                        'RD-Pleth_EG', 'Snore_EG',  'Thorax_EG']} #### NOCHMAL ÜBERLEGEN wegen EG channels?!
    
        
        
        self.channel_groups = {
            'eeg_eog': [],
            'emg': [ ],
            'ecg': ['EKG'],
            'thoraco_abdo_resp': ['Flow', ],
            'nasal_pressure': ['Nasal','Thorax','Abdomen', ],
            'snoring': ['Snore']
        }        

        
        self.file_extensions = {
            'psg_ext': '**/*.edf',
            'ann_ext': '*-nsrr.xml' # not available for this dataset, just for consistency
        }

    def dataset_paths(self) -> Tuple[str, str]:
        """Dataset paths for HEARTBEAT dataset"""
        return [
            os.path.join(self.dataset_name, "polysomnography", "edfs"),
            self.dataset_name   # not available for this dataset, just for consistency
        ]