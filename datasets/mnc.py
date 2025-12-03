"""
MNC - Mignot Nature Communications
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import xml.etree.ElementTree as ET

from .base import BaseDataset
from .registry import register_dataset


@register_dataset("MNC")
class MNC(BaseDataset):
    """MNC - Mignot Nature Communications"""
    
    def __init__(self):
        super().__init__("MNC","MNC - Mignot Nature Communications")

    def _setup_dataset_config(self):
        self.ann2label = {
                "wake": 0,
                "NREM1": 1,
                "NREM2": 2,
                "NREM3": 3,
                "NREM4": 3,
                "REM": 4,
                "unscored": 6,
                "9": 6,  # unknown?
                "8": 6, # unknown?
                " ": 6,  # empty lines
                "NaN": 6,
        }
        
        
        self.channel_names = ['lleg1_2', 'ECG1', 'F3', 'spo2', 'thorax', 'ECG', 'abdomen', 'cs_EEG', 'rleg', 'cs_EMG', 'cs_ECG', 'nas_pres', 'lleg', 'position', 'cs_ROC', 
                    'E1', 'O1', 'cchin_l', 'chin', 'cs_LOC', 'ECG3', 'C4', 'sum', 'C3', 'snore', 'pulse', 'F4', 'etco2', 'O2', 'rleg1_2', 'flow', 'ECG2', 'ECG1_2',
                    'therm', 'E2', 'lleg_r',  'leak',  'rleg2',  'lleg1', 'cchin', 'rchin', 'PTT', 'pap_pres', 'pap_flow', 'lchin', 'rleg1', 'tvol',
]
        
        
        self.channel_types = {'analog': ['lleg1_2', 'ECG1', 'F3', 'spo2', 'thorax', 'ECG', 'abdomen', 'cs_EEG', 'rleg', 'cs_EMG', 'cs_ECG', 'nas_pres', 'lleg', 'cs_ROC',
                               'E1', 'O1', 'cchin_l', 'chin', 'cs_LOC', 'ECG3', 'C4', 'sum', 'C3', 'snore', 'F4', 'etco2', 'O2', 'rleg1_2', 'flow', 'ECG2', 
                               'ECG1_2', 'therm', 'E2',
                               'cs_EMG', 'F4', 'cs_ECG', 'O2', 'C4', 'cs_ROC', 'E1', 'cs_LOC', 'C3', 'chin', 'ECG', 'O1', 'E2', 'cs_EEG',
                               'lleg_r', 'C4', 'leak', 'cs_LOC', 'spo2', 'rleg2', 'E1', 'lleg1', 'pulse', 'etco2', 'E2', 'therm', 'abdomen', 'flow', 
                               'nas_pres', 'thorax', 'snore', 'cchin', 'cs_ROC', 'rchin', 'PTT', 'O2', 'cs_ECG', 'cs_EMG', 'pap_pres', 'F4', 'pap_flow', 'lchin',
                               'position', 'C3', 'cs_EEG', 'rleg1', 'tvol', 'ECG', 'O1'], 
                    'digital': ['position', 'pulse']}
        
        
        self.channel_groups = {
            'eeg_eog': ['C4', 'cs_LOC', 'E1','E2', 'F3','O2','cs_ROC','cs_EEG', 'F4', 'C3', 'O1', 'O2'],
            'emg': ['rleg1_2','rleg2','rleg', 'lleg1','lleg_r','rchin','cchin', 'cs_EMG','lleg1_2', 'rleg1','lchin','lleg',  'cchin_l', 'chin'],
            'ecg': ['cs_ECG','ECG',  'ECG1', 'ECG3',   'ECG2', 'ECG1_2'],
            'thoraco_abdo_resp': ['therm', 'abdomen', 'flow',  'thorax'],
            'nasal_pressure': ['nas_pres',],
            'snoring': ['snore']
        }
        
        
        self.file_extensions = {
            'psg_ext': "**/*.edf",
            'ann_ext': "**/*.xml"
        }
    

    def dataset_paths(self) -> Tuple[str, str]:
        """
        MNC dataset paths.
        """
        data_dir = "MNC - Mignot Nature Communications/sleep_data"
        ann_dir = "MNC - Mignot Nature Communications/sleep_data"
        return data_dir, ann_dir

    def ann_parse(self, ann_fname: str, epoch_duration: Optional[int] = None) -> Tuple[List[Dict], datetime]:
        """
        function to parse the annotation file of the dataset into sleep stage events with start and duration
    
        """
        ann_stage_events = []
        ann_startdatetime = None

        # Parse XML
        ann_f = ET.parse(ann_fname)
        ann_root = ann_f.getroot()

        # Extract StartTime
        start_time = ann_root.findtext("StartTime")
        if start_time != ".":
            ann_startdatetime = datetime.strptime(start_time,"%H.%M.%S")

        # Iterate over Instances
        instances = ann_root.find("Instances").findall("Instance")

        for instance in instances:
            instance_class = instance.attrib.get("class")
            start = instance.findtext("Start")
            duration = instance.findtext("Duration")

            ann_stage_events.append(
                {
                    "Stage":instance_class,
                    "Start": start,
                    "Duration": duration,
                }
            )

        return ann_stage_events, ann_startdatetime