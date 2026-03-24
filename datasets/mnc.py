"""
MNC - Mignot Nature Communications
"""

import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import xml.etree.ElementTree as ET

from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("MNC")
class MNC(BaseDataset):
    """MNC - Mignot Nature Communications"""
    
    def __init__(self):
        super().__init__("MNC","MNC - Mignot Nature Communications",keep_folder_structure=False)

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

        self.intra_dataset_mapping = {
            "ECG": ["ECG","ECG1_2","ECG1"],
            "chin": ['chin','cchin'],
        }

        #https://sleepdata.org/datasets/mnc/pages/montage-and-sampling-rate-information.md
        self.inter_dataset_mapping = {
            "flow": self.Mapping(self.TTRef.AIRFLOW, None),
            "O1": self.Mapping(self.TTRef.O1, self.TTRef.Fpz),
            "O2": self.Mapping(self.TTRef.O2, self.TTRef.Fpz),
            "F3": self.Mapping(self.TTRef.F3, self.TTRef.Fpz),
            "M1": self.Mapping(self.TTRef.LPA, self.TTRef.Fpz),
            "M2": self.Mapping(self.TTRef.RPA, self.TTRef.Fpz),
            "F4": self.Mapping(self.TTRef.F4, self.TTRef.Fpz),
            "E1": self.Mapping(self.TTRef.EL, self.TTRef.Fpz),
            "E2": self.Mapping(self.TTRef.ER, self.TTRef.Fpz),
            "C3": self.Mapping(self.TTRef.C3, self.TTRef.Fpz),
            "C4": self.Mapping(self.TTRef.C4, self.TTRef.Fpz),
            "rleg": self.Mapping(self.TTRef.EMG_RLEG, None),
            "lleg": self.Mapping(self.TTRef.EMG_LLEG, None),
            "chin": self.Mapping(self.TTRef.EMG_CHIN, None),
            "cchin_l": self.Mapping(self.TTRef.EMG_CHIN, self.TTRef.EMG_LCHIN),
            "rchin_c": self.Mapping(self.TTRef.EMG_CHIN, self.TTRef.EMG_RCHIN),
            "rchin": self.Mapping(self.TTRef.EMG_RCHIN, None),
            "position": self.Mapping(self.TTRef.POSITION, None),
            "abdomen": self.Mapping(self.TTRef.ABDOMINAL, None),
            "thorax": self.Mapping(self.TTRef.THORACIC, None),
            "spo2": self.Mapping(self.TTRef.SPO2, None),
            "snore": self.Mapping(self.TTRef.SNORE, None),
            "ECG": self.Mapping(self.TTRef.ECG, None),
            "pap_flow": self.Mapping(self.TTRef.CPAP, None),
        }
        
        
        self.channel_names = ['O1', 'ECG', 'F3', 'activity', 'lleg1_2', 'ECG2', 'position', 'therm', 'etco2', 'lleg_r', 'ECG1', 'cchin_l',
                              'M1', 'lleg', 'F4', 'tvol', 'light', 'pap_pres', 'sum', 'rleg', 'abdomen', 'pap_flow', 'gravx', 'leak', 'E1',
                              'rleg2', 'phase', 'PTT', 'spo2', 'pulse', 'elevation', 'rchin_c', 'cs_LOC', 'cs_EMG', 'lchin', 'cchin', 'r_r',
                              'pleth', 'gravy', 'lleg1', 'thorax', 'E2', 'C3', 'C4', 'ECG1_2', 'spo2bb', 'cs_ECG', 'O2', 'cs_ROC', 'M2',
                              'rleg1', 'cs_EEG', 'ECG3', 'chin', 'snore', 'hr', 'flow', 'rleg1_2', 'nas_pres', 'rchin']
        
        
        self.channel_types = {'analog': ['etco2', 'flow', 'lleg1_2', 'ECG1', 'rleg', 'ECG3', 'rchin', 'E2', 'activity', 'E1', 'snore', 
                                         'gravy', 'tvol', 'chin', 'lleg1', 'ECG1_2', 'cchin', 'C4', 'lchin', 'pleth', 'abdomen', 'rchin_c', 
                                         'F3', 'rleg1_2', 'spo2', 'r_r', 'pap_pres', 'cchin_l', 'thorax', 'C3', 'cs_ECG', 'cs_LOC', 'cs_EEG', 
                                         'M1', 'gravx', 'F4', 'pap_flow', 'O2', 'rleg1', 'rleg2', 'therm', 'hr', 'M2', 'sum', 'O1', 'leak', 
                                         'phase', 'cs_EMG', 'ECG', 'cs_ROC', 'lleg_r', 'light', 'elevation', 'lleg', 'PTT', 'nas_pres', 'ECG2', 'pulse'],
                             'digital': ['spo2bb', 'position']}
        
        
        self.channel_groups = {
            'eeg_eog': ['C4', 'cs_LOC', 'E1','E2', 'F3','O2','cs_ROC','cs_EEG', 'F4', 'C3', 'O1', 'O2','M1','M2'],
            'emg': ['rleg1_2','rleg2','rleg', 'lleg1','lleg_r','rchin','cchin', 'cs_EMG','lleg1_2', 'rleg1','lchin','lleg',  'cchin_l', 'chin','rchin_c'],
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
        return [
            '',
            ''
        ]

    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
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

        return ann_stage_events, ann_startdatetime, None, None