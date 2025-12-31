import os
from typing import Dict, List, Tuple
from datetime import datetime
import csv

from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("APOE")
class APOE(BaseDataset):
    """APOE (Sleep Disordered Breathing, ApoE and Lipid Metabolism) dataset"""

    def __init__(self):
        super().__init__("APOE","APOE - Sleep Disordered Breathing, ApoE and Lipid Metabolism")
        
    def _setup_dataset_config(self):
        self.ann2label = {  0: 0,
                            1: 1,
                            2: 2,
                            3: 3,
                            4: 3,  # Follow AASM Manual
                            5: 4,
                            7: 6,
                            8: 6,
                            10: 6,
                        }
        
        
        self.alias_mapping = {'flow': ['Flow II', 'FLOW'],
                            'thorax': ['CHEST #1', 'Chest', 'THOR'],
                            'LOC': ['LOC_A2', 'LOC/A2', 'L-EOG'],
                            'abdomen': ['ABDOM', 'Abdmn', 'Abdominal', 'Abdomen'],
                            'arm': ['arm', 'Arm', 'ARM', 'Arms'],
                            'snore': ['Snore2', 'MIC', 'MIC1', 'MIC2', 'Snore'],
                            'chin': ['Chin-Ctr', 'Chin EMG'],
                            'lchin': ['Chin-L'],
                            'rchin': ['Chin-R'],
                            'chin2': ['Chin2 EMG'],
                            'RAT-L': ['L/RAT', 'RAT-L'],
                            'ROC': ['R-EOG', 'ROC/A1', 'ROC-A1'],
                            'Cannula': ['CANNULA'],
                            'C4-A1': ['C4/A1'],
                            'F3-A2': ['F3/A2'],
                            'C3-A2': ['C3/A2'],
                            'F4-A1': ['F4/A1'],
                            'O1-A2': ['O1/A2'],
                            'O2-A1': ['O2/A1'],
                            'FP1-C3': ['Fp1-C3'],
                            'FP2-C4': ['Fp2-C4'],
                            'FP--': ['FP-?'],
                            'TcCO2': ['TcpCO2'],
                            'O1_A2': ['01_A2'],
                            }
        
        
        self.channel_names = [ 'Abdmn',  'Abdomen','ABDOM','Abdominal', 'Airflow', 'RAT-L','THOR', 'RIC','Flow II',
                            'RAT-U', 'Chin-R','EtCO2', 'L/RAT',  'LOC', 'Chin-L',  'EKG', 'Chin-Ctr',  'Chin EMG', 'Chin2 EMG',
                            'Nasal-Sum','PAP Pressure','CANNULA',  'JAWAC Sensor', 'PAP Patient Flow','Nasal-Left', 'Arm', 'Pulse Rate', 'Pressure', 'Snore', 'Nasal-Right', 'PAP Tidal Volume',
                            'MIC2', 'Line','LAT-L', 'SpO2', 'FLOW',  'Mic', 'ARM', 'Esophageal Press', 'VPW','PAP Flow', 
                            'LAT-U', 'Nasal Pressure', 'RIC-L', 'TcCO2', 'MIC1', 'Arms', 'Oral Thermistor', 'Bar', 
                            'PES',  'Oral2', 'blood pressure',  'CHEST #1','Pulse Transit Ti', 'TcpCO2', 'Cannula', 'Position', 'RIC-U',  'PAP Leak', 
                            'FP-?',  'Snore2',  'T3', 'Chest','ROC/A1', 'ROC','ROC-A1','R-EOG','LOC/A2','L-EOG','C4/A1','F4/A1','C4-A1','Fp2-C4', 'C3-O1','F1-A2', 'C4-A2','T4-O2','F2-C4', 'F3-A2',
                            'C3-A1', 'F3/A2', '01-A2','F4-A1', 'F2-T4','FP1-T3', 'T3-O1','Fz-A2', 'C3-A2', 'FP1-C3',
                            'C3/A2', 'LOC-A2','Fz-A1', 'O1/A2','FP2-C4', 'O2/A1','Fp1-C3', 'O1-x','O2-x']
        
        
        self.channel_types = {'analog': ['LOC', 'ROC/A1', 'VPW', 'ROC', 'T4-O2', 'Pulse Rate', 'O1/A2', 'F4-A1', 'Chin-L', 'Snore', 'Chin2 EMG', 'Abdominal',
                           'LOC-A2', 'F1-A2', 'FLOW', 'L-EOG', 'RIC-L', 'RIC', 'RAT-U', 'C3-A1', 'Bar', 'C3/A2', 'ARM', 'Fz-A2', 'FP2-C4', 
                           'RAT-L', 'Fz-A1', 'Fp1-C3', 'Nasal-Right', 'Cannula', 'Snore2', 'F4/A1', 'Nasal Pressure', 'Fp2-C4', 'O2/A1', 'Airflow', 
                           'C4-A2', 'Chin-Ctr', 'Abdomen', 'EKG', 'Chin EMG', 'Chest', 'EtCO2', 'Nasal-Sum', 'ROC-A1', 'F3-A2', 'T3', 'Position',
                           'MIC2', 'THOR', 'CHEST #1', 'LOC/A2', 'Oral Thermistor', 'Mic', 'R-EOG', 'C4/A1', 'CANNULA', 'LAT-U', 'F2-C4', 'Line',
                           'PAP Leak', 'Abdmn', 'Chin-R', 'ABDOM', 'F3/A2', 'Flow II', 'LAT-L', 'FP1-C3', 'FP1-T3', 'PAP Pressure', 'PES', 'F2-T4',
                           'O2-x', 'Pulse Transit Ti', 'C3-O1', 'L/RAT', 'C4-A1', 'C3-A2', 'O1-x', '01-A2', 'Arm', 'SpO2', 'FP-?', 'Arms', 'blood pressure',
                           'Esophageal Press', 'Nasal-Left', 'T3-O1', 'Oral2', 'MIC1', 'RIC-U'], 
                'digital': ['PAP Flow', 'TcCO2', 'PAP Tidal Volume', 'JAWAC Sensor', 'PAP Patient Flow', 'Pressure', 'TcpCO2']}
        
        
        self.channel_groups = {
            'eeg_eog': ['Chest', 'CHEST #1', 'Arms', 'LAT-U', 'Chin EMG', 'Chin2 EMG', 'ARM', 'RAT-L', 'RAT-U', 'Chin-R', 'L/RAT', 'Arm', 'Chin-L', 'LAT-L', 'Chin-Ctr'],
            'emg': ['Chest', 'CHEST #1', 'Arms', 'LAT-U', 'Chin EMG', 'Chin2 EMG', 'ARM', 'RAT-L', 'RAT-U', 'Chin-R', 'L/RAT', 'Arm', 'Chin-L', 'LAT-L', 'Chin-Ctr'],
            'ecg': ['EKG'],
            'thoraco_abdo_resp': ['ABDOM', 'Abdmn', 'Abdominal', 'Abdomen', 'CHEST #1', 'Chest', 'THOR', 'RIC-U', 'RIC-L'],
            'nasal_pressure': ['Nasal Pressure'],
            'snoring': ['Snore', 'Snore2']
        }
             
        
        self.file_extensions = {
            'psg_ext': '*.EDF',
            'ann_ext': '*.STA'
        }
        

    def dataset_paths(self) -> Tuple[str, str, str]:
        """Dataset paths for APOE dataset"""
        return (
            "APOE - Sleep Disordered Breathing, ApoE and Lipid Metabolism/original/PSG",
            "APOE - Sleep Disordered Breathing, ApoE and Lipid Metabolism/original/PSG"
        )

    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        Parse APOE STA annotation files.
        STA files contain space or tab-separated values with epoch number and sleep stage.
        
        Args:
            ann_fname: Path to STA annotation file
            epoch_duration: Duration of each epoch in seconds (default: 30)
            
        Returns:
            Tuple of (sleep_stage_events, start_datetime)
        """
        ann_stage_events = []
        epoch_duration = 30  # APOE uses 30-second epochs
        
        with open(ann_fname, 'r') as file:
            # Try space delimiter first
            for row in csv.reader(file, delimiter=" ", skipinitialspace=True):
                if len(row) >= 2:
                    epoch_num = int(float(row[0]))
                    stage = int(float(row[1]))
                    start = (epoch_num - 1) * epoch_duration  # Convert 1-based to 0-based
                    ann_stage_events.append({
                        'Stage': stage,
                        'Start': start,
                        'Duration': epoch_duration
                    })
            
        return ann_stage_events, None  # APOE doesn't provide start datetime in STA files
    
    def align_end(self, logger, psg_fname, ann_fname, signals, labels):
        # Alignment necessary due to cropping signal to whole epochs

        if len(labels) == len(signals) + 1:
            logger.info(f"Labels (len: {len(labels)}) are shortend to match signal ({len(signals)})")
            labels = labels[:len(signals)]

        assert len(signals) == len(labels), f"Length mismatch: signal ({os.path.basename(psg_fname)})={len(signals)}, labels({os.path.basename(ann_fname)})={len(labels)} TODO: implement alignment function"

        return signals, labels
