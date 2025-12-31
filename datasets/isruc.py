import os
import numpy as np
import pandas as pd
import shutil
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("ISRUC")
class ISRUC(BaseDataset):
    """ISRUC dataset with multiple scorers"""
    
    def __init__(self):
        super().__init__("ISRUC","ISRUC")
        
    def _setup_dataset_config(self):
        self.ann2label =  {
            0: 0,   # Wake
            1: 1,   # Stage 1
            2: 2,   # Stage 2
            3: 3,   # Stage 3
            5: 4    # REM (ISRUC uses 5 for REM)
        }
        
        
        self.alias_mapping = {
            'ECG': ['25', 'X2'],
            'Chin': ['24', 'X1'],
            'LLeg': ['26', 'X3'],
            'RLeg': ['27', 'X4'],
            'Snore': ['28', 'X5'],
            'Flow': ['29', 'X6', 'DC3'],
            'Abdominal_X7': ['X7',  '30'],
            'Abdominal_X8': ['X8', '31'],
            'Body': ['DC8'],
            'O2_M1': ['O2-M1', 'O2-A1'],
            'C4_M1': ['C4-M1', 'C4-A1'],
            'C3_M2': ['C3-M2', 'C3-A2'],
            'O1_M2': ['O1-M2', 'O1-A2'],
            'F3_M2': ['F3-M2', 'F3-A2'],
            'F4_M1': ['F4-M1', 'F4-A1'],
            'SpO2': ['SpO2', 'SaO2'],
            'LOC-M2': ['LOC-A2', 'E1-M2'],
            'ROC-M1': ['ROC-A1', 'E2-M1'],
        }
        
        
        self.channel_names = [
            'O1', 'C3', 'F3', 'F4', 'O2', 'C4', 'LOC', 'ROC', 'A1', 'A2',
            'O2-M1', 'O2-A1', 'C4-M1', 'C4-A1', 'C3-M2', 'C3-A2', 
            'O1-M2', 'O1-A2', 'F3-M2', 'F3-A2', 'F4-M1', 'F4-A1',
            'LOC-A2', 'E1-M2', 'ROC-A1', 'E2-M1',
            '24', 'X1', '26', 'X3', '27', 'X4', 
            '25', 'X2', '28', 'X5', '29', 'X6', 'DC3',
            'X7', 'X8', '30', '31', 'X7-X8',
            'DC8', 'SpO2', 'SaO2'
        ]
        
        
        self.channel_types =  {'analog': ['LOC-A2', 'O2-A1', 'X7', '24', 'C3-M2', 'ROC', 'X8', 'O2', 'F3-A2', 'X7-X8', 'X1', 'C4-A1', '30', 'C4-M1', 'E1-M2', '28',
                           'SpO2', 'O1-M2', 'O1-A2', 'A2', 'C3', 'O2-M1', 'F3', '31', 'X4', 'O1', '29', 'F4-A1', 'F4-M1', 'LOC', 'X2', 'F4', '26', '25',
                           'X6', '27', 'C4', 'ROC-A1', 'C3-A2', 'SaO2', 'F3-M2', 'DC3', 'E2-M1', 'X5', 'X3', 'A1'],
                'digital': ['DC8']}
        
        
        self.channel_groups = {
            'eeg_eog': ['O1', 'C3', 'F3', 'F4', 'O2', 'C4', 'LOC', 'ROC', 'C3', 'O1', 'O2', 'A1', 'C4', 'A2', 'F4', 'O2-M1', 'O2-A1', 'C4-M1', 'C4-A1', 'C3-M2', 'C3-A2', 'O1-M2', 'O1-A2', 'F3-M2', 'F3-A2', 'F4-M1', 'F4-A1', 'LOC-A2', 'E1-M2', 'ROC-A1', 'E2-M1'],
            'emg': ['24', 'X1', '26', 'X3', '27', 'X4'],
            'ecg': ['25', 'X2'],
            'thoraco_abdo_resp': ['X7', 'X8', '30', '31', 'X7-X8'],
            'snoring': ['28', 'X5']
        }
        
        
        self.file_extensions = {
            'psg_ext': '*/*.rec',
            'ann_ext': '*/*_1.txt'
        }
    
    def dataset_paths(self) -> Tuple[str, str]:
        """
        ISRUC dataset paths.
        """
        data_dir = "ISRUC/Data"
        ann_dir = "ISRUC/Data"
        return data_dir, ann_dir
    
    def ann_parse(self, ann_fname: str) -> Tuple[List[List[Dict]], datetime]:
        """Parse ISRUC annotation files (multiple scorers in separate files)"""
        # ISRUC typically has two annotation files: *1.txt and *2.txt
        base_fname = ann_fname.replace('1.txt', '')

        epoch_duration = 30  # ISRUC uses 30-second epochs
        
        ann_stage_events_1 = []
        ann_stage_events_2 = []
        
        # Load first scorer
        ann_fname_1 = base_fname + '1.txt'
        if os.path.exists(ann_fname_1):
            df = pd.read_csv(ann_fname_1, names=['Stage'])
            for i, row in df.iterrows():
                ann_stage_events_1.append({
                    'Start': i * epoch_duration,
                    'Duration': epoch_duration,
                    'Stage': row['Stage']
                })
        
        # Load second scorer
        ann_fname_2 = base_fname + '2.txt'
        if os.path.exists(ann_fname_2):
            df = pd.read_csv(ann_fname_2, names=['Stage'])
            for i, row in df.iterrows():
                ann_stage_events_2.append({
                    'Start': i * epoch_duration,
                    'Duration': epoch_duration,
                    'Stage': row['Stage']
                })
        
        return [ann_stage_events_1, ann_stage_events_2], None
    
    def ann_label(self, logger, ann_stage_events: List[List[Dict]], epoch_duration: int) -> np.ndarray:
        """
        Convert multi-scorer sleep stage events to epoch-wise labels for ISRUC dataset.
        Returns 2D array (n_epochs, n_scorers).
        """
        labels = [np.array([]), np.array([])]

        for i, annotation in enumerate(ann_stage_events):  # two scorers
            total_duration = 0
            for event in annotation:
                onset_sec = event['Start']
                duration_sec = event['Duration']
                ann_str = event['Stage']

                # Sanity check
                assert onset_sec == total_duration, f"Onset sec of epoch is {onset_sec} but last epoch ended at {total_duration}"

                # Get label value
                if ann_str in self.ann2label:
                    label = self.ann2label[ann_str]
                else:
                    logger.info(f"Something unexpected: label {ann_str} not found")
                    raise Exception(f"Something unexpected: label {ann_str} not found")

                # Compute # of epoch for this stage
                if duration_sec % epoch_duration != 0:
                    logger.info(f"Something wrong: {duration_sec} {epoch_duration}")
                    raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
                duration_epoch = int(duration_sec / epoch_duration)

                # Generate sleep stage labels
                label_epoch = np.ones(duration_epoch, dtype=np.int32) * label
                labels[i] = np.append(labels[i], label_epoch)

                total_duration += duration_sec

                # logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
                #     onset_sec, duration_sec, label, ann_str
                # ))

        # Pad shorter annotation to match longer one
        if len(labels[0]) != len(labels[1]):
            max_len = max(len(labels[0]), len(labels[1]))
            labels[0] = np.pad(labels[0], (0, max_len - len(labels[0])), mode='constant', constant_values=6)
            labels[1] = np.pad(labels[1], (0, max_len - len(labels[1])), mode='constant', constant_values=6)

        labels = np.array(labels).T  # Transpose to (n_epochs, n_scorers)
        
        return labels
    
    def align_end(self, logger, psg_fname: str, ann_fname:str, signals: np.ndarray,
                  labels: np.ndarray,
                  ) -> Tuple[np.ndarray, np.ndarray]:
        
        if len(labels) > len(signals):
            logger.info(f"Labels (len: {len(labels)}) are shortend to match signal ({len(signals)})")
            labels = labels[:len(signals)]
        
        assert len(signals) == len(labels), f"Length mismatch: signal ({(psg_fname)})={len(signals)}, labels({os.path.basename(ann_fname)})={len(labels)} TODO: implement alignment function"

        return signals, labels
        

    def preprocess(self, data_dir, ann_dir, output_dir):
        print("\n ISRUC files originally are stored in an inconvenient way and therefor should be preprocessed/resorted ... \n \
              This will not modify the original file content")
        
        execute_preprocess = input("Do you want to perform the resorting now? (Y/N) ")
        
        if str(execute_preprocess).lower() == "y":
            organizer = ISRUCFileOrganizer(data_dir)
            organizer.organize_files()
        
            if str(input("Do you want to continue with processing now? (Y/N) ")).lower() == "n":
                return False
        return True
            

class ISRUCFileOrganizer:
    """Handles reorganization of ISRUC dataset files."""

    def __init__(self, base_dir: str):
        """
        Initialize the organizer.

        Args:
            base_dir: Path to ISRUC dataset root directory
        """
        self.base_dir = Path(base_dir)

    def organize_files(self):
        """Reorganize all files from group subfolders.
            Instead of nested folders, store the files with unique filenames all directly inside the subgroup folders
        """

        # Process each subgroup
        subgroups_folders = [str(f) for f in self.base_dir.iterdir() if f.is_dir()]

        for subgroup_folder in subgroups_folders:
            print(subgroup_folder)
            files = glob.glob(subgroup_folder+"/*/**/*.*", recursive=True)
            
            for file in files:
                new_filename = os.path.basename(subgroup_folder)+'_'+os.path.basename(file).replace('/','_')
                print(f"Copy file {file} to {subgroup_folder+'/'+new_filename}")
                shutil.copy(file,subgroup_folder+'/'+new_filename)
                
        print('Resorting done, if you want you can delete the duplicates at their original location.')

