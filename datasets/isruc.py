import os
import shutil
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from datasets.base import BaseDataset, DatasetProcessingError
from datasets.registry import register_dataset
from psg_processing.utils import SleepStage


@register_dataset("ISRUC")
class ISRUC(BaseDataset):
    """ISRUC dataset with multiple scorers"""

    def __init__(self):
        super().__init__("ISRUC","ISRUC")
        self.use_both_scorers = True
        self.has_end_alignment = True

    def _setup_dataset_config(self):
        self.ann2label =  {
            "W": SleepStage.W,   # Wake
            "w": SleepStage.W,     # in file "Subgroup 3_9_9_9_1.xlsx"
            "N": SleepStage.N1,     # in file "Subgroup 1_4_4_4_1.xlsx", compared to "Subgroup 1_4_4_4_1.txt" -> N1
            "N1": SleepStage.N1,   # Stage 1
            "N2": SleepStage.N2,   # Stage 2
            "n2": SleepStage.N2,   # in file "Subgroup 1_2_2_2.xlsx"
            "N3": SleepStage.N3,   # Stage 3
            "R": SleepStage.REM,    # REM
            "U": SleepStage.UNK,    # Unknown
        }


        self.intra_dataset_mapping = {
            'ECG': ['25', 'X2'],
            'Chin': ['24', 'X1'],
            'LLeg': ['26', 'X3'],
            'RLeg': ['27', 'X4'],
            'Snore': ['28', 'X5'],
            'Flow-1': ['X6'],
            'Flow-2': ['DC3'],
            'Abdominal_X7': ['X7',  '30'],
            'Abdominal_X8': ['X8', '31'],
            'Body': ['DC8'],
            'O2_A1': ['O2-M1', 'O2-A1'],
            'C4_A1': ['C4-M1', 'C4-A1'],
            'C3_A2': ['C3-M2', 'C3-A2'],
            'O1_A2': ['O1-M2', 'O1-A2'],
            'F3_A2': ['F3-M2', 'F3-A2'],
            'F4_A1': ['F4-M1', 'F4-A1'],
            'SpO2': ['SpO2', 'SaO2'],
            'LOC_A2': ['LOC-A2', 'E1-M2'],
            'ROC_A1': ['ROC-A1', 'E2-M1'],
        }


        self.inter_dataset_mapping = {
            "F3_A2": self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            "C3_A2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "F4_A1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            "C4_A1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "O1_A2": self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            "O2_A1": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "ROC_A1": self.Mapping(self.TTRef.ER, self.TTRef.LPA),
            "LOC_A2": self.Mapping(self.TTRef.EL, self.TTRef.RPA),
            "SpO2": self.Mapping(self.TTRef.SPO2, None),
            "ECG": self.Mapping(self.TTRef.ECG, None),
            "Chin": self.Mapping(self.TTRef.EMG_CHIN, None),
            "LLeg": self.Mapping(self.TTRef.EMG_LLEG, None),
            "RLeg": self.Mapping(self.TTRef.EMG_RLEG, None),
            "Body": self.Mapping(self.TTRef.POSITION, None),
            "Abdominal_X7": self.Mapping(self.TTRef.ABDOMINAL, None),
            "Snore": self.Mapping(self.TTRef.SNORE, None),
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
            'thoraco_abdo_resp': ['X7', 'X8', '30', '31', 'X7-X8','X6','DC3'],
            'snoring': ['28', 'X5']
        }

        self.file_extensions = {
            'psg_ext': '*/*.rec',
            'ann_ext': '*/*_1.xlsx'
        }

    def dataset_paths(self):
        return [
            'Data',
            'Data'
        ]

    def ann_parse(self, ann_fname: str):
        """Parse ISRUC annotation files (multiple scorers in separate files)"""
        # ISRUC has two annotation files: *1.xlsx and *2.xlsx
        base_fname = ann_fname.replace('1.xlsx', '')

        epoch_duration = 30  # ISRUC uses 30-second epochs

        ann_stage_events = []

        lights_off, lights_on = [], []

        exp_col_names = ['Epoch', 'Stage', 'SpO2', 'HR', 'Events', 'BPOS', 'Txln', 'TxEx', 'Technote']

        # Load both scorer files
        for scorer_suffix in ['1.xlsx', '2.xlsx']:
            ann_fname_scorer = base_fname + scorer_suffix
            if os.path.exists(ann_fname_scorer):
                df = pd.read_excel(ann_fname_scorer, usecols=range(9))
                if 'Stage' not in df.columns:
                    df = pd.read_excel(ann_fname_scorer, header=None, usecols=range(9), names=exp_col_names)
                scorer_events = []
                epoch_col = df.columns[0]  # Assuming the first column is the epoch column (usually named "Epoch" but sometimes "hich")
                for i, row in df.iterrows():
                    if pd.isna(row[epoch_col]):
                        continue  # Skip rows without epoch information
                    if row.astype(str).str.contains('L Out').any():
                        lights_off.append(i* epoch_duration)
                    if row.astype(str).str.contains('L On').any():
                        lights_on.append((i+1) * epoch_duration)

                    scorer_events.append({
                        'Start': i * epoch_duration,
                        'Duration': epoch_duration,
                        'Stage': row['Stage']
                    })
                ann_stage_events.append(scorer_events)
            else:
                ann_stage_events.append([])  # No annotation for this scorer

        if not lights_off:
            raise DatasetProcessingError(
                "No Lights Off event found in annotations", self.dset_name, "ann_parse"
            ) # should not occur
        elif len(lights_off) > 2 or lights_off[0] != lights_off[1]:
            if "Subgroup 3_7_7_7" in ann_fname:
                # In this file there are two Lights Off events (epoch 0 and epoch 17), the second one seems more reasonable (appears in both scorer files)
                lights_off = [lights_off[1]]
            else:
                raise DatasetProcessingError(
                    f"Multiple Lights Off events found in annotations: {lights_off}",
                    self.dset_name,
                    "ann_parse",
                )

        if not lights_on:
            raise DatasetProcessingError(
                "No Lights On event found in annotations", self.dset_name, "ann_parse"
            )  # should not occur
        elif len(lights_on) > 2 or lights_on[0] != lights_on[1]:
            raise DatasetProcessingError(
                f"Multiple Lights On events found in annotations: {lights_on}",
                self.dset_name,
                "ann_parse",
            ) # should not occur

        return ann_stage_events, None, lights_off[0], lights_on[0]

    def preprocess(self, n_workers, data_dir, ann_dir):
        print("\n ISRUC files originally are stored in an inconvenient way and therefor should be preprocessed/resorted ... \n \
              This will not modify the original file content")

        execute_preprocess = input("Do you want to perform the resorting now? (Y/N) ")

        if str(execute_preprocess).lower() == "y":
            organizer = ISRUCFileOrganizer(data_dir)
            organizer.organize_files(n_workers)

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

    def organize_files(self, n_workers):
        """Reorganize all files from group subfolders.
            Instead of nested folders, store the files with unique filenames all directly inside the subgroup folders
        """

        # Process each subgroup
        subgroups_folders = [str(f) for f in self.base_dir.iterdir() if f.is_dir()]

        for subgroup_folder in subgroups_folders:

            files = glob.glob(subgroup_folder+"/*/**/*.*", recursive=True)

            try:
                with ThreadPoolExecutor(max_workers=n_workers) as executor:

                    futures = [executor.submit(self._copy_file, file, subgroup_folder) for file in files]
                    for future in tqdm(as_completed(futures), total=len(files), desc=f"Resorting files in {subgroup_folder}"):
                        future.result()
            except KeyboardInterrupt:
                print("Preprocessing interrupted by user. Shutting down...")
                executor.shutdown(cancel_futures=True)

        print('Resorting done, if you want you can delete the duplicates at their original location.')

    def _copy_file(self, file, subgroup_folder):
        new_filename = os.path.basename(subgroup_folder)+'_'+str(Path(file).relative_to(Path(subgroup_folder))).replace('/','_')
        if not os.path.exists(subgroup_folder+'/'+new_filename):
            # print(f"Copy file {file} to {subgroup_folder+'/'+new_filename}")
            shutil.copy2(file,subgroup_folder+'/'+new_filename)
