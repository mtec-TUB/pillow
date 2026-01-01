import os
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("FDCSR")
class FDCSR(BaseDataset):
    """FDCSR (Forced Desynchrony with and without Chronic Sleep Restriction) dataset."""
    
    def __init__(self):
        super().__init__("FDCSR","FDCSR - Forced Desynchrony with and without Chronic Sleep Restriction ", keep_folder_structure = False)

    def _setup_dataset_config(self):
        self.ann2label =  {
            5: 0,  # Wake
            1: 1,  # NREM Stage 1
            2: 2,  # NREM Stage 2
            3: 3,  # NREM Stage 3
            4: 3,  # Follow AASM Manual
            6: 4,  # REM sleep
            0: 6,  # Unknown
            7: 6,  # Movement
            8: 6,  # Lights Out
            9: 6,  # Lights On
        }
        
        
        self.alias_mapping = {
            'Cz_Mx': ['Cz Mx', 'Cz Ax'],
            'Pz_Mx': ['Pz Mx', 'Pz Ax'],
            'Oz_Mx': ['Oz Mx', 'Oz Ax'],
            'C4_Mx': ['C4 Mx', 'C4 Ax', 'C4 A1'],
            'C3_Mx': ['C3 Mx', 'C3 Ax', 'C3 A2'],
            'O1_M2': ['O1-M2', 'O1-A2'],
            'Fz_Mx': ['Fz Mx', 'Fz Ax'],
            'LOC': ['LOC A2', 'LOC Ax', 'E1 Mx'],
            'ROC': ['ROC A1', 'ROC Ax', 'E2 Mx'],
        }
        
        
        self.channel_names = [
            # EEG channels
            'ROC Ax', 'Cz Mx', 'E1 Mx', 'ROC A1', 'Oz Ax', 'Fz Ax', 'E2 Mx', 'Cz Ax',
            'C3 A2', 'O2 A1', 'O1 A2', 'C4 A1', 'C3 Mx', 'C4 Mx', 'Oz Mx', 'C4 Ax',
            'C3 Ax', 'LOC Ax', 'Fz Mx', 'LOC A2', 'Pz Ax', 'Pz Mx',
            'EMG sm',
            'ECG',
            'Marker'
        ]
        
        
        self.channel_types = {
            'analog': [
                'ROC Ax', 'Cz Mx', 'E1 Mx', 'ROC A1', 'Oz Ax', 'Fz Ax', 'E2 Mx', 'Cz Ax',
                'C3 A2', 'O2 A1', 'O1 A2', 'C4 A1', 'C3 Mx', 'C4 Mx', 'Oz Mx', 'C4 Ax',
                'C3 Ax', 'LOC Ax', 'Fz Mx', 'LOC A2', 'Pz Ax', 'Pz Mx', 'EMG sm', 'ECG'
            ],
            'digital': ['Marker']
        }
                
        self.channel_groups = {
            'eeg_eog': ['ROC Ax', 'Cz Mx', 'E1 Mx', 'ROC A1', 'Oz Ax', 'Fz Ax', 'E2 Mx', 'Cz Ax', 'C3 A2', 'O2 A1', 'O1 A2', 'C4 A1', 'C3 Mx', 'C4 Mx', 'Oz Mx', 'C4 Ax', 'C3 Ax', 'LOC Ax', 'Fz Mx', 'LOC A2', 'Pz Ax', 'Pz Mx'],
            'emg': ['EMG sm'],
            'ecg': ['ECG']
        }
        
        
        self.file_extensions = {
            'psg_ext': '*/*.edf',
            'ann_ext': '*/*_score.csv'
        }
        

    
    def dataset_paths(self) -> Tuple[str, str]:
        """
        FDCSR dataset paths.
        """
        data_dir = "FDCSR - Forced Desynchrony with and without Chronic Sleep Restriction /subjects"
        ann_dir = "FDCSR - Forced Desynchrony with and without Chronic Sleep Restriction /subjects"
        return data_dir, ann_dir
    
    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        Parse FDCSR annotation files with custom datetime handling.
        """

        # recordings only contain information about month and day (not year),
        # recordings took place between 2000 and 2016, default value 1900 to distinguish
        study_start_datetime = datetime(year=1900,month=1,day=1)     

        epoch_duration = 30  # FDCSR uses 30-second epochs   

        ann_df = pd.read_csv(ann_fname,sep=',',header=0)
    
        ann_Startdatetime = (study_start_datetime + timedelta(hours=ann_df.iloc[0]['labtime']))

        # round to full seconds (like in psg file)
        new_sec = ann_Startdatetime.second + round(ann_Startdatetime.microsecond/1000000)
        new_min = int(ann_Startdatetime.minute)
        new_hour = int(ann_Startdatetime.hour)
        if new_sec == 60:
            new_sec = 0
            new_min = new_min + 1
            if new_min == 60:
                new_min = 0
                new_hour = new_hour + 1
        ann_Startdatetime = ann_Startdatetime.replace(hour=new_hour, minute=new_min, second=new_sec, microsecond=0)

        ann_stage_events = []
        for i, row in ann_df.iterrows():
            
            start = i * epoch_duration
            duration = epoch_duration
            stage = row['stage']
            
            ann_stage_events.append({'Stage': stage,
                                    'Start': start,
                                    'Duration': duration})        

        return ann_stage_events, ann_Startdatetime

    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(labels) == len(signals) + 1:
            return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)

        if len(signals) == len(labels) + 1:
            return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)
    
    def preprocess(self, data_dir, ann_dir, output_dir):
        print("\n FDCSR files originally are stored in an unsupported format and therefor need to be preprocessed/resorted \n")
        
        execute_preprocess = input("Do you want to perform the preprocessing now? (Y/N) ")
        
        if str(execute_preprocess).lower() == "y":
            splitter = FDCSRSleepScoreSplitter(data_dir)
            unscored_files = splitter.process_sleep_scores()
            if unscored_files:
                dest_folder = os.path.join(os.path.split(output_dir)[0], "unscored_files")
                print(f"{len(unscored_files)} unscored files were found. They will be moved to {dest_folder}")
                FDCSRFileOrganizer.move_unscored_files(data_dir, dest_folder, unscored_files)
                print("Successfully ended preprocessing")
                
                if str(input("Do you want to continue with processing now? (Y/N) ")).lower() == "n":
                    return False
        return True
            

class FDCSRSleepScoreSplitter:
    """Handles splitting of FDCSR sleep score files and file organization."""

    def __init__(self, base_folder: str):
        """
        Initialize the splitter.

        Args:
            base_folder: Path to FDCSR subjects folder
            start_times_file: Path to CSV file with EDF timing information
        """
        self.base_folder = Path(base_folder)
        self.start_times_file = os.path.join(base_folder,"fdcsr_edf_start_times.csv")
        self.not_scored = []

    def process_sleep_scores(self):
        """Process all sleep score files based on EDF timing information."""
        start_times_df = pd.read_csv(self.start_times_file, sep=",", header=0)

        print(f"Processing {len(start_times_df)} EDF files...")

        for i, edf_info in start_times_df.iterrows():
            self._process_single_edf(edf_info)

        print(f"\nProcessing complete. Unscored files: {len(self.not_scored)}")
        if self.not_scored:
            print("Unscored files:", self.not_scored)

        return self.not_scored

    def _process_single_edf(self, edf_info):
        """Process a single EDF file's sleep scores."""
        edf_filename = edf_info["filename"]

        # Skip unscored files
        if edf_info["comment"] == "not scored":
            self.not_scored.append(edf_filename + ".edf")
            return

        edf_folder_name = edf_filename.split("_")[0]

        # Construct file paths
        edf_filepath = self.base_folder / edf_folder_name.upper() / f"{edf_filename}.edf"
        score_filepath = self.base_folder / edf_folder_name.upper() / f"{edf_folder_name}-scoredsleep.csv"
        output_filepath = self.base_folder / edf_folder_name.upper() / f"{edf_filename}_score.csv"

        # Skip if output already exists
        if output_filepath.exists():
            return

        print(f"\nProcessing: {edf_filename}")
        print(f"  EDF: {edf_filepath}")
        print(f"  Score source: {score_filepath}")

        self._extract_matching_scores(edf_info, score_filepath, output_filepath)


    def _extract_matching_scores(self, edf_info, score_filepath, output_filepath):
        """Extract sleep scores matching the EDF file timing."""


        found_time_mismatch = False
        
        if not pd.isna(edf_info["start time"]):
            start_time = datetime.strptime(edf_info["start time"],"%H:%M:%S")
            start_time_as_delta = timedelta(hours=start_time.hour, minutes=start_time.minute,seconds=start_time.second)

            hours = timedelta(hours=((edf_info["start labtime"]/24) - int((edf_info["start labtime"]/24)))*24) # remove full days
            start_lab_time = timedelta(seconds=round(hours.total_seconds()))

            if start_time_as_delta != start_lab_time:
                found_time_mismatch = True
                print(start_time_as_delta)
                print(start_lab_time)
                print('found start time mismatch')
            else:
                print(start_time_as_delta)
                print(start_lab_time)
                print('start time is matching')

        # Get timing information
        start_labtime = round(edf_info["start labtime"], 6)
        last_line_time = round(edf_info["last labtime"], 6)
        n_lines = int(edf_info["last line"]) - 1

        print(f"  Time range: {start_labtime} to {last_line_time} ({n_lines} lines)")

        # Load sleep scores (handle special case for 3339gx)
        edf_folder_name = edf_info["filename"].split("_")[0]
        if edf_folder_name == "3339gx":
            sleep_scores_df = pd.read_csv(
                score_filepath,
                sep=",",
                names=["subjectID", "periodID", "labtime", "stage", "None"],
            )
            sleep_scores_df.drop("None", axis=1, inplace=True)
        else:
            try:
                sleep_scores_df = pd.read_csv(
                    score_filepath,
                    sep=",",
                    names=["subjectID", "periodID", "labtime", "stage"],
                )
            except Exception:
                print(f'File {edf_folder_name} not found')
                return
        # Round labtime for matching
        sleep_scores_df["labtime"] = round(sleep_scores_df["labtime"], 6)

        # Find matching time range
        idx_start = (sleep_scores_df["labtime"] == start_labtime).idxmax()
        idx_stop = (sleep_scores_df["labtime"] == last_line_time).argmax()

        print(f"  Score indices: {idx_start} to {idx_stop}")

        # Validate the extraction
        assert sleep_scores_df.iloc[idx_stop]["labtime"] == sleep_scores_df.iloc[idx_start + n_lines]["labtime"], f"Time mismatch: expected {sleep_scores_df.iloc[idx_start + n_lines]['labtime']}, got {sleep_scores_df.iloc[idx_stop]['labtime']}"

        # Extract and save matching scores
        matching_scores = sleep_scores_df.loc[idx_start:idx_stop].copy()
        if found_time_mismatch or any(subfolder in edf_folder_name for subfolder in ['3441gx', '3450gx', '3525gx', '3531gx', '3540gx']):
            matching_scores["labtime"] = matching_scores["labtime"] + 1

        matching_scores.to_csv(output_filepath, index=False)
        print(f"  SUCCESS: Saved {len(matching_scores)} score records")


class FDCSRFileOrganizer:
    """Handles moving unscored FDCSR files to a separate directory."""

    @staticmethod
    def move_unscored_files(source_root_folder: str, destination_folder: str, filenames_to_move: List[str]):
        """
        Move specific files from subfolders to a destination directory.

        Args:
            source_root_folder: Root folder to search for files
            destination_folder: Destination folder for moved files
            filenames_to_move: List of filenames to move
        """
        source_root_path = Path(source_root_folder)
        destination_path = Path(destination_folder)

        # Create destination folder
        destination_path.mkdir(parents=True, exist_ok=True)
        print(f"Destination folder: {destination_path}")

        stats = {"moved": 0, "not_found": 0, "errors": 0}

        for filename in filenames_to_move:
            try:
                # Find file in subfolders
                found_files = list(source_root_path.glob(f"*/{filename}"))

                if len(found_files) == 1:
                    source_file = found_files[0]
                    target_file = destination_path / filename

                    # Move the file
                    shutil.move(source_file, target_file)
                    print(f"✓ Moved: {filename}")
                    stats["moved"] += 1

                elif len(found_files) == 0:
                    print(f"✗ Not found: {filename}")
                    stats["not_found"] += 1

                else:
                    print(f"✗ Multiple matches for: {filename}")
                    stats["errors"] += 1

            except Exception as e:
                print(f"✗ Error moving {filename}: {e}")
                stats["errors"] += 1

        # Print summary
        print("\n--- Move Operation Summary ---")
        print(f"Files moved: {stats['moved']}")
        print(f"Files not found: {stats['not_found']}")
        print(f"Errors: {stats['errors']}")
