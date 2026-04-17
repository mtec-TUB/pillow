import os
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import pyedflib
from typing import Dict, List, Optional, Tuple
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("FDCSR")
class FDCSR(BaseDataset):
    """FDCSR (Forced Desynchrony with and without Chronic Sleep Restriction) dataset."""
    
    def __init__(self):
        super().__init__("FDCSR","FDCSR - Forced Desynchrony with and without Chronic Sleep Restriction", keep_folder_structure = False)

    def _setup_dataset_config(self):
        self.ann2label =  {
            5: "W",  # Wake
            1: "N1",  # NREM Stage 1
            2: "N2",  # NREM Stage 2
            3: "N3",  # NREM Stage 3
            4: "N3",  # Follow AASM Manual
            6: "REM",  # REM sleep
            0: "UNK",  # Unknown
            7: "MOVE",  # Movement
            8: "UNK",  # Lights Out
            9: "UNK",  # Lights On
        }
        
        
        self.intra_dataset_mapping = {
            'Cz': ['Cz Mx', 'Cz Ax'],
            'Pz': ['Pz Mx', 'Pz Ax'],
            'Oz': ['Oz Mx', 'Oz Ax'],
            'C4': ['C4 Mx', 'C4 Ax'],
            'C3': ['C3 Mx', 'C3 Ax'],
            'C3_A2': ['C3 A2', 'C3 M2'],
            'C4_A1': ['C4 A1', 'C4 M1'],
            'O1_A2': ['O1 M2', 'O1 A2'],
            'O2_A1': ['O2 A1', 'O2 M1'],
            'Fz_Ax': ['Fz Mx', 'Fz Ax'],
            'LOC': ['LOC Ax', 'E1 Mx'],
            'ROC': ['ROC Ax', 'E2 Mx'],
        }

        # doi: 10.1038/npp.2010.63 
        self.inter_dataset_mapping = {
            "Cz": self.Mapping(self.TTRef.Cz, self.TTRef.LRPA),
            "Pz": self.Mapping(self.TTRef.Pz, self.TTRef.LRPA),
            "P3 M2": self.Mapping(self.TTRef.P3, self.TTRef.RPA),
            "P4 M1": self.Mapping(self.TTRef.P4, self.TTRef.LPA),
            "Oz": self.Mapping(self.TTRef.Oz, self.TTRef.LRPA),
            "C4": self.Mapping(self.TTRef.C4, None),
            "C3": self.Mapping(self.TTRef.C3, None),
            "E2 M1": self.Mapping(self.TTRef.ER, self.TTRef.LPA),
            "E1 M2": self.Mapping(self.TTRef.EL, self.TTRef.RPA),
            "O1_A2": self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            "O2_A1": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "Fz_Ax": self.Mapping(self.TTRef.Fz, self.TTRef.LRPA),
            "F4 M1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            "F3 M2": self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            "LOC": self.Mapping(self.TTRef.EL, None),
            "LOC A2": self.Mapping(self.TTRef.EL, self.TTRef.RPA),
            "ROC": self.Mapping(self.TTRef.ER, None),
            "ROC A1": self.Mapping(self.TTRef.ER, self.TTRef.LPA),
            "C4_A1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "C3_A2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "ECG": self.Mapping(self.TTRef.ECG, None),
            "EMG sm": self.Mapping(self.TTRef.EMG_CHIN, None),
        }
        
        
        self.channel_names = [
            'ROC Ax', 'Cz Mx', 'E1 Mx', 'ROC A1', 'Oz Ax', 'Fz Ax', 'E2 Mx', 'Cz Ax',
            'C3 A2', 'O2 A1', 'O1 A2', 'C4 A1', 'C3 Mx', 'C4 Mx', 'Oz Mx', 'C4 Ax',
            'C3 Ax', 'LOC Ax', 'Fz Mx', 'LOC A2', 'Pz Ax', 'Pz Mx','F4 M1', 'F3 M2',
            'O1 M2', 'C3 M2', 'P3 M2',  'P4 M1','O2 M1',  'E2 M1', 'E1 M2', 'LOC A2', 'C4 M1',
            'EMG sm', 'ECG', 'Marker', 'Ubatt'
        ]
        
        
        self.channel_types = {
            'analog': [
                'ROC Ax', 'Cz Mx', 'E1 Mx', 'ROC A1', 'Oz Ax', 'Fz Ax', 'E2 Mx', 'Cz Ax',
                'C3 A2', 'O2 A1', 'O1 A2', 'C4 A1', 'C3 Mx', 'C4 Mx', 'Oz Mx', 'C4 Ax',
                'C3 Ax', 'LOC Ax', 'Fz Mx', 'LOC A2', 'Pz Ax', 'Pz Mx','F4 M1', 'F3 M2',
                'O1 M2', 'C3 M2', 'P3 M2',  'P4 M1','O2 M1',  'E2 M1', 'E1 M2', 'LOC A2',
                'C4 M1', 'EMG sm', 'ECG'
            ],
            'digital': ['Marker','Ubatt']
        }
                
        self.channel_groups = {
            'eeg_eog': ['ROC Ax', 'Cz Mx', 'E1 Mx', 'ROC A1', 'Oz Ax', 'Fz Ax', 'E2 Mx', 'Cz Ax',
                'C3 A2', 'O2 A1', 'O1 A2', 'C4 A1', 'C3 Mx', 'C4 Mx', 'Oz Mx', 'C4 Ax',
                'C3 Ax', 'LOC Ax', 'Fz Mx', 'LOC A2', 'Pz Ax', 'Pz Mx','F4 M1', 'F3 M2',
                'O1 M2', 'C3 M2', 'P3 M2',  'P4 M1','O2 M1',  'E2 M1', 'E1 M2', 'LOC A2',
                'C4 M1'],
            'emg': ['EMG sm'],
            'ecg': ['ECG']
        }
        
        
        self.file_extensions = {
            'psg_ext': '*/*.edf',
            'ann_ext': '*/*_score.csv'
        }
        
    def dataset_paths(self):
        return [
            "subjects",
            "subjects"
        ]
    
    def ann_parse(self, ann_fname: str):
        """
        Parse FDCSR annotation files with custom datetime handling.
        """

        # recordings only contain information about month and day (not year),
        # recordings took place between 2000 and 2016, default value 1900 to distinguish
        study_start_datetime = datetime(year=1900,month=1,day=1)     

        epoch_duration = 30  # FDCSR uses 30-second epochs   

        ann_df = pd.read_csv(ann_fname,sep=',',header=0)
    
        ann_Startdatetime = study_start_datetime + timedelta(hours=ann_df.iloc[0]['labtime'])

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

        lights_off = ann_df.loc[ann_df['stage'] == 8, 'labtime'].values
        if len(lights_off) == 1:
            lights_off = (study_start_datetime + timedelta(hours=lights_off[0])).time()
        elif len(lights_off) > 1:
            raise Exception(f"Expected exactly one 'Lights Off' event, found {len(lights_off)}")
        else:
            lights_off = None
        
        lights_on = ann_df.loc[ann_df['stage'] == 9, 'labtime'].values
        if len(lights_on) == 1:
            lights_on = (study_start_datetime + timedelta(hours=lights_on[0])).time()
        elif len(lights_on) > 1:
            raise Exception(f"Expected exactly one 'Lights On' event, found {len(lights_on)}")
        else:
            lights_on = None

        return ann_stage_events, ann_Startdatetime, lights_off, lights_on

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
                print(f"{len(unscored_files)} unscored files were found.")
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

        print(f"Processing: {edf_filename}") #, EDF: {edf_filepath}, Score: {score_filepath}")

        self._extract_matching_scores(edf_info, score_filepath, output_filepath, edf_filepath)


    def _extract_matching_scores(self, edf_info, score_filepath, output_filepath, edf_filepath):
        """Extract sleep scores matching the EDF file timing."""

        found_time_mismatch = False

        hours = timedelta(hours=((edf_info["start labtime"]/24) - (edf_info["start labtime"]//24))*24) # remove full days
        start_lab_time = timedelta(seconds=round(hours.total_seconds()))
        
        psg_f = pyedflib.EdfReader(str(edf_filepath))
        start_time_edf = psg_f.getStartdatetime()
        duration_edf = timedelta(seconds=psg_f.getFileDuration())

        duration = timedelta(hours=edf_info["last labtime"] - edf_info["start labtime"])

        if start_time_edf.time() != (datetime.min + start_lab_time).time():
            duration_epoched = timedelta(seconds=np.floor(duration_edf.total_seconds()/30)*30)  # round to full epochs (like in psg file)
            if (duration_epoched - duration).total_seconds() <= 90:
                found_time_mismatch = True
                print(start_time_edf.time(), (datetime.min + start_lab_time).time(), 'found start time mismatch')
            else:
                raise Exception(edf_filepath,(duration_epoched - duration).total_seconds() )
        else:
            print(start_time_edf.time(), (datetime.min + start_lab_time).time(), 'start time is matching')

        # Get timing information
        start_labtime = round(edf_info["start labtime"], 6)
        last_line_time = round(edf_info["last labtime"], 6)
        n_lines = int(edf_info["last line"]) - 1

        # print(f"  Time range: {start_labtime} to {last_line_time} ({n_lines} lines)")

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

        # print(f"  Score indices: {idx_start} to {idx_stop}")

        # Validate the extraction
        assert sleep_scores_df.iloc[idx_stop]["labtime"] == sleep_scores_df.iloc[idx_start + n_lines]["labtime"], f"Time mismatch: expected {sleep_scores_df.iloc[idx_start + n_lines]['labtime']}, got {sleep_scores_df.iloc[idx_stop]['labtime']}"

        # Extract and save matching scores
        matching_scores = sleep_scores_df.loc[idx_start:idx_stop].copy()
        if found_time_mismatch:
            matching_scores["labtime"] = matching_scores["labtime"] + 1 # add 1 hour to labtime to match the EDF timing

        matching_scores.to_csv(output_filepath, index=False)
