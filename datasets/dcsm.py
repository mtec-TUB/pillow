import os
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("DCSM")
class DCSM(BaseDataset):
    """DCSM (Danish Center of Sleep Medicine) dataset."""
    
    def __init__(self):
        super().__init__("DCSM","DCSM - Danish Center of Sleep Medicine", keep_folder_structure = False)
    
    def _setup_dataset_config(self):
        self.ann2label = {
            "W": 0,      # Wake
            "N1": 1,     # NREM Stage 1
            "N2": 2,     # NREM Stage 2  
            "N3": 3,     # NREM Stage 3
            "REM": 4,    # REM sleep
        }
        

        self.channel_names = ['E1-M2', 'E2-M2', 'F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1',
                'LAT', 'RAT', 'SNORE', 'NASAL', 'THORAX', 'ABDOMEN', 'SPO2', 'CHIN', 'ECG-II']
        
        
        self.channel_types =  {
            'analog': ['E1-M2', 'E2-M2', 'F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1',
                'CHIN', 'LAT', 'RAT', 'SNORE', 'NASAL', 'THORAX', 'ABDOMEN', 'ECG-II'
            ],
            'digital': ['SPO2']
        }
        
        self.channel_groups = {
            'eeg_eog': ['E1-M2', 'E2-M2', 'F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'],
            'emg': ['CHIN', 'LAT', 'RAT'],
            'ecg': ['ECG-II'],
            'thoraco_abdo_resp': ['THORAX', 'ABDOMEN'],
            'nasal_pressure': ['NASAL'],
            'snoring': ['SNORE']
        }

        self.inter_dataset_mapping = {
            "E1-M2": self.Mapping(self.TTRef.EL, self.TTRef.RPA),
            "E2-M2": self.Mapping(self.TTRef.ER, self.TTRef.RPA),
            "C3-M2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "C4-M1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "F3-M2": self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            "F4-M1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            "O1-M2": self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            "O2-M1": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "CHIN": self.Mapping(self.TTRef.EMG_CHIN, None),
            "THORAX": self.Mapping(self.TTRef.THORACIC, None),
            "ABDOMEN": self.Mapping(self.TTRef.ABDOMINAL, None),
            "SNORE": self.Mapping(self.TTRef.SNORE, None),
            "ECG-II": self.Mapping(self.TTRef.ECG, None),
            "LAT": self.Mapping(self.TTRef.EMG_LLEG, None),
            "RAT": self.Mapping(self.TTRef.EMG_RLEG, None),
            "SPO2": self.Mapping(self.TTRef.SPO2, None),
        }     

        self.file_extensions = {
            'psg_ext': '*_psg.edf',
            'ann_ext': '*_hypnogram.ids'
        }

    def dataset_paths(self) -> Tuple[str, str]:
        """
        DCSM dataset paths.
        """
        data_dir = "DCSM - Danish Center of Sleep Medicine/edfs"
        ann_dir = "DCSM - Danish Center of Sleep Medicine/annot"
        return data_dir, ann_dir
    

    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        Parse DCSM .ids annotation files.
        
        DCSM uses CSV format with columns: start, duration, stage
        No header in the CSV file.
        
        Args:
            ann_fname: Path to .ids annotation file
            
        Returns:
            Tuple of (sleep_stage_events, start_datetime)
        """
        ann_stage_events = []
        
        # Read CSV file without header
        ann_df = pd.read_csv(ann_fname, sep=',', names=['start', 'duration', 'stage'])
        
        for i, row in ann_df.iterrows():
            start = row['start']
            duration = row['duration']
            stage = row['stage']
            
            ann_stage_events.append({
                'Stage': stage,
                'Start': start,
                'Duration': duration
            })
        
        # DCSM .ids files don't contain start datetime information
        # Return None for start_datetime
        return ann_stage_events, None
    
    def preprocess(self, data_dir, ann_dir, output_dir):
        print("\n DCSM files originally are stored in an unsupported way and therefor need to be preprocessed/resorted ... \n \
              This will not modify the original file content")
        
        execute_preprocess = input("Do you want to perform the preprocessing now? (Y/N) ")
        
        if str(execute_preprocess).lower() == "y":
            organizer = DCSMFileOrganizer(os.path.split(data_dir)[0])
            organizer.organize_files()

            if str(input("Do you want to continue with processing now? (Y/N) ")).lower() == "n":
                return False
        return True
    
class DCSMFileOrganizer:
    """Handles reorganization of DCSM dataset files."""

    def __init__(self, base_dir: str):
        """
        Initialize the organizer.

        Args:
            base_dir: Path to DCSM dataset root directory
        """
        self.base_dir = Path(base_dir)
        self.target_dirs = {
            "annot": self.base_dir / "annot",
            "edfs": self.base_dir / "edfs",
            "h5s": self.base_dir / "h5s",
        }

        # File mapping: source filename -> target directory
        self.file_mapping = {
            "hypnogram.ids": "annot",
            "psg.edf": "edfs",
            "psg.h5": "h5s",
        }

    def organize_files(self):
        """Reorganize all files from subject subfolders."""
        # Create target directories
        self._create_target_directories()

        # Process each subject subfolder
        subject_folders = [f for f in self.base_dir.iterdir() if f.is_dir() and f.name not in ["annot", "edfs", "h5s"]]

        print(f"Found {len(subject_folders)} subject folders to process...")

        stats = {"moved": 0, "skipped": 0, "errors": 0}

        for subject_folder in subject_folders:
            try:
                self._process_subject_folder(subject_folder, stats)
            except Exception as e:
                print(f"ERROR processing {subject_folder.name}: {e}")
                stats["errors"] += 1

        self._print_summary(stats)

    def _create_target_directories(self):
        """Create the target directory structure."""
        for dir_name, dir_path in self.target_dirs.items():
            dir_path.mkdir(exist_ok=True)

    def _process_subject_folder(self, subject_folder: Path, stats: Dict[str, int]):
        """Process files in a single subject folder."""
        subject_id = subject_folder.name
        print(f"\nProcessing subject: {subject_id}")

        for source_filename, target_dir_name in self.file_mapping.items():
            source_file = subject_folder / source_filename

            if source_file.exists():
                # Create new filename with subject ID
                if source_filename == "hypnogram.ids":
                    new_filename = f"{subject_id}_hypnogram.ids"
                elif source_filename == "psg.edf":
                    new_filename = f"{subject_id}_psg.edf"
                elif source_filename == "psg.h5":
                    new_filename = f"{subject_id}_psg.h5"

                target_file = self.target_dirs[target_dir_name] / new_filename

                # Copy file (preserve original)
                try:
                    shutil.copy2(source_file, target_file)
                    print(f" Copied {source_filename} -> {new_filename}")
                    stats["moved"] += 1
                except Exception as e:
                    print(f" Failed to copy {source_filename}: {e}")
                    stats["errors"] += 1
            else:
                print(f" - Missing: {source_filename}")
                stats["skipped"] += 1

    def _print_summary(self, stats: Dict[str, int]):
        """Print processing summary."""
        print("\n" + "=" * 50)
        print("DCSM File Organization Summary")
        print("=" * 50)
        print(f"Files successfully moved: {stats['moved']}")
        print(f"Files skipped (missing): {stats['skipped']}")
        print(f"Errors encountered: {stats['errors']}")
