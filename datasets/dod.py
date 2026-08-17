import os
import h5py
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from datasets.base import BaseDataset
from datasets.registry import register_dataset

class DOD(BaseDataset):
    """DOD (Dreem Open Dataset) base class."""

    def __init__(self, dataset_name: str, description: str):
        super().__init__(dataset_name, description)
        self._file_handler = None  # DOD uses custom H5 handling directly implemented here

    def dataset_paths(self):
        return [
            '',
            ''
        ]

    def get_channels(self, logger, filepath):
        """Extract channel names from file."""
        try:
            channel_names = []
            with h5py.File(filepath, "r") as f:

                def visitor(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        channel_names.append(name)

                f["signals"].visititems(visitor)    # only get channels in signals subfolder
            return channel_names
        except Exception as e:
            logger.error(f"Error during channel extraction: {e}")
            raise

    def read_signal(self, logger, filepath, channel):
        """Read signal from file for specific channel."""
        try:
            with h5py.File(filepath, "r") as f:
                if channel not in f["signals"]:
                    return None

                signal = f["signals"][channel][:]
                return signal
        except Exception as e:
            logger.error(f"Error during signal extraction: {e}")
            raise

    def get_file_info(self, logger, filepath):
        """Get start datetime and file duration."""
        try:
            with h5py.File(filepath, "r") as f:

                start_time = f.attrs.get("start_time")
                if start_time is None:
                    logger.warning(f"No start_time attribute found, defaulting to datetime(1970, 1, 1)")
                    start_datetime = datetime(1970, 1, 1)
                else:
                    start_datetime = datetime.fromtimestamp(start_time)

                file_duration = f.attrs.get("duration")

        except Exception as e:
            logger.error(f"Error during file info retrieval: {e}")
            raise

        return {"start_datetime": start_datetime, "file_duration": file_duration}

    def get_signal_data(self, logger, filepath, channel):
        """Get complete signal information for specific channel."""
        try:
            with h5py.File(filepath, "r") as f:
                dataset = f["signals"][channel]
                signal = dataset[:]

                unit = dataset.parent.attrs.get("unit", "n/a")
                if isinstance(unit, bytes):
                    unit = unit.decode("utf-8")

                sampling_rate = dataset.parent.attrs.get("fs")

                return {
                    "signal": signal,
                    "sampling_rate": sampling_rate,
                    "unit": unit
                }
        except Exception as e:
            logger.error(f"Error during data retrieval: {e}")
            raise

    def ann_parse(self, ann_fname: str):
        """
        DOD datasets don't use separate annotation files.
        Annotations are embedded in H5 files.
        """
        ann_stage_events = []
        epoch_duration = 30  # DOD datasets use 30-second epochs
        with h5py.File(ann_fname, "r") as f:
            stages = f["hypnogram"][:]
            for i, stage in enumerate(stages):
                onset = i * epoch_duration
                duration = epoch_duration
                ann = {
                    'Stage': stage,
                    'Start': onset,
                    'Duration': duration
                }
                ann_stage_events.append(ann)

        return ann_stage_events, None, None, None

@register_dataset("DOD-O")
class DODO(DOD):
    """DOD-O (Dreem Open Dataset - Obstructive) dataset."""
    
    def __init__(self):
        super().__init__("DOD-O","DOD-O - Dreem Open Dataset - Obstructive")

    def _setup_dataset_config(self):
        self.ann2label =  {
            0: "W",  # Wake
            1: "N1",   # NREM Stage 1
            2: "N2",   # NREM Stage 2
            3: "N3",   # NREM Stage 3
            4: "REM",   # REM sleep
            -1: "UNK",   # Unscored
        }
        
        
        self.intra_dataset_mapping = {
            "O1_M1": ["eeg/O1_M1"],
            "C4_M1": ["eeg/C4_M1"],
            "F3_O1": ["eeg/F3_O1"],
            "O1_M2": ["eeg/O1_M2"],
            "F4_O2": ["eeg/F4_O2"],
            "F3_F4": ["eeg/F3_F4"],
            "C3_M2": ["eeg/C3_M2"],
            "F3_M2": ["eeg/F3_M2"],
            "O2_M1": ["eeg/O2_M1"],
            "EOG1": ["eog/EOG1"],
            "EOG2": ["eog/EOG2"],
            "EMG": ["emg/EMG"],
            "ECG": ["emg/ECG"]
        }

        self.inter_dataset_mapping= {
            "C3_M2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "C4_M1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "F3_F4": self.Mapping(self.TTRef.F3, self.TTRef.F4),
            "F3_M2": self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            "F3_O1": self.Mapping(self.TTRef.F3, self.TTRef.O1),
            "F4_O2": self.Mapping(self.TTRef.F4, self.TTRef.O2),
            "O1_M2": self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            "O2_M1": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "O1_M1": self.Mapping(self.TTRef.O1, self.TTRef.LPA),
            "EOG1": self.Mapping(self.TTRef.EL, self.TTRef.RPA),
            "EOG2": self.Mapping(self.TTRef.ER, self.TTRef.RPA),
            "ECG": self.Mapping(self.TTRef.ECG, None),
            "EMG": self.Mapping(self.TTRef.EMG_CHIN, None)
        }
        
        
        self.channel_names = [
            'eeg/C4_M1', 'eeg/F3_O1', 'eeg/O1_M2',
            'eeg/F4_O2', 'eeg/F3_F4', 'eeg/C3_M2', 'eeg/F3_M2',
            'eeg/O2_M1',
            'eog/EOG1', 'eog/EOG2',
            'emg/EMG',
            'emg/ECG'
        ]
        
        
        self.channel_types = {
            'analog': [
                'eeg/O2_M1', 'emg/ECG', 'emg/EMG', 'eeg/C4_M1', 
                'eeg/F3_O1', 'eeg/O1_M2', 'eog/EOG1', 'eeg/F4_O2', 
                'eeg/F3_F4', 'eog/EOG2', 'eeg/C3_M2', 'eeg/F3_M2',
            ],
            'digital': []
        }
    
        
        self.channel_groups = {
            'eeg_eog': ['eeg/C4_M1', 'eeg/F3_O1', 'eeg/O1_M2', 'eog/EOG1', 'eeg/F4_O2', 'eeg/F3_F4', 'eog/EOG2', 'eeg/C3_M2', 'eeg/F3_M2', 'eeg/O2_M1'],
            'emg': ['emg/EMG'],
            'ecg': ['emg/ECG']
        }

                
        self.file_extensions = {
            'psg_ext': '*.h5',
            'ann_ext': '*.h5'  # Annotations are embedded in data H5 files
        }
        

@register_dataset("DOD-H")
class DODH(DOD):
    """DOD-H (Dreem Open Dataset - Healthy) dataset."""
    
    def __init__(self):
        super().__init__("DOD-H","DOD-H - Dreem Open Dataset - Healthy")

    def _setup_dataset_config(self):
        self.ann2label = {
            0: "W",   # Wake
            1: "N1",   # NREM Stage 1
            2: "N2",   # NREM Stage 2
            3: "N3",   # NREM Stage 3
            4: "REM",   # REM sleep
            -1: "UNK",   # Unscored
        }
        
        self.intra_dataset_mapping = {
            "F4_O2": ["eeg/F4_O2"],
            "FP2_F4": ["eeg/FP2_F4"],
            "F3_M2": ["eeg/F3_M2"],
            "F3_O1": ["eeg/F3_O1"],
            "F4_M1": ["eeg/F4_M1"],
            "C3_M2": ["eeg/C3_M2"],
            "FP1_M2": ["eeg/FP1_M2"],
            "F3_F4": ["eeg/F3_F4"],
            "FP2_O2": ["eeg/FP2_O2"],
            "FP1_O1": ["eeg/FP1_O1"],
            "FP2_M1": ["eeg/FP2_M1"],
            "FP1_F3": ["eeg/FP1_F3"],
            "EOG1": ["eog/EOG1"],
            "EOG2": ["eog/EOG2"],
            "EMG": ["emg/EMG"],
            "ECG": ["emg/ECG"]
        }

        self.inter_dataset_mapping= {
            "C3_M2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "F4_M1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            "FP2_F4": self.Mapping(self.TTRef.Fp2, self.TTRef.F4),
            "FP1_M2": self.Mapping(self.TTRef.Fp1, self.TTRef.RPA),
            "F3_F4": self.Mapping(self.TTRef.F3, self.TTRef.F4),
            "FP2_O2": self.Mapping(self.TTRef.Fp2, self.TTRef.O2),
            "FP1_O1": self.Mapping(self.TTRef.Fp1, self.TTRef.O1),
            "FP2_M1": self.Mapping(self.TTRef.Fp2, self.TTRef.LPA),
            "FP1_F3": self.Mapping(self.TTRef.Fp1, self.TTRef.F3),
            "F3_M2": self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            "F3_O1": self.Mapping(self.TTRef.F3, self.TTRef.O1),
            "F4_O2": self.Mapping(self.TTRef.F4, self.TTRef.O2),
            "EOG1": self.Mapping(self.TTRef.EL, None),
            "EOG2": self.Mapping(self.TTRef.ER, None),
            "ECG": self.Mapping(self.TTRef.ECG, None),
            "EMG": self.Mapping(self.TTRef.EMG_CHIN, None)
        }
        
        self.channel_names = [
            'eeg/F4_O2', 'eeg/FP2_F4', 'eeg/F3_M2', 'eeg/F3_O1',
            'eeg/F4_M1', 'eeg/C3_M2', 'eeg/FP1_M2', 'eeg/F3_F4',
            'eeg/FP2_O2', 'eeg/FP1_O1', 'eeg/FP2_M1', 'eeg/FP1_F3',
            'eog/EOG1', 'eog/EOG2', 'emg/EMG', 'emg/ECG'
        ]
        
        self.channel_types = {
            'analog': [
                'eeg/F4_O2', 'eeg/FP2_F4', 'emg/EMG', 'eeg/F3_M2',
                'emg/ECG', 'eeg/F3_O1', 'eeg/F4_M1', 'eeg/C3_M2',
                'eeg/FP1_M2', 'eeg/F3_F4', 'eeg/FP2_O2', 'eeg/FP1_O1',
                'eeg/FP2_M1', 'eog/EOG2', 'eeg/FP1_F3', 'eog/EOG1'
            ],
            'digital': []
        }
        
        self.channel_groups = {
            'eeg_eog': ['eeg/F4_O2', 'eeg/FP2_F4', 'eeg/F3_M2', 'eeg/F3_O1', 'eeg/F4_M1', 'eeg/C3_M2', 'eeg/FP1_M2', 'eeg/F3_F4', 'eeg/FP2_O2', 'eeg/FP1_O1', 'eeg/FP2_M1', 'eeg/FP1_F3', 'eog/EOG2', 'eog/EOG1'],
            'emg': ['emg/EMG'],
            'ecg': ['emg/ECG']
        }
        
        self.file_extensions = {
            'psg_ext': '*.h5',
            'ann_ext': '*.h5'  # Annotations are embedded in data H5 files
        }
