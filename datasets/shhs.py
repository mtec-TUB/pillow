"""
SHHS - Sleep Heart Health Study
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("SHHS")
class SHHS(BaseDataset):
    """SHHS - Sleep Heart Health Study"""

    def __init__(self):
        super().__init__("SHHS","SHHS - Sleep Heart Health Study")

    def _setup_dataset_config(self):
        self.ann2label = {
            "Wake": "W",
            "Stage 1 sleep": "N1",
            "Stage 2 sleep": "N2",
            "Stage 3 sleep": "N3", #    
            "Stage 4 sleep": "N3", # Follow AASM Manual
            "REM sleep": "REM",
            "Unscored": "UNK",
            "Movement": "MOVE"
        }
        
        self.intra_dataset_mapping = {
            'NewAirflow':['new air','NEW AIR','New AIR','New Air','NEWAIR','New A/F'],
            'OxStatus': ['OX STAT','OX stat',],
            'EEG_sec': ['EEG(sec)', 'EEG(SEC)', 'EEG sec', 'EEG2','EEG 2']
            }
        
        # https://sleepdata.org/datasets/shhs/pages/11-montage-and-sampling-rate-information-shhs2.md and https://sleepdata.org/datasets/shhs/pages/11-montage-and-sampling-rate-information-shhs1.md
        self.inter_dataset_mapping = {
            "EEG_sec": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "EEG": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "EOG(L)": self.Mapping(self.TTRef.EL, self.TTRef.Nz),
            "EOG(R)": self.Mapping(self.TTRef.ER, self.TTRef.Nz),
            "ECG": self.Mapping(self.TTRef.ECG, None),
            "EMG": self.Mapping(self.TTRef.EMG_CHIN, None),
            "Position": self.Mapping(self.TTRef.POSITION, None),
            "SaO2": self.Mapping(self.TTRef.SPO2, None),
            "SOUND": self.Mapping(self.TTRef.SNORE, None),
            "CPAP": self.Mapping(self.TTRef.CPAP, None),
            "ABDO RES": self.Mapping(self.TTRef.ABDOMINAL, None),
            "AIRFLOW": self.Mapping(self.TTRef.AIRFLOW, None),
            "THOR RES": self.Mapping(self.TTRef.THORACIC, None),
            "H.R.": self.Mapping(self.TTRef.HR, None),
        }
        
        
        self.channel_names = ['EEG', 'EEG(sec)', 'EMG', 'ABDO RES', 'new air', 'CPAP', 'POSITION', 'EOG(L)', 'H.R.', 'LEG(L)', 'EEG2', 'LIGHT', 'epms', 'NEW AIR', 
                              'New AIR', 'SOUND', 'THOR RES', 'SaO2', 'OX STAT', 'New Air', 'NASAL', 'New A/F', 'NEWAIR', 'PR', 'EEG 2', 'ECG', 'EEG(SEC)', 'EEG sec', 
                              'OX stat', 'AIRFLOW', 'EPMS', 'LEG(R)', 'EOG(R)', 'AUX']
        
        
        self.channel_types = {'analog': ['EMG', 'EEG sec', 'AIRFLOW', 'EOG(R)', 'ECG', 'EEG', 'NEW AIR', 'ABDO RES', 'New A/F', 'SOUND', 'PR', 'H.R.','AUX', 'EEG 2',  
                                         'NEWAIR', 'New Air', 'EEG2', 'new air', 'EOG(L)', 'New AIR', 'EEG(sec)', 'THOR RES', 'EEG(SEC)'],
                              'digital': ['LEG(L)', 'OX STAT', 'epms', 'EPMS','NASAL', 'LEG(R)', 'OX stat', 'SaO2', 'CPAP', 'POSITION', 'LIGHT']}

        
        self.channel_groups = {
            'eeg_eog': ['EEG','EEG2', 'EEG 2','EOG(L)','EOG(R)', 'EEG(SEC)', 'EEG sec'],
            'emg': ['EMG'],
            'ecg': ['ECG'],
            'thoraco_abdo_resp': ['ABDO RES', 'THOR RES'],
            'snoring': ['SOUND']
        }
       
        self.file_extensions = {
            'psg_ext': '**/*.edf',
            'ann_ext': '**/*-nsrr.xml'
        }

    def get_light_times(self, logger, psg_fname):
        """Light marker in shhs are not always reliable (fluctuate between on/off or are reversed calibrated)
        There are two dataset files which contain a boolean flag for each psg file that tells if the included lights signal is appropiate or not
        We use this information to decide if we compute the lightsOff/On markers based on the psg LIGHT channel or not
        If the marker is appropiate we take the first transition from Lights On to Off which is maintained for at least 30 min

        For more information see:
        - https://github.com/nsrr/shhs-data-dictionary/blob/master/variables/Sleep%20Monitoring/Polysomnography/Signal%20Quality/SHHS2/ligh.json
        - https://github.com/nsrr/shhs-data-dictionary/blob/master/variables/Sleep%20Monitoring/Polysomnography/Signal%20Quality/SHHS1/lightoff.json
        """
        lights_appropiate = False
        if 'shhs1' in psg_fname:
            overview_file = os.path.join(self.dset_dir, 'datasets','shhs1-dataset-0.21.0.csv')
            overview_table = pd.read_csv(overview_file, encoding='latin',header=0,usecols=['nsrrid','lightoff'])

            nsrrid = int(Path(psg_fname).stem.split('shhs1-')[1])
            row = overview_table[overview_table['nsrrid'] == nsrrid]
            if not row.empty:
                lights_appropiate = bool(row['lightoff'].values[0])

        elif 'shhs2' in psg_fname:
            overview_file = os.path.join(self.dset_dir, 'datasets','shhs2-dataset-0.21.0.csv')
            overview_table = pd.read_csv(overview_file, encoding='latin',header=0,usecols=['nsrrid','ligh'])

            nsrrid = int(Path(psg_fname).stem.split('shhs2-')[1])
            row = overview_table[overview_table['nsrrid'] == nsrrid]
            if not row.empty:
                lights_appropiate = bool(row['ligh'].values[0])

        if not lights_appropiate:
            logger.info(f"Light off times not appropriate for {os.path.basename(psg_fname)} according to {os.path.basename(overview_file)}.")
            return None, None

        if "LIGHT" in self._file_handler.get_channels(logger, psg_fname):

            light_data = self._file_handler.get_signal_data(logger, psg_fname, "LIGHT")

            light_signal = light_data["signal"]
            fs = light_data["sampling_rate"]
            window_samples = int(30 * 60 * fs)

            # Lights off marker: first index where LIGHT == 1 and stays 1 for 30 minutes.
            lights_off_mask = (light_signal == 1)
            valid_off_idx = np.array([], dtype=int)
            if lights_off_mask.size >= window_samples:
                rolling_sum = np.convolve(
                    lights_off_mask.astype(np.int32),
                    np.ones(window_samples, dtype=np.int32),
                    mode="valid"
                )
                valid_off_idx = np.flatnonzero(rolling_sum == window_samples)

            if valid_off_idx.size > 0:
                lights_off_idx = int(valid_off_idx[0])
                lights_off_sec = lights_off_idx / fs

                # Last occurence of light off (1)
                light_off_indices = np.flatnonzero(lights_off_mask)
                light_on_idx = int(light_off_indices[-1]) + 1
                lights_on_sec = light_on_idx / fs
            else:
                lights_off_sec = None
                lights_on_sec = None

            return lights_off_sec, lights_on_sec
        
        else:
            logger.info(f"Light channel not found in {os.path.basename(psg_fname)}. Cannot determine light on/off times")
            return None, None