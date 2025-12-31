import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("WSC")
class WSC(BaseDataset):
    """WSC (Wisconsin Sleep Cohort) dataset."""
    
    def __init__(self):
        super().__init__("WSC","WSC - Wisconsin Sleep Cohort")
    
    def _setup_dataset_config(self):
        self.ann2label = {
            # String-based labels
            "STAGE - W": 0,
            "STAGE - N1": 1,
            "STAGE - N2": 2,
            "STAGE - N3": 3,
            "STAGE - N4": 3,  # Follow AASM Manual
            "STAGE - R": 4,
            "STAGE - NO STAGE": 6,
            "STAGE - MVT": 6,
            # Numeric labels
            0: 0,  # Wake
            1: 1,  # NREM Stage 1
            2: 2,  # NREM Stage 2
            3: 3,  # NREM Stage 3
            4: 3,  # NREM Stage 4 (Follow AASM Manual)
            5: 4,  # REM
            6: 6,  # Unscored
            7: 6   # Movement
        }
        
        
        self.channel_names = ['thorax', 'C3_AVG', 'flow', 'O1_M2', 'C4_M1', 'nas_pres', 
                'O1_M1', 'F3_AVG', 'cchin_r', 'Pz_M2', 'cchin_l', 'ECG',
                'C4_M2', 'Fz_AVG', 'E1', 'Pz_Cz', 'lleg1_2', 'pap_flow', 
                'abdomen', 'oralflow', 'F3_M1', 'rleg1_2', 'F3_M2', 'snore',
                'C3_M1', 'chin', 'Cz_M1', 'E2', 'F4_M2', 'Pz_AVG', 'Cz_M2', 
                'C3_M2', 'position', 'O2_M2', 'sum', 'O2_M1', 'pap_pres',
                'nasalflow', 'spo2', 'rchin_l', 'Fz_M1', 'F4_M1', 'C4_AVG',
                'F4_AVG', 'O1_AVG', 'lleg_r', 'Cz_AVG', 'Fz_M2']
        
        
        self.channel_types = {'analog': ['thorax', 'C3_AVG', 'flow', 'O1_M2', 'C4_M1', 'nas_pres', 'O1_M1', 'F3_AVG', 'cchin_r', 'Pz_M2', 'cchin_l', 'ECG', 'C4_M2', 
                           'Fz_AVG', 'E1', 'Pz_Cz', 'lleg1_2', 'pap_flow', 'abdomen', 'oralflow', 'F3_M1', 'rleg1_2', 'F3_M2', 'snore', 'C3_M1', 'chin', 
                           'Cz_M1', 'E2', 'F4_M2', 'Pz_AVG', 'Cz_M2', 'C3_M2', 'position', 'O2_M2', 'sum', 'O2_M1', 'pap_pres', 'nasalflow', 'spo2',
                           'rchin_l', 'Fz_M1', 'F4_M1', 'C4_AVG', 'F4_AVG', 'O1_AVG', 'lleg_r', 'Cz_AVG', 'Fz_M2'], 
                'digital': []}
        
        
        self.channel_groups = {
            'eeg_eog': ['Cz_AVG', 'Fz_M2', 'Fz_M1', 'F4_M1', 'C4_AVG', 'F4_AVG', 'O1_AVG', 'O2_M1', 'O2_M2', 'Cz_M1', 'E2', 'F4_M2', 'Pz_AVG', 'Cz_M2', 'C3_M2', 'C3_M1', 'Cz_M1', 'F3_M2', 'F3_M1', 'C3_AVG', 'O1_M2', 'C4_M1', 'O1_M1', 'F3_AVG', 'Pz_M2', 'C4_M2', 'Fz_AVG', 'E1', 'Pz_Cz'],
            'emg': ['lleg_r', 'cchin_r', 'cchin_l', 'lleg1_2', 'rleg1_2', 'chin', 'rchin_l'],
            'ecg': ['ECG'],
            'thoraco_abdo_resp': ['thorax', 'abdomen', 'oralflow', 'nasalflow'],
            'nasal_pressure': ['nas_pres'],
            'snoring': ['snore']
        }
        
        
        self.file_extensions = {
            'psg_ext': '*.edf',
            'ann_ext': '*.stg.txt',
            'ann_ext2': '*.allscore.txt'  # WSC has dual annotation file types
        }

    def get_file_identifier(self, psg_fname, ann_fname):
        psg_ext = self.file_extensions['psg_ext'].split('*')[-1]
        ann_ext = self.file_extensions['ann_ext'].split('*')[-1]
        ann_ext2 = self.file_extensions['ann_ext2'].split('*')[-1]

        psg_id = psg_fname.split(psg_ext)[0]
        if ann_fname.endswith(ann_ext):
            ann_id = ann_fname.split(ann_ext)[0]
            return psg_id, ann_id
        else:
            ann_id = ann_fname.split(ann_ext2)[0]
            return psg_fname.split(psg_ext)[0], ann_fname.split(ann_ext2)[0]
    
    def dataset_paths(self) -> Tuple[str, str]:
        """
        WSC dataset paths.
        """
        data_dir = "WSC - Wisconsin Sleep Cohort/polysomnography"
        ann_dir = "WSC - Wisconsin Sleep Cohort/polysomnography"
        return data_dir, ann_dir
    
    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        Parse WSC CSV annotation files.
        
        Args:
            ann_fname: Path to CSV annotation file
            
        Returns:
            Tuple of (sleep_stage_events, start_datetime)
        """
        ann_stage_events = []
        
        epoch_duration = 30  # WSC uses 30-second epochs

        if 'stg.txt' in ann_fname:
            ann_Startdatetime = None
            data = pd.read_csv(ann_fname, sep="\t", header=0, names=['Epoch','Stage', 'CAST_Stage'])
            
            for i,row in data.iterrows():
                ann_stage_events.append({'Stage': row['Stage'],
                                            'Start': i * epoch_duration,
                                            'Duration': epoch_duration})
            
        elif 'allscore.txt' in ann_fname:
            df = pd.read_csv(ann_fname, sep="\t", names=['Timestamp','Info'], encoding='latin',na_filter=False)
            
            start_idx = df[df['Info'] == 'START RECORDING'].index
            if len(start_idx) > 1:
                raise Exception
            elif len(start_idx) == 0:
                start_idx = df[df['Info'] == 'STAGE - NO STAGE'].index[0].astype(int)
            else:
                start_idx = start_idx[0].astype(int)
                
            ann_Startdatetime = datetime.strptime(df.iloc[start_idx]['Timestamp'],'%H:%M:%S.%f')
            
            df = df.iloc[start_idx:].reset_index()
            
            
            df = df[df['Info'].str.contains("STAGE")].reset_index()

            
            for i,row in df.iterrows():
                stage = row['Info']
                start = int((datetime.strptime(row['Timestamp'],'%H:%M:%S.%f') - ann_Startdatetime).seconds)
                if i == 0 and start != 0:
                    ann_stage_events.append({'Stage': 6,
                                                    'Start': 0,
                                                    'Duration': start})
                if i+1 != len(df):
                    duration = int((datetime.strptime(df.iloc[i+1]['Timestamp'],'%H:%M:%S.%f') - datetime.strptime(row['Timestamp'],'%H:%M:%S.%f')).seconds)
                else:
                    duration = epoch_duration
                
                ann_stage_events.append({'Stage': stage,
                                                'Start': start,
                                                'Duration': duration})

        return ann_stage_events, ann_Startdatetime
    
    
    def align_end(self, logger, psg_fname: str, ann_fname: str, signals: np.ndarray,
                  labels: np.ndarray,
                  ) -> Tuple[np.ndarray, np.ndarray]:
        
        if ('allscore.txt' in ann_fname):
            if len(signals) > len(labels):
                logger.info(f"Signal (len: {len(signals)}) is shortened to match label length ({len(labels)})")
                signals = signals[:len(labels)]
            if len(labels) > len(signals):
                logger.info(f"Labels (len: {len(labels)}) are shortend to match signal ({len(signals)})")
                labels = labels[:len(signals)]

        
        if ('stg.txt' in ann_fname) and len(signals) == len(labels) + 1:
            logger.info(f"Signal (len: {len(signals)}) is shortened to match label length ({len(labels)})")
            signals = signals[:len(labels)]

        assert len(signals) == len(labels), f"Length mismatch: signal={len(signals)}, labels={len(labels)} \n TODO: implement alignment function"
        
        return signals, labels
