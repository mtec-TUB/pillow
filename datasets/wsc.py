import os
import pandas as pd
from datetime import datetime
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("WSC")
class WSC(BaseDataset):
    """WSC (Wisconsin Sleep Cohort) dataset."""
    
    def __init__(self):
        super().__init__("WSC","WSC - Wisconsin Sleep Cohort")
        self.has_front_alignment = True
        self.has_end_alignment = True

    def _setup_dataset_config(self):
        # see wsc-scoring-annotation-documentation.xlsx
        self.ann2label = {
            # String-based labels
            "STAGE - W": "W",
            "STAGE - N1": "N1",
            "STAGE - N2": "N2",
            "STAGE - N3": "N3",
            "STAGE - N4": "N3",  # Follow AASM Manual
            "STAGE - R": "REM",
            "STAGE - NO STAGE": "UNK",
            "STAGE - MVT": "MOVE",
            # Numeric labels
            0: "W",  # Wake
            1: "N1",  # NREM Stage 1
            2: "N2",  # NREM Stage 2
            3: "N3",  # NREM Stage 3
            4: "N3",  # NREM Stage 4 (Follow AASM Manual)
            5: "REM",  # REM
            6: "MOVE",   # Movement
            7: "UNK",  # Unscored
        }

        self.inter_dataset_mapping = {
            'thorax': self.Mapping(self.TTRef.THORACIC, None),
            'abdomen': self.Mapping(self.TTRef.ABDOMINAL, None),
            'C3_AVG': self.Mapping(self.TTRef.C3, None),
            'C4_AVG': self.Mapping(self.TTRef.C4, None),
            'nasalflow': self.Mapping(self.TTRef.AIRFLOW, None),
            'O1_M2': self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            'C4_M1': self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            'O1_M1': self.Mapping(self.TTRef.O1, self.TTRef.LPA),
            'F3_AVG': self.Mapping(self.TTRef.F3, None),
            'Pz_M2': self.Mapping(self.TTRef.Pz, self.TTRef.RPA),
            'ECG': self.Mapping(self.TTRef.ECG, None),
            'C4_M2': self.Mapping(self.TTRef.C4, self.TTRef.RPA),
            'Fz_AVG': self.Mapping(self.TTRef.Fz, None),
            'E1': self.Mapping(self.TTRef.EL, None),
            'Pz_Cz': self.Mapping(self.TTRef.Pz, self.TTRef.Cz),
            'F3_M1': self.Mapping(self.TTRef.F3, self.TTRef.LPA),
            'F3_M2': self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            'snore': self.Mapping(self.TTRef.SNORE, None),
            'C3_M1': self.Mapping(self.TTRef.C3, self.TTRef.LPA),
            'chin': self.Mapping(self.TTRef.EMG_CHIN, None),
            'cchin_l': self.Mapping(self.TTRef.EMG_LCHIN, None),
            'cchin_r': self.Mapping(self.TTRef.EMG_RCHIN, None),
            'Cz_M1': self.Mapping(self.TTRef.Cz, self.TTRef.LPA),
            'E2': self.Mapping(self.TTRef.ER, None),
            'F4_M2': self.Mapping(self.TTRef.F4, self.TTRef.RPA),
            'Pz_AVG': self.Mapping(self.TTRef.Pz, None),
            'Cz_M2': self.Mapping(self.TTRef.Cz, self.TTRef.RPA),
            'C3_M2': self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            'position': self.Mapping(self.TTRef.POSITION, None),
            'O2_M2': self.Mapping(self.TTRef.O2, self.TTRef.RPA),
            'O2_M1': self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            'spo2': self.Mapping(self.TTRef.SPO2, None),
            'Fz_M1': self.Mapping(self.TTRef.Fz, self.TTRef.LPA),
            'F4_M1': self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            'C4_AVG': self.Mapping(self.TTRef.C4, None),
            'F4_AVG': self.Mapping(self.TTRef.F4, None),
            'O1_AVG': self.Mapping(self.TTRef.O1, None),
            'Cz_AVG': self.Mapping(self.TTRef.Cz, None),
            'Fz_M2': self.Mapping(self.TTRef.Fz, self.TTRef.RPA),
            'lleg_r': self.Mapping(self.TTRef.EMG_LLEG, self.TTRef.EMG_RLEG),
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

    def get_file_identifier(self, psg_fname=None, ann_fname=None):
        psg_id, ann_id = None, None

        if psg_fname:
            psg_ext = self.file_extensions['psg_ext'].split('*')[-1]
            psg_id = psg_fname.split(psg_ext)[0]

        if ann_fname:
            ann_ext = self.file_extensions['ann_ext'].split('*')[-1]
            ann_ext2 = self.file_extensions['ann_ext2'].split('*')[-1]
            if ann_fname.endswith(ann_ext):
                ann_id = ann_fname.split(ann_ext)[0]
            else:
                ann_id = ann_fname.split(ann_ext2)[0]
        
        return psg_id, ann_id
    
    def dataset_paths(self):
        return [
            'polysomnography',
            'polysomnography'
        ]
    
    def ann_parse(self, ann_fname: str):
        """
        Parse WSC CSV annotation files.
        
        Args:
            ann_fname: Path to CSV annotation file
            
        Returns:
            Tuple of (sleep_stage_events, start_datetime)
        """
        ann_stage_events = []
        
        epoch_duration = 30  # WSC uses 30-second epochs
        lights_df = None

        ann_Startdatetime = None

        if 'stg.txt' in ann_fname:
            # Check if there is a header row or not and read
            if open(ann_fname).readline().startswith("Epoch"):
                data = pd.read_csv(ann_fname, sep="\t", header=0, names=['Epoch', 'Stage', 'CAST_Stage'])
            else:
                data = pd.read_csv(ann_fname, sep="\t", header=None, names=['Epoch', 'Stage', 'CAST_Stage'])

            for i,row in data.iterrows():
                ann_stage_events.append({'Stage': row['Stage'],
                                            'Start': i * epoch_duration,
                                            'Duration': epoch_duration})
                
            log_file = ann_fname.replace('.stg.txt', '.log.txt')
            if os.path.exists(log_file):
                log_df = pd.read_csv(log_file, sep="\t", names=['Timestamp', 'Info'], usecols=[1,2], header=None)
                first_epoch_idx = log_df[log_df['Info'] == 'Recording Started'].index
                if len(first_epoch_idx) > 1:
                    if "wsc-visit1-67336-nsrr" in ann_fname:
                       first_epoch_idx = first_epoch_idx[1]
                    elif "wsc-visit1-80964-nsrr" in ann_fname:
                        first_epoch_idx = first_epoch_idx[0]
                    else:
                        first_epoch_idx = first_epoch_idx[-1]  # Take last occurence (verified in multiple WSC log files)
                elif len(first_epoch_idx) == 0:
                    first_epoch_idx = 0
                else:
                    first_epoch_idx = int(first_epoch_idx[0])
                
                ann_Startdatetime = datetime.strptime(log_df.iloc[first_epoch_idx]['Timestamp'].split(' ')[0], '%H:%M:%S')
                ann_Startdatetime = datetime.combine(datetime(1985, 1, 1).date(), ann_Startdatetime.time())

                lights_df = log_df
            
        elif 'allscore.txt' in ann_fname:
            df = pd.read_csv(ann_fname, sep="\t", names=['Timestamp','Info'], encoding='latin',na_filter=False)
            
            start_idx = df[df['Info'] == 'START RECORDING'].index
            if len(start_idx) > 1:
                raise Exception("Multiple START RECORDING entries found in annotation file.")
            elif len(start_idx) == 0:
                # Fallback to STAGE - NO STAGE if START RECORDING is not found
                fallback_idx = df[df['Info'] == 'STAGE - NO STAGE'].index
                if len(fallback_idx) == 0:
                    raise Exception("Neither START RECORDING nor STAGE - NO STAGE found in annotation file.")
                start_idx = int(fallback_idx[0])
            else:
                start_idx = int(start_idx[0])
                
            df = df.iloc[start_idx:].reset_index()
            
            # Filter for sleep stage events
            df_scoring = df[df['Info'].str.contains("STAGE")].reset_index()

            
            for i,row in df_scoring.iterrows():
                stage = row['Info']
                start_time = datetime.strptime(row['Timestamp'],'%H:%M:%S.%f')
                if ann_Startdatetime == None:
                    ann_Startdatetime = start_time

                start_sec = int((start_time - ann_Startdatetime).seconds)
                
                # for all but the last event, duration is the time until the next event; for the last event, we can set a default duration (e.g., 30s)
                if i+1 != len(df_scoring):
                    duration = int((datetime.strptime(df_scoring.iloc[i+1]['Timestamp'],'%H:%M:%S.%f') - start_time).seconds)
                else:
                    duration = epoch_duration
                
                ann_stage_events.append({'Stage': stage,
                                                'Start': start_sec,
                                                'Duration': duration})
                
                lights_df = df

            # Adapt start date to be same with signal start date (always starting at 1985-01-01) to avoid issues with datetime subtraction
            if (datetime.strptime(df.iloc[start_idx]['Timestamp'],'%H:%M:%S.%f') - ann_Startdatetime).total_seconds() > 12 * 3600:
                ann_Startdatetime = datetime.combine(datetime(1985, 1, 2).date(), ann_Startdatetime.time())
            else:
                ann_Startdatetime = datetime.combine(datetime(1985, 1, 1).date(), ann_Startdatetime.time())

        lights_off, lights_on = None, None
        if lights_df is not None:
            lights_off_values = lights_df.loc[lights_df['Info'].str.contains(r'LIGHTS? OUT', case=False, na=False), 'Timestamp'].values
            if len(lights_off_values) == 1:
                timestamp = lights_off_values[0]
            elif len(lights_off_values) > 1:
                timestamp = lights_off_values[-1]  # take last occurence
            try:
                lights_off = datetime.strptime(timestamp.split(' ')[0], '%H:%M:%S').time()
            except ValueError:
                try:
                    lights_off = datetime.strptime(timestamp.split(' ')[0], '%H:%M:%S.%f').time()
                except ValueError:
                    pass  # In some cases, the timestamp might be malformed, so we skip parsing it

            
            lights_on_values = lights_df.loc[lights_df['Info'].str.contains(r'LIGHT?S ON', case=False, na=False), 'Timestamp'].values
            if len(lights_on_values) == 1:
                timestamp = lights_on_values[0]

            elif len(lights_on_values) > 1:
                timestamp = lights_on_values[-1]  # take last occurence
            try:
                lights_on = datetime.strptime(timestamp.split(' ')[0], '%H:%M:%S').time()
            except ValueError:
                try:
                    lights_on = datetime.strptime(timestamp.split(' ')[0], '%H:%M:%S.%f').time()
                except ValueError:
                    pass  # In some cases, the timestamp might be malformed, so we skip parsing it

        return ann_stage_events, ann_Startdatetime, lights_off, lights_on

