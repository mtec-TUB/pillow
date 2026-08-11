import pandas as pd

from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("BDSP")
class BDSP(BaseDataset):
    """BDSP dataset. Contains multiple sub-cohorts (S0001, I0002, I0003, I0004, I0006) that share
    the same file structure and annotation layout. The subset is chosen
    interactively at construction time.
    """

    SUBSETS = ("S0001", "I0002", "I0003", "I0004", "I0006")

    def __init__(self):
        self.subset = self._prompt_subset()
        super().__init__("BDSP", "bdsp", keep_folder_structure=False)
        self.has_end_alignment = True

    def _prompt_subset(self):
        choice = input(f"Select BDSP subset {self.SUBSETS}: ").strip()
        while choice not in self.SUBSETS:
            choice = input(f"Invalid choice. Select one of {self.SUBSETS}: ").strip()
        return choice

    def _setup_dataset_config(self):
        config = getattr(self, f"_config_{self.subset.lower()}")()
        for key, value in config.items():
            setattr(self, key, value)

    def dataset_paths(self):
        return ['PSG/bids', 'PSG/bids']

    def _config_i0002(self):
        return {
            "ann2label": {
                "W": "W",
                "N1": "N1",
                "N2": "N2",
                "N3": "N3",
                "R": "REM",
                "UNSCORED": "UNK",
                "L": "UNK",  # all epochs before lights off and after lights on
                "X": "UNK",
                "MT": "MOVE",
            },

            "inter_dataset_mapping": {
                "ECG": self.Mapping(self.TTRef.ECG, None),
                "SpO2": self.Mapping(self.TTRef.SPO2, None),
                "Airflow": self.Mapping(self.TTRef.AIRFLOW, None),
                "Abdominal": self.Mapping(self.TTRef.ABDOMINAL, None),
                "Thoracic": self.Mapping(self.TTRef.THORACIC, None),
                "Chin": self.Mapping(self.TTRef.EMG_CHIN, None),
                "Snore": self.Mapping(self.TTRef.SNORE, None),
                "LOC": self.Mapping(self.TTRef.EL, None),
                "ROC": self.Mapping(self.TTRef.ER, None),
                "C3-M2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
                "C4-M1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
                "O1-M2": self.Mapping(self.TTRef.O1, self.TTRef.RPA),
                "O2-M1": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
                "P3-O1": self.Mapping(self.TTRef.P3, self.TTRef.O1),
                "P4-O2": self.Mapping(self.TTRef.P4, self.TTRef.O2),
                "T3-M2": self.Mapping(self.TTRef.T7, self.TTRef.RPA),
                "T4-M1": self.Mapping(self.TTRef.T8, self.TTRef.LPA),
                "T5-M2": self.Mapping(self.TTRef.P7, self.TTRef.RPA),
                "T5-O1": self.Mapping(self.TTRef.P7, self.TTRef.O1),
                "T6-M1": self.Mapping(self.TTRef.P8, self.TTRef.LPA),
                "T6-O2": self.Mapping(self.TTRef.P8, self.TTRef.O2),
                "F7-M2": self.Mapping(self.TTRef.F7, self.TTRef.RPA),
                "F8-M1": self.Mapping(self.TTRef.F8, self.TTRef.LPA),
                "f7-f8": self.Mapping(self.TTRef.F7, self.TTRef.F8),
                "t3-t4": self.Mapping(self.TTRef.T7, self.TTRef.T8),
                "F3-M2": self.Mapping(self.TTRef.F3, self.TTRef.RPA),
                "F4-M1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
                "F3-C3": self.Mapping(self.TTRef.F3, self.TTRef.C3),
                "F4-C4": self.Mapping(self.TTRef.F4, self.TTRef.C4),
                "C3-P3": self.Mapping(self.TTRef.C3, self.TTRef.P3),
                "C4-P4": self.Mapping(self.TTRef.C4, self.TTRef.P4),
                "T3-T5": self.Mapping(self.TTRef.T7, self.TTRef.P7),
                "T4-T6": self.Mapping(self.TTRef.T8, self.TTRef.P8),
                "F7-T3": self.Mapping(self.TTRef.F7, self.TTRef.T7),
                "F8-T4": self.Mapping(self.TTRef.F8, self.TTRef.T8),
                "FP1-F7": self.Mapping(self.TTRef.Fp1, self.TTRef.F7),
                "FP2-F8": self.Mapping(self.TTRef.Fp2, self.TTRef.F8),
                "FP1-F3": self.Mapping(self.TTRef.Fp1, self.TTRef.F3),
                "FP1-FZ": self.Mapping(self.TTRef.Fp1, self.TTRef.Fz),
                "FP2-F4": self.Mapping(self.TTRef.Fp2, self.TTRef.F4),
                "F4-A2": self.Mapping(self.TTRef.F4, self.TTRef.RPA),
                "LAT": self.Mapping(self.TTRef.EMG_LLEG, None),
                "RAT": self.Mapping(self.TTRef.EMG_RLEG, None),
                "Cflow": self.Mapping(self.TTRef.CPAP, None),
            },

            "intra_dataset_mapping": {
                "ECG": ['EKG', 'ECG'],
                "SpO2": ['SaO2', 'SpO2', 'SAO2'],
                "EtCO2": ['ETCO2','EtCO2','ETco2','ETC02','EtC02'],
                "Wave EtC02": ['WAV ETCO2', 'Wave EtC02'],
                "TcCo2": ["TCCO2", "TccO2", "tcco2", "TcC02", "TcCO2"],
                "Sentec-TC": ['Sentec-TC', 'SenTec', 'SENTEC-TC','Sentec','Sentec Tc','SenTec-TC','sentec','SENTEC_TC','SenTEc','SentecTC', 'sentec-TC','Sentec TC','Sentec-Tc'],
                "Nasal Pressure": ['NPT', 'npt'],
                "Airflow": ["AIRFLOW", "Airflow"],
                "Abdominal": ["ABDOMINAL", "Abdomen", "ABD"],
                "Snore": ["SNORE", "Snore"],
                "Chin": ["CHIN", "Chin", "CHIN-1"],
                "lArm": ["Left Arm", "Left arm"],
                "rArm": ["Right Arm", "Right arm"],
                "lFlexor": ["Left Flexor", "Left flexor", "Left Flex", "Left flex"],
                "rFlexor": ["Right Flexor", "Right flexor", "Right Flex"],
                "Flexor": ["Flexor", "flexor", "flexors", "flex"],
                "Extensor": ["Extensor", "extensor", "extensors"],
                "lrFoot": ["L/R Foot", "L/R foot"],
                "lrExtensor": ["L/R extensor-foo", "L/R extendor (fo", "L/R extensor- fo"],
                "LOC": ["E1", "EOG-L"],
                "ROC": ["E2", "EOG-R"],
                "RR": ["RR", "R-R"],
                "Pleth": ["Pleth", "PLETH"],
                "Cflow": ["C-FLOW", "CFLOW", "Cflow", "CFlow", "cflow", "CFLOWresmed", "C flow", "CPAP flow"],
                "Cpress": ["C PRESS", "CPREss",],
                "Thermistor": ["THERM", "therm", "thnerm", "Term"], #"Therm" duplicate in at least one file
                "Leak": ["LEAK", "leak", "Leak"],
                "Thoracic": ["THORACIC", "THOR","CHEST", "Chest"],
                "Fino": ["Fino", "FINO", "fino", "Finometer"],
                "C3-M2": ["C3-M2", "C3-A2"],
                "C4-M1": ["C4-M1", "C4-A1"],
                "O1-M2": ["O1-M2", "O1-A2"],
                "O2-M1": ["O2-M1", "O2-A1"],
                "P3-O1": ["P3-01", "P3-O1"],
                "P4-O2": ["P4-02", "P4-O2"],
                "Rad": ["RAD", "Rad", "RAD-TC", "Rad Tc", "rad"],
                "T3-M2": ["T3-M2", "T3-A2"],
                "T4-M1": ["T4-M1", "T4-A1"],
                "T5-M2": ["T5-M2", "T5-A2"],
                "T5-O1": ["T5-O1", "T5-01"],
                "T6-M1": ["T6-M1", "T6-A1"],
                "T6-O2": ["T6-02", "T6-O2"],
                "F7-M2": ["F7-M2", "F7-A2"],
                "F8-M1": ["F8-M1", "F8-A1"],
                "FP1-F3": ["FP1-F3", "Fp1-F3"],
                "FP2-F8": ["FP2-F8", "Fp2-F8"],
                "LAT": ["LAT","LEG-L"],
                "RAT": ["RAT","LEG-R"],
                "Tidal Volume": ["Tidal Volume", "Tidol Vol", "T Vol"],
                "Deltoids": ["DELTS", "Deltoids", "DELTOIDS"],
                "Brachioradials": ["BRACH", "Brachioradials", "BRACHIORADIALIS", "BRACHIORADIALS"],
                "0V": ["0V", "OV"],
            },

            "channel_names": ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1', 'E2', 'CHIN', 'SNORE', 'NPT', 'C-FLOW', 'CHEST', 'ABDOMINAL', 'LAT', 'RAT',
                              'EKG', 'RR', 'SaO2', 'Pleth', 'Sentec-TC', 'C PRESS', 'LEAK', 'ETCO2', 'THERM', 'PTAF', 'CPAP', 'THORACIC', 'F7-A2', 'F8-A1', 'T3-A2',
                              'T4-A1', 'T5-A2', 'T6-A1', 'Flexor', 'Extensor', 'F3-C3', 'F4-C4', 'C3-P3', 'C4-P4', 'T3-T5', 'T4-T6', 'T5-O1', 'T6-02', 'P3-01', 'P4-02',
                              'SUM', 'RAD-TC', 'SenTec', 'Tidal Volume', 'F7-T3', 'F8-T4', 'FP1-F7', 'FP2-F8', 'WAV ETCO2', 'SENTEC-TC', 'CPAP vol', 'Sentec', 'EtC02',
                              'therm', 'Therm', 'BRACH', 'DELTS', 'CFLOW', 'TcCO2', 'Left flexor', 'Right flexor', 'L/R extensor-foo', 'Chin 2', 'Rad Tc', 'Fp1-F3',
                              'Fp2-F8', 'TCCO2', 'C3-A2', 'C4-A1', 'O1-A2', 'O2-A1', 'EOG-L', 'EOG-R', 'Chin', 'ECG', 'R-R', 'Leg', 'Snore', 'Airflow', 'Chest',
                              'Abdomen', 'SpO2', 'EtCO2', 'Body', 'PECTORALS', 'STERNOMASTIDS', 'INTERCOSTAL', 'Sentec Tc', 'Fino', 'VNS', 'extensors', 'flex',
                              'T3-M2', 'T4-M1', 'T5-M2', 'T6-M1', 'F7-M2', 'F8-M1', 'BRACHIORADIALIS', 'PLETH', 'Left Arm', 'Right Arm', 'L/R Foot', 'Left Flexor',
                              'Right Flexor', 'L/R extendor (fo', 'L/R extensor- fo', 'SenTec-TC','Finometer', 'C flow', 'npt', 'sentec', 'FINO', 'TccO2', 'CFlow',
                              'PTT', 'flexor', 'extensor', 'fino', 'Wave EtC02', 'P4-O2', 'T6-O2', 'FP1-FZ', 'FP2-F4', 'AIRFLOW', 'FP1-F3', 'P3-O1', 'OV', 'SENTEC_TC',
                              'SenTEc', 'CPAP flow', 'BRACHIORADIALS', 'DELTOIDS', 'flexors', 'Pressure', 'leak', 'thnerm', 'ETco2', 'CPREss', 'SEN', 'THOR', 'F4-A2',
                              'T5-01', 'DC04-Gnd', 'ABD', 'f7-f8', 't3-t4', 'Rad', 'CHIN-1', 'LEG-L', 'LEG-R', 'SAO2', 'Brachioradials', 'Deltoids', 'ETC02', 'TC',
                              'Biceps', 'SentecTC', '0V', 'tcco2', 'Tidol Vol', 'rad', 'Intentinal LEAK', 'tq', 'cflow', 'sentec-TC','Cflow', 'TcC02', 'Flow', 'Leak',
                              'Term', 'Sentec TC', 'RAD', 'Mass 1', 'Mass 2', 'Sentec-Tc', 'CFLOWresmed', 'T Vol', 'Left arm', 'Right arm', 'L/R foot', 'Left flex',
                              'Right Flex'],

            "channel_types": {'analog': ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1', 'E2', 'CHIN', 'SNORE', 'NPT', 'C-FLOW', 'CHEST', 'ABDOMINAL',
                                         'LAT', 'RAT', 'EKG', 'RR', 'Pleth', 'Sentec-TC', 'C PRESS', 'LEAK', 'ETCO2', 'THERM', 'PTAF', 'CPAP', 'THORACIC',
                                         'F7-A2', 'F8-A1', 'T3-A2', 'T4-A1', 'T5-A2', 'T6-A1', 'Flexor', 'Extensor', 'F3-C3', 'F4-C4', 'C3-P3', 'C4-P4', 'T3-T5',
                                         'T4-T6', 'T5-O1', 'T6-02', 'P3-01', 'P4-02', 'SUM', 'RAD-TC', 'SenTec', 'Tidal Volume', 'F7-T3', 'F8-T4', 'FP1-F7', 'FP2-F8',
                                         'WAV ETCO2', 'SENTEC-TC', 'CPAP vol', 'Sentec', 'EtC02', 'therm', 'Therm', 'BRACH', 'DELTS', 'CFLOW', 'TcCO2', 'Left flexor',
                                         'Right flexor', 'L/R extensor-foo', 'Chin 2', 'Rad Tc', 'Fp1-F3', 'Fp2-F8', 'TCCO2', 'C3-A2', 'C4-A1', 'O1-A2', 'O2-A1',
                                         'EOG-L', 'EOG-R', 'Chin', 'ECG', 'R-R', 'Leg', 'Snore', 'Airflow', 'Chest', 'Abdomen', 'EtCO2', 'Body', 'PECTORALS',
                                         'STERNOMASTIDS', 'INTERCOSTAL', 'Sentec Tc', 'Fino', 'VNS', 'extensors', 'flex', 'T3-M2', 'T4-M1', 'T5-M2', 'T6-M1',
                                         'F7-M2', 'F8-M1', 'BRACHIORADIALIS', 'PLETH', 'Left Arm', 'Right Arm', 'L/R Foot', 'Left Flexor', 'Right Flexor',
                                         'L/R extensor- fo', 'SenTec-TC', 'Finometer', 'C flow', 'npt', 'sentec', 'FINO', 'TccO2', 'CFlow', 'PTT', 'flexor',
                                         'extensor', 'fino', 'Wave EtC02', 'P4-O2', 'T6-O2', 'FP1-FZ', 'FP2-F4', 'AIRFLOW', 'FP1-F3', 'P3-O1', 'OV', 'SENTEC_TC',
                                         'SenTEc', 'CPAP flow', 'BRACHIORADIALS', 'DELTOIDS', 'flexors', 'Pressure', 'leak', 'thnerm', 'CPREss', 'SEN', 'THOR', 'F4-A2',
                                         'T5-01', 'DC04-Gnd', 'ABD', 'f7-f8', 't3-t4', 'Rad', 'CHIN-1', 'LEG-L', 'LEG-R', 'Deltoids', 'TC', 'Biceps', 'SentecTC', '0V',
                                         'tcco2', 'Tidol Vol', 'rad', 'Intentinal LEAK', 'tq', 'cflow', 'sentec-TC', 'Cflow', 'TcC02', 'Flow', 'Leak', 'Term',
                                         'Sentec TC', 'RAD', 'Mass 1', 'Mass 2', 'Sentec-Tc', 'CFLOWresmed', 'T Vol', 'Left arm', 'Right arm', 'L/R foot', 'Left flex',
                                         'Right Flex'],
                              'digital': ['SpO2', 'L/R extendor (fo', 'ETco2', 'Brachioradials', 'ETC02','SaO2', 'SAO2']
                              },

            "channel_groups": {'eeg_eog': ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1', 'E2','F7-A2', 'F8-A1', 'T3-A2',
                                        'T4-A1', 'T5-A2', 'T6-A1','F3-C3', 'F4-C4', 'C3-P3', 'C4-P4', 'T3-T5', 'T4-T6', 'T5-O1', 'T6-02', 'P3-01', 'P4-02',
                                        'F7-T3', 'F8-T4', 'FP1-F7', 'FP2-F8','Fp1-F3', 'Fp2-F8', 'TCCO2', 'C3-A2', 'C4-A1', 'O1-A2', 'O2-A1', 'EOG-L', 'EOG-R',
                                        'T3-M2', 'T4-M1', 'T5-M2', 'T6-M1', 'F7-M2', 'F8-M1','P4-O2', 'T6-O2', 'FP1-FZ', 'FP2-F4','FP1-F3', 'P3-O1','F4-A2',
                                        'T5-01','f7-f8', 't3-t4'],
                                'emg': ['CHIN','LAT', 'RAT','Chin 2','Chin','Leg','PECTORALS', 'STERNOMASTIDS', 'INTERCOSTAL','Left Arm', 'Right Arm', 'L/R Foot',
                                        'Left arm', 'Right arm', 'L/R foot','Flexor', 'Extensor','Left flexor', 'Right flexor','Left Flexor',
                                        'Right Flexor','flexor','flexors','L/R extensor-foo','extensors', 'flex','L/R extendor (fo', 'L/R extensor- fo', 'extensor',
                                        'CHIN-1', 'LEG-L', 'LEG-R','Left flex', 'Right Flex'],
                                'ecg': ['EKG', 'ECG'],
                                'thoraco_abdo_resp': ['CHEST', 'ABDOMINAL','THORACIC','Airflow', 'Chest', 'Abdomen','AIRFLOW', 'ABD'],
                                'nasal_pressure': ['NPT','npt'],
                                'snoring': ['SNORE','Snore']
                                },

            # len(signal_files) == len(label_files) for I0002
            "file_extensions": {'psg_ext': 'I0002/**/*-PSG_eeg.edf',
                                'ann_ext': 'I0002/**/*-psg_sleep_annotations.csv'},
        }

    def _config_i0003(self):
        # TODO: fill in real ann2label/mappings/channel names/types/groups for I0003
        return {
            "ann2label": {},
            "inter_dataset_mapping": {},
            "intra_dataset_mapping": {},
            "channel_names": [],
            "channel_types": {'analog': [], 'digital': []},
            "channel_groups": {
                'eeg_eog': [], 'emg': [], 'ecg': [],
                'thoraco_abdo_resp': [], 'nasal_pressure': [], 'snoring': [],
            },
            "file_extensions": {'psg_ext': 'I0003/**/*-PSG_eeg.edf',
                                'ann_ext': 'I0003/**/*-psg_sleep_annotations.csv'},
        }

    def _config_i0004(self):
        # TODO: fill in real ann2label/mappings/channel names/types/groups for I0004
        return {
            "ann2label": {},
            "inter_dataset_mapping": {},
            "intra_dataset_mapping": {},
            "channel_names": [],
            "channel_types": {'analog': [], 'digital': []},
            "channel_groups": {
                'eeg_eog': [], 'emg': [], 'ecg': [],
                'thoraco_abdo_resp': [], 'nasal_pressure': [], 'snoring': [],
            },
            "file_extensions": {'psg_ext': 'I0004/**/*-PSG_eeg.edf',
                                'ann_ext': 'I0004/**/*-psg_sleep_annotations.csv'},
        }
    
    def _config_i0006(self):
        # TODO: fill in real ann2label/mappings/channel names/types/groups for I0006
        return {
            "ann2label": {},
            "inter_dataset_mapping": {},
            "intra_dataset_mapping": {},
            "channel_names": [],
            "channel_types": {'analog': [], 'digital': []},
            "channel_groups": {
                'eeg_eog': [], 'emg': [], 'ecg': [],
                'thoraco_abdo_resp': [], 'nasal_pressure': [], 'snoring': [],
            },
            "file_extensions": {'psg_ext': 'I0006/**/*-PSG_eeg.edf',
                                'ann_ext': 'I0006/**/*-psg_sleep_annotations.csv'},
        }

    def _config_s0001(self):
        # TODO: fill in real ann2label/mappings/channel names/types/groups for S0001
        return {
            "ann2label": {},
            "inter_dataset_mapping": {},
            "intra_dataset_mapping": {},
            "channel_names": [],
            "channel_types": {'analog': [], 'digital': []},
            "channel_groups": {
                'eeg_eog': [], 'emg': [], 'ecg': [],
                'thoraco_abdo_resp': [], 'nasal_pressure': [], 'snoring': [],
            },
            "file_extensions": {'psg_ext': 'S0001/**/*-PSG_eeg.edf',
                                'ann_ext': 'S0001/**/*-psg_sleep_annotations.csv'},
        }

    def ann_parse(self, ann_fname):

        ann_df = pd.read_csv(ann_fname, sep=',', header=0)

        ann_stage_events = []
        start_time = None
        epoch_duration = 30

        if "sub-I0002150029658_ses-2_task-psg_sleep_annotations" in ann_fname:
            ann_df.at[ann_df.index[101], 'Epoch'] = '102'
            ann_df['Epoch'] = ann_df['Epoch'].astype('int64')

        for i, row in ann_df.iterrows():

            start = (row["Epoch"] - 1) * epoch_duration     # 1-based indexing
            stage = row['Stage']
            if pd.isna(stage):
                stage = "UNSCORED"

            if start_time is None:
                start_time = start

            ann_stage_events.append({'Stage': stage,
                                        'Start': start - start_time,
                                        'Duration': epoch_duration})

        sleep_epochs = ann_df[ann_df['Stage'] != 'L']
        lights_off = int((sleep_epochs.iloc[0]['Epoch'] - 1) * epoch_duration)
        lights_on = int(sleep_epochs.iloc[-1]['Epoch'] * epoch_duration)

        if lights_off is None:
            raise Exception(f"Could not determine lights off time for {ann_fname}")
        if lights_on is None:
            raise Exception(f"Could not determine lights on time for {ann_fname}")

        # events_file = ann_fname.replace('_sleep_annotations.csv', '_events_annotations.csv')

        return ann_stage_events, start_time, lights_off, lights_on
