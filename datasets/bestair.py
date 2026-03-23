from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("BESTAIR")
class BESTAIR(BaseDataset):
    """BESTAIR (Best Apnea Interventions in Research) dataset"""

    def __init__(self):
        super().__init__("BESTAIR","BESTAIR - Best Apnea Interventions in Research")

    # Note: BESTAIR has limited recordings with EEG. Most of the recordings are done using a home sleep test (HST).
    # Sleep staging is using HST was done using 0: wake, 2: sleep.
    def _setup_dataset_config(self):
        self.ann2label = {
            "Wake": 0,
            "Stage 1 sleep": 1,
            "Stage 2 sleep": 2,
            "Stage 3 sleep": 3,
            "Stage 4 sleep": 3,  # Follow AASM Manual
            "REM sleep": 4,
            "Unscored": 6
        }
        
        # https://gitlab-scm.partners.org/zzz-public/nsrr/-/blob/master/studies/legacy/bestair/sigs.alias.1
        self.intra_dataset_mapping = {
            'abdomen': ['abdomen', 'Abdomen', 'ABDOMEN', 'Effort Abd', 'X21'],
            'activity': ['Activity'],
            'battery': ['Battery'],
            'C3_M2': ['C3-M2', 'EEG C3-M2', 'M2-C3'],
            'C4_M1': ['C4-M1', 'EEG C4-M1', 'M1-C4'],
            'cchin': ['EMG Chin 1', 'X6'],
            'cchin_l': ['X6-X5'],
            'chin': ['CHIN', 'EMG Chin'],
            'lchin': ['CHIN2', 'CHIN 2'],
            'rchin': ['EMG Chin 2', 'X7'],
            'rchin_c': ['X7-X6'],
            'dif_pres': ['Differential Pre'],
            'E1': ['EOG Left', 'LOC', 'M1-X1'],
            'E1_M2': ['E1-M2', 'M1-PG2'],
            'E2': ['EOG Right', 'M2-X2', 'ROC'],
            'E2_M2': ['E2-M2','M2-PG1'],
            'ECG': ['EKG', 'X15'],
            'elevation': ['Elevation'],
            'ex_pres': ['xPAP EPAP'],
            'F3_M2': ['EEG F3-M2', 'F3-M2', 'M2-F3', 'M2-X3'],
            'F4_M1': ['EEG F4-M1', 'F4-M1', 'M1-F4', 'M1-X4'],
            'flow': ['FLOW', 'Flow', 'Flow Patient2'],
            'gravx': ['Gravity X'],
            'gravy': ['Gravity Y'],
            'in_pres': ['xPAP IPAP'],
            'leak': ['Leak'],
            'lleg': ['Leg L-LEG', 'L-Leg', 'L-LEG', 'X17'],
            'M1_M2': ['EEG M1-M2'],
            'nas_pres': ['DC03', 'Nasal', 'NAF', 'Flow Patient 3', 'Flow Patient'],
            'O1_M2': ['EEG O1-M2', 'O1-M2', 'M2-O1'],
            'O2_M1': ['EEG O2-M1', 'O2-M1', 'M1-O2'],
            'oxstat': ['SpO2-Quality'],
            'pap_flow': ['CFlow', 'CFLOW', 'DC07'],
            'pap_pres': ['DC05', 'CPAP'],
            'phase': ['Phase'],
            'pleth': ['RD-Pleth'],
            'plethstat': ['RD-Quality'],
            'position': ['Position', 'Body', 'BODY', 'DC02'],
            'pulse': ['Pulse'],
            'r_r': ['RR', 'R-R'],
            'rleg': ['R-Leg', 'R-LEG', 'X16'],
            'snore': ['Snore', 'SNORE', 'X18'],
            'spo2': ['DC01', 'SaO2', 'SAO2', 'SpO2'],
            'spo2bb': ['SpO2-BB'],
            'spo2w': ['SAO2W'],
            'sum': ['Sum'],
            'therm': ['Thermistor', 'X19', 'Flow Patient 1'],
            'thorax': ['Effort Tho', 'THO', 'Thorax', 'THORAX', 'X20'],
            'tvol': ['Tidal Volume'],
            'xflow': ['XFlow'],
            'xsum': ['XSum'],
        }

        self.inter_dataset_mapping = {
            'E1': self.Mapping(self.TTRef.EL, None),
            'E1_M2': self.Mapping(self.TTRef.EL, self.TTRef.RPA),
            'E2': self.Mapping(self.TTRef.ER, None),
            'E2_M2': self.Mapping(self.TTRef.ER,  self.TTRef.RPA),
            'ECG': self.Mapping(self.TTRef.ECG, None),
            'C3_M2': self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            'C4_M1': self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            'F3_M2': self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            'F4_M1': self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            'O1_M2': self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            'O2_M1': self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            'M1_M2': self.Mapping(self.TTRef.LRPA, None),
            'snore': self.Mapping(self.TTRef.SNORE, None),
            'spo2': self.Mapping(self.TTRef.SPO2, None),
            'lleg': self.Mapping(self.TTRef.EMG_LLEG, None),
            'rleg': self.Mapping(self.TTRef.EMG_RLEG, None),
            'abdomen': self.Mapping(self.TTRef.ABDOMINAL, None),
            'thorax': self.Mapping(self.TTRef.THORACIC, None),
            'position': self.Mapping(self.TTRef.POSITION, None),
            'chin': self.Mapping(self.TTRef.EMG_CHIN, None),
            'lchin': self.Mapping(self.TTRef.EMG_LCHIN, None),
            'rchin': self.Mapping(self.TTRef.EMG_RCHIN, None),
            'flow': self.Mapping(self.TTRef.AIRFLOW, None),
            'papflow': self.Mapping(self.TTRef.CPAP , None),
        }
        
        
        self.channel_names = ['LOC', 'E1-M2', 'M2-X3', 'M2-F3', 'E2-M2', 'M1-X1', 
                'M1-O2', 'O1-M2', 'X1', 'M2-C3', 'EEG O2-M1', 'EEG C4-M1', 'M1-X4', 
                'C3-M2', 'X2', 'EEG F4-M1', 'C4-M1', 'M1-C4', 'EEG M1-M2',
                'M2-PG1', 'M1-F4', 'F4-M1', 'EEG C3-M2', 'EOG Right', 
                'F3-M2', 'M2-X2', 'EEG O1-M2', 'ROC', 'T2-T1', 'EOG Left', 'M1-PG2', 'O2-M1',  
                'EEG F3-M2', 'M2-O1', 'EMG Chin', 'Leg L-LEG', 'xPAP IPAP', 'Effort Tho', 'ABDOMEN', 'CFLOW', 'DHR', 'DC03', 'Tidal Volume', 'R-R',
                'THORAX', 'Leak', 'Phase', 'Body', 'Snore', 'Chin', 'SpO2', 'Sum', 'Effort Abd', 'DC07', 'Thermistor',
                'EKG', 'RD-Pleth', 'CHIN', 'Flow Patient 2', 'RMI', 'FLOW', 'L-LEG', 'SNORE', 'SpO2-BB', 'BODY', 'Activity',
                'XSum', 'SpO2-Quality', 'SAO2', 'CHIN2', 'DC04', 'THO', 'Pulse', 'L-Leg', 'SAO2W', 'Battery', 'XFlow', 
                'Gravity Y', 'DC05', 'Thorax', 'CPAP', 'Position', 'DC02', 'NAF', 'Gravity X', 'Elevation', 
                'CHIN 2', 'RD-Quality', 'CFlow', 'Flow Patient 3', 'RR', 'xPAP EPAP', 'EMG Chin 2', 'R-LEG', 'R-Leg',
                'Nasal', 'Flow Patient 1', 'DC01', 'Flow', 'Abdomen', 'SaO2', 'ECG', 'EMG Chin 1', 'Differential Pre', 'X6-X5',
                'X15', 'X18', 'X16', 'X3', 'X7', 'X5', 'X19', 'X4', 'X21', 'X17', 'X6', 'X7-X6', 'X20']
        
        self.channel_types = {
            'analog': ['RMI', 'RR', 'L-Leg', 'Flow Patient 2', 'Flow', 'E2-M2', 'Position', 'M1-X4', 'X15', 'Gravity Y', 'X1', 'SNORE',
                       'DC07', 'Battery', 'CFLOW', 'EEG O1-M2', 'Pulse', 'X7', 'DC04', 'X21', 'M1-F4', 'THO', 'F4-M1', 'M2-X3', 'THORAX',
                       'Nasal', 'EEG F3-M2', 'XFlow', 'M2-O1', 'ROC', 'EEG M1-M2', 'EEG C4-M1', 'X20', 'M2-PG1', 'M1-PG2', 'M1-O2', 
                       'Thorax', 'DHR', 'Phase', 'Snore', 'M2-F3', 'X4', 'EOG Right', 'FLOW', 'X5', 'CFlow', 'EEG F4-M1', 'Elevation',
                       'E1-M2', 'X7-X6', 'X6', 'NAF', 'X18', 'X19', 'ECG', 'EMG Chin', 'EEG C3-M2', 'EOG Left', 'R-R', 'Gravity X', 'Sum',
                       'CHIN2', 'X17', 'F3-M2', 'Thermistor', 'L-LEG', 'Flow Patient 3', 'O2-M1', 'CPAP', 'CHIN', 'X6-X5', 'EKG', 'EEG O2-M1',
                       'X16', 'R-Leg', 'Leg L-LEG', 'X3', 'C4-M1', 'Chin', 'O1-M2', 'Tidal Volume', 'Effort Abd', 'C3-M2', 'EMG Chin 2', 'DC03',
                       'CHIN 2', 'LOC', 'M2-C3', 'Activity', 'XSum', 'M1-C4', 'M2-X2', 'X2', 'ABDOMEN', 'M1-X1', 'Abdomen', 'T2-T1', 'EMG Chin 1', 
                       'Effort Tho', 'R-LEG', 'Differential Pre'], 
            'digital': ['Body', 'SAO2', 'SAO2W', 'DC01', 'BODY', 'DC05', 'SpO2', 'RD-Pleth', 'xPAP IPAP', 'Flow Patient 1', 'xPAP EPAP', 'Leak', 
                        'SpO2-BB', 'SpO2-Quality', 'DC02', 'RD-Quality', 'SaO2']
        }
    
        
        self.channel_groups =  {
            'eeg_eog': ['LOC', 'E1-M2', 'M2-X3', 'M2-F3', 'E2-M2', 'M1-X1', 'M1-O2', 'O1-M2', 'X1', 'M2-C3', 'EEG O2-M1', 'EEG C4-M1', 'M1-X4', 'C3-M2', 'X2', 'EEG F4-M1', 'C4-M1', 'M1-C4', 'EEG M1-M2', 'M2-PG1', 'M1-F4', 'F4-M1', 'EEG C3-M2', 'EOG Right', 'F3-M2', 'M2-X2', 'EEG O1-M2', 'ROC', 'T2-T1', 'EOG Left', 'M1-PG2', 'O2-M1', 'EEG F3-M2', 'M2-O1'],
            'emg': ['chin', 'CHIN', 'Chin', 'EMG_Chin', 'lchin', 'CHIN2', 'CHIN_2', 'X5', 'rchin', 'EMG_Chin_2', 'X7', 'rchin_c', 'X7-X6', 'cchin', 'EMG_Chin_1', 'X6', 'cchin_l', 'X6-X5', 'lleg', 'Leg_L-LEG', 'L-Leg', 'L-LEG', 'X17', 'rleg', 'R-Leg', 'R-LEG', 'X16'],
            'ecg': ['ECG', 'EKG', 'X15'],
            'thoraco_abdo_resp': ['abdomen', 'Abdomen', 'ABDOMEN', 'Effort_Abd', 'X21', 'thorax', 'Effort_Tho', 'THO', 'Thorax', 'THORAX', 'X20','Thermistor', 'X19', 'Flow Patient 1','FLOW', 'Flow', 'Flow Patient2'],
            'nasal_pressure': ['nas_pres'],
            'snoring': ['snore', 'Snore', 'SNORE', 'X18']
        }
                
        self.file_extensions = {
            'psg_ext': '**/*.edf',
            'ann_ext': '**/*-nsrr.xml'
        }
