import os
from typing import Tuple
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("HOMEPAP")
class HOMEPAP(BaseDataset):
    """HOMEPAP (Home Positive Airway Pressure) dataset."""
    
    def __init__(self):
        super().__init__("HOMEPAP","HOMEPAP - Home Positive Airway Pressure",keep_folder_structure=True)

    def _setup_dataset_config(self):
        self.ann2label = {
            "Wake": "W",
            "Stage 1 sleep": "N1",
            "Stage 2 sleep": "N2",
            "Stage 3 sleep": "N3",
            "Stage 4 sleep": "REM",
            "REM sleep": "REM",
            "Movement": 5
        }
        
        
        self.intra_dataset_mapping = {
                'abdomen': [ 'ABDOMEN', 'Abdomen','ABD'],
                'cannula_flow': ['Cannulaflow', 'CannlulaFlow','CannualFlow','Cannula Flow', 'CannulaFlow','CannulaFLow',],
                'cannula_nasal': ['Nasal Cannula','CannulaNasal'],
                'chin': ['CHIN', 'Chin EMG', 'Chin', 'EMG Chin',],
                'chest': ['CHEST','Chest','Thorax'],
                'lchin': ['LCHIN','Lchin','LChin'],
                'rchin': ['Rchin','RChin','RCHIN',],
                'cchin':['Cchin','CChin', 'CCHIN'],
                'E1': ['E1', 'L-EOG','LOC'],
                'E2': ['E-2','E2', 'R-EOG','ROC'],
                'ECG': ['EKG', 'ECG'],
                'ECG1':['EKG1','ECG I','ECG1',],
                'ECG2':['ECG2','EKG2','ECG II'],
                'ECG3':['EKG3','ecg3','ECG3'],
                'EtCo2': ['EtCO2 #', 'ETCO2#', 'EtCO2'],
                'EtCo2_Wave': ['ETCO2Wave','EtCO2 Wave'],
                'TcCO2': ['TcCO2 #','TcCO2','TCCO2#','TCCO2',],
                'airflow': ['AIRFLOW','Airflow','AirFlow','AIR-flow','Flow'],
                'heartrate': ['HeartRate','Heart Rate','HRate'],
                'leak': ['Leak', 'LEAK1','MaskLeak', 'LEAK'],
                'lleg': ['L-Legs', 'Lleg', 'LLeg', 'LLEG','L Leg','Lleg1','LLeg1','L-LEG1','LLEG1'],     
                'lleg2': ['L-LEG2','LLEG2','LLeg2','Lleg2'],
                'pap_flow': ['CFLOW', 'CPAPFLOW', 'PAP Flow'],
                'pap_mask': [ 'CPAPMask','Mask'],
                'mask_flow': ['MaskFlow', 'MaskFLow','Mask Flow'],
                'pap_pres': ['xPAP CPAP', 'PAP', 'CPAPPressure', 'PAP Press', 'xPAP','CPAP'],
                'phase': [ 'Phase'],
                'pleth': ['PLeth','PLETH', 'Pleth'],
                'position': ['Positiom', 'Position'],
                'body_position': ['Body Position','BodyPosition'],
                'pressure': ['Pressure', 'pressure','PRESSURE1'],
                'pulse': [ 'PULSE', 'Pulse'],
                'r_r': ['R-R', 'Resp Rate', 'RespRate', 'Resp','RR'],
                'rleg': [ 'RLEG', 'RLeg', 'R Leg', 'R-Legs','Rleg','RLEG1','RLeg1','Rleg1','R-LEG1'],
                'rleg2': ['RLeg2','Rleg2', 'RLEG2','R-LEG2' ],                
                'snore': ['Snore'],
                'snore2': ['SNORE'],
                'snore_mic': ['Snore Mic','SNORE mic','SNORE Mic','SNORE MIC 1'],
                'snore_sensor': ['Snore Sensor','SnoreSensor'],
                'cannula_snore': ['CannulaSnore','Cannula Snore'],
                'spo2': ['SpO2', 'SPO2'],
                'saO2': ['SaO2', 'Sa02','SA02', 'SAO2'],
                'spostat': ['OXSTAT','Ox Status','Ox',],
                'therm': ['Thermistor'],
                'tvol': ['TidVol','Tidal Volume'],
                'xsum': ['XSum'],
                }
        
        self.inter_dataset_mapping = {
            "E1": self.Mapping(self.TTRef.EL, self.TTRef.Fpz),
            "E2": self.Mapping(self.TTRef.ER, self.TTRef.Fpz),
            "F3": self.Mapping(self.TTRef.F3, self.TTRef.Fpz),
            "F4": self.Mapping(self.TTRef.F4, self.TTRef.Fpz),
            "C3": self.Mapping(self.TTRef.C3, self.TTRef.Fpz),
            "C4": self.Mapping(self.TTRef.C4, self.TTRef.Fpz),
            "O1": self.Mapping(self.TTRef.O1, self.TTRef.Fpz),
            "O2": self.Mapping(self.TTRef.O2, self.TTRef.Fpz),
            "M1": self.Mapping(self.TTRef.LPA, self.TTRef.Fpz),
            "M2": self.Mapping(self.TTRef.RPA, self.TTRef.Fpz),
            "E1-M2": self.Mapping(self.TTRef.EL, self.TTRef.RPA),
            "E2-M1": self.Mapping(self.TTRef.ER, self.TTRef.LPA),
            "F3-M2": self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            "F4-M1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            "C3-M2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "C4-M1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "O1-M2": self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            "O2-M1": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "ECG": self.Mapping(self.TTRef.ECG, self.TTRef.Fpz),
            "lleg": self.Mapping(self.TTRef.EMG_LLEG, self.TTRef.Fpz),
            "rleg": self.Mapping(self.TTRef.EMG_RLEG, self.TTRef.Fpz),
            "chin": self.Mapping(self.TTRef.EMG_CHIN, self.TTRef.Fpz),
            "lchin": self.Mapping(self.TTRef.EMG_LCHIN, self.TTRef.Fpz),
            "rchin": self.Mapping(self.TTRef.EMG_RCHIN, self.TTRef.Fpz),
            "position": self.Mapping(self.TTRef.POSITION, None),
            "spo2": self.Mapping(self.TTRef.SPO2, None),
            "pap_flow": self.Mapping(self.TTRef.CPAP, None),
            "snore": self.Mapping(self.TTRef.SNORE, None),
            "abdomen": self.Mapping(self.TTRef.ABDOMINAL, None),
            "chest": self.Mapping(self.TTRef.THORACIC, None),
            "airflow": self.Mapping(self.TTRef.AIRFLOW, None),
        }
        
        
        self.channel_names = [ # lab
                                'A1', 'A2', 'ABD', 'ABDOMEN', 'AIR-flow', 'AIRFLOW', 'Abd', 'Abd2', 'Abdomen', 'AirFlow', 'Airflow', 'Arm',
                               'Body', 'Body Position', 'BodyPosition', 'C', 'C.', 'C3', 'C3-M2', 'C4', 'C4-M1', 'CCHIN', 'CChin', 'CFLOW',
                                'CHEST', 'CHIN', 'CPAP', 'CPAPFLOW', 'CPAPMask', 'CPAPPressure', 'CannlulaFlow', 'CannualFlow', 'Cannula',
                                'Cannula Flow', 'Cannula Snore', 'CannulaFLow', 'CannulaFlow', 'CannulaNasal', 'CannulaSnore', 'Cannulaflow',
                                'Cchin', 'Chest', 'Chest1', 'Chin', 'Chin EMG', 'Chin1', 'Chin1-Chin2', 'Chin2', 'Chin2-Chin3', 'Chin3', 
                                'DC02', 'DC09', 'DHR', 'E-1', 'E-2', 'E1', 'E1-E2', 'E1-M2', 'E2', 'E2-M1', 'ECG', 'ECG I', 'ECG II', 'ECG1',
                                'ECG1-ECG2', 'ECG1-ecg3', 'ECG2', 'ECG3', 'ECG3-ECG1', 'EKG', 'EKG1', 'EKG2', 'EKG3', 'EMG', 'EMG Chin',
                                'EMG1', 'EMG2', 'EMG3', 'ETCO2#', 'ETCO2Wave', 'EtCO2', 'EtCO2 #', 'EtCO2 Wave', 'F2-M2', 'F3', 'F3-M2',
                                'F4', 'F4-M1', 'FLOW', 'FLOW2', 'FLOW5', 'FPz', 'Gravity', 'HR', 'HRate', 'Heart', 'Heart Rate',
                                'HeartRate', 'Imp', 'L', 'L Leg', 'L-EOG', 'L-LEG1', 'L-LEG2', 'L-Legs', 'L.', 'LARM6', 'LCHIN', 'LChin', 
                                'LEAK', 'LEAK1', 'LEFT', 'LEFT IC1', 'LEFT IC2', 'LEG/L2', 'LEG/R1', 'LEG/R2', 'LLEG', 'LLEG1', 'LLEG2', 'LLeg',
                                'LLeg1', 'LLeg1-LLeg2', 'LLeg2', 'LOC', 'Lchin', 'Lchin-Cchin', 'Leak', 'Leg Ltibial', 'Leg Rtibial', 'Lleg',
                                'Lleg1', 'Lleg2', 'M1', 'M1-M2', 'M2', 'Mask', 'Mask Flow', 'Mask Leak', 'MaskFLow', 'MaskFlow', 'MaskLeak', 
                                'NP', 'NPV', 'NPV flow', 'Nasal', 'Nasal Cannula', 'Nasal Pressure', 'O1', 'O1-M2', 'O2', 'O2-M1', 'OXSTAT', 
                                'Ox', 'Ox Status', 'PAP', 'PAP Flow', 'PAP Press', 'PLETH', 'PLeth', 'PRESSURE1', 'PULSE', 'Phase', 'Pleth', 
                                'Positiom', 'Position', 'Pressure', 'PressureE', 'PressureI', 'ProTech', 'Pulse', 'R', 'R Leg', 'R-EOG', 
                                'R-LEG1', 'R-LEG2', 'R-Legs', 'R-R', 'R.', 'RARM7', 'RCHIN', 'RChin', 'REF', 'RES', 'RIGHT', 'RIGHT IC1', 
                                'RIGHT IC2', 'RLEG', 'RLEG1', 'RLEG2', 'RLeg', 'RLeg1', 'RLeg1-RLeg2', 'RLeg2', 'RMI', 'ROC', 'Rchin', 'Rchon', 
                                'Reg2', 'Resp', 'Resp Rate', 'RespRate', 'Rleg', 'Rleg1', 'Rleg2', 'Room Light', 'SA02', 'SAO2', 'SNORE', 
                                'SNORE MIC 1', 'SNORE Mic', 'SPO2', 'SUM', 'Sa02', 'SaO2', 'Snore', 'Snore Mic', 'Snore Sensor', 'SnoreSensor', 
                                'SpO2', 'TCCO2', 'TCCO2#', 'THOR', 'TcCO2', 'TcCO2 #', 'TcCO2 Pleth', 'Thermistor', 'TidVol', 'VTinsp', 'Volume', 
                                'XFlow', 'XSum', 'ecg3', 'ecg3-ECG2', 'pressure', 'xPAP', 'xPAP CPAP',
                                # home
                                'Battery', 'XFlow_PDS', 'Elevation', 'Activity', 'Flattening', 'RD-Pleth', 'RR', 'RD-Quality', 'Tidal Volume',
                                'Gravity X', 'SpO2-Quality', 'SpO2-BB', 'Thorax', 'Gravity Y', 'Flow']
        
        
        self.channel_types = {'analog': ['RLeg', 'Cannulaflow', 'AIR-flow', 'NP', 'C4', 'C3-M2', 'RLeg2', 'O2', 'ECG1-ECG2', 'PULSE', 'RCHIN', 
                                         'SUM', 'ECG1-ecg3', 'ECG2', 'MaskFlow', 'Abd', 'CannlulaFlow', 'Chin3', 'ECG3-ECG1', 'PLETH', 
                                         'Chin2-Chin3', 'F3-M2', 'CannulaSnore', 'Lleg2', 'SAO2', 'Imp', 'Abdomen', 'LEAK', 'RIGHT IC2', 
                                         'LEG/R2', 'ABDOMEN', 'RARM7', 'Lleg1', 'LEFT', 'EKG', 'Cannula Snore', 'PLeth', 'LLEG', 'L-LEG2', 
                                         'RespRate', 'THOR', 'L-EOG', 'Chin', 'Nasal', 'Heart', 'CChin', 'EKG1', 'Nasal Pressure', 'LLeg1', 
                                         'DHR', 'F2-M2', 'C3', 'ProTech', 'SNORE Mic', 'SnoreSensor', 'TcCO2 Pleth', 'LLeg2', 'Rchin', 'Snore', 
                                         'F4-M1', 'Airflow', 'DC02', 'EMG3', 'R-LEG2', 'Heart Rate', 'R.', 'ABD', 'Cchin', 'A2', 'Rleg', 'E-1', 
                                         'CFLOW', 'Abd2', 'C4-M1', 'ecg3-ECG2', 'ECG II', 'Arm', 'Lchin', 'Nasal Cannula', 'LOC', 'EKG3', 
                                         'E1-M2', 'LChin', 'TcCO2', 'A1', 'Pleth', 'EKG2', 'EMG Chin', 'R-Legs', 'LLEG2', 'Lleg', 'R-LEG1', 
                                         'LEG/L2', 'Snore Sensor', 'ecg3', 'Chin2', 'Mask Leak', 'Position', 'E-2', 'Snore Mic', 'CPAPMask', 
                                         'LLeg1-LLeg2', 'RChin', 'L.', 'R-EOG', 'LEFT IC2', 'LEFT IC1', 'R Leg', 'L-LEG1', 'Rleg1', 'M1-M2', 
                                         'E2', 'XSum', 'RIGHT IC1', 'Rchon', 'R-R', 'Thermistor', 'RLeg1', 'Chest1', 'LEG/R1', 'Chin EMG', 
                                         'PAP Flow', 'CannulaFLow', 'LLEG1', 'MaskLeak', 'CannualFlow', 'Mask', 'LARM6', 'FLOW5', 'EMG', 
                                         'Pressure', 'SNORE', 'AIRFLOW', 'RLEG2', 'O2-M1', 'EMG1', 'SNORE MIC 1', 'REF', 'M1', 'CPAPFLOW', 
                                         'ECG', 'Lchin-Cchin', 'LLeg', 'R', 'AirFlow', 'Pulse', 'PAP Press', 'F3', 'MaskFLow', 'HR', 'CCHIN', 
                                         'F4', 'FLOW2', 'RLEG1', 'LCHIN', 'Cannula', 'RLEG', 'TCCO2', 'Leg Rtibial', 'CPAPPressure', 'E2-M1', 
                                         'CHIN', 'FPz', 'Leg Ltibial', 'ROC', 'Reg2', 'ECG I', 'C.', 'Mask Flow', 'HeartRate', 'Phase', 'L', 
                                         'FLOW', 'Cannula Flow', 'Chin1', 'CannulaFlow', 'E1', 'CHEST', 'Leak', 'ECG3', 'O1-M2', 'CannulaNasal', 
                                         'Rleg2', 'M2', 'L-Legs', 'O1', 'HRate', 'XFlow', 'E1-E2', 'L Leg', 'EMG2', 'RIGHT', 'Chest', 
                                         'RLeg1-RLeg2', 'ECG1', 'Chin1-Chin2','Battery', 'XFlow_PDS', 'Elevation', 'Activity', 'Flattening',
                                         'RD-Pleth', 'RR', 'Tidal Volume', 'Gravity X', 'Thorax', 'Gravity Y', 'Flow'], 
                            'digital': ['PressureE', 'C', 'xPAP CPAP', 'PRESSURE1', 'Ox Status', 'Room Light', 'NPV flow', 
                                        'Body Position', 'VTinsp', 'Resp', 'SA02', 'EtCO2 Wave', 'Positiom', 'Gravity', 'ETCO2#', 
                                        'DC09', 'TCCO2#', 'EtCO2 #', 'pressure', 'Volume', 'RES', 'xPAP', 'PAP', 'BodyPosition', 
                                        'SaO2', 'SpO2', 'Body', 'PressureI', 'OXSTAT', 'CPAP', 'LEAK1', 'Ox', 'Sa02', 'NPV', 'RMI', 
                                        'ETCO2Wave', 'SPO2', 'TcCO2 #', 'Resp Rate', 'EtCO2', 'TidVol', 'RD-Quality', 'SpO2-Quality', 'SpO2-BB']}

        
        self.channel_groups = {
            'eeg_eog': ['C3-M2', 'C3', 'C4', 'C4-M1', 'F2-M2', 'F3-M2', 'F3', 'F4-M1', 'F4', 'M1-M2', 'O1-M2', 'O2-M1','O2',
                        'E1-M2', 'E1-E2', 'E1', 'L-EOG', 'LOC', 'E2-M1', 'E-2', 'E2', 'R-EOG', 'ROC'],
            'emg': ['Chin2-Chin3', 'LCHIN', 'CHIN', 'Chin EMG', 'Chin1-Chin2', 'Lchin', 'Chin', 'RChin', 'RCHIN', 
                    'Cchin', 'Chin2', 'CChin', 'Rchin', 'LChin', 'Chin1', 'Chin3', 'Lchin-Cchin', 'EMG Chin', 'CCHIN',
                    'EMG1', 'EMG', 'EMG3', 'EMG2', 'L-Legs', 'Lleg', 'L-LEG2', 'Lleg1', 'LLeg', 'LLeg1-LLeg2', 
                    'LEG/L2', 'LLEG1', 'LLEG', 'LLeg1', 'L Leg', 'L-LEG1', 'LLEG2', 'LLeg2', 'Lleg2', 'Leg Ltibial',
                    'LEG/R1', 'RLEG1', 'RLEG', 'RLeg2', 'Rleg2', 'RLeg', 'RLeg1', 'RLEG2', 'R Leg', 'Rleg1', 
                    'RLeg1-RLeg2', 'R-LEG2', 'R-LEG1', 'R-Legs', 'Rleg', 'LEG/R2', 'Leg Rtibial'],
            'ecg': ['EKG3', 'ecg3-ECG2', 'EKG1', 'ECG3-ECG1', 'ecg3', 'ECG2', 'EKG', 'EKG2', 'ECG', 'ECG I', 
                    'ECG1-ECG2', 'ECG3', 'ECG1', 'ECG1-ecg3', 'ECG II'],
            'thoraco_abdo_resp': ['Thermistor', 'ABDOMEN', 'Abdomen', 'ABD', 'Abd', 'Abd2', 'THOR', 'CHEST', 'Chest', 'Chest1','Thermistor','AIRFLOW','Airflow','AirFlow','AIR-flow','Flow'],
            'nasal_pressure': ['Nasal', 'Nasal Pressure'],
            'snoring': ['SNORE', 'Snore', 'Snore Mic', 'Snore Sensor', 'CannulaSnore', 'SnoreSensor', 'SNORE mic', 
                        'Cannula Snore', 'SNORE MIC 1']
        }
        
        
        self.file_extensions = {
            'psg_ext': '**/*.edf',
            'ann_ext': '**/*-nsrr.xml'
        }