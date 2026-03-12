"""
STAGES - Stanford Technology Analytics and Genomics in Sleep
"""
import os
import pandas as pd
import pyedflib
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("STAGES")
class STAGES(BaseDataset):
    """STAGES - Stanford Technology Analytics and Genomics in Sleep dataset."""
    
    def __init__(self):
        super().__init__("STAGES","STAGES - Stanford Technology Analytics and Genomics in Sleep", keep_folder_structure=False)
    
    def _setup_dataset_config(self):
        self.ann2label = {
                'Wake': 0,
                'Stage1': 1,
                'Stage2': 2,
                'Stage3': 3,
                'REM': 4,
                'UnknownStage': 6,
        }

        # does not include channels that are only present in very few recordings
        self.inter_dataset_mapping = {
            "Abdomen": self.Mapping(self.TTRef.ABDOMINAL, None),
            "Airflow": self.Mapping(self.TTRef.AIRFLOW, None),
            "Cflow": self.Mapping(self.TTRef.CPAP, None),
            "C3-M2": self.Mapping(self.TTRef.C3, self.TTRef.RPA),
            "C4-M1": self.Mapping(self.TTRef.C4, self.TTRef.LPA),
            "Chest": self.Mapping(self.TTRef.THORACIC, None),
            "Snore": self.Mapping(self.TTRef.SNORE, None),
            "Chin": self.Mapping(self.TTRef.EMG_CHIN, None),
            "ECG": self.Mapping(self.TTRef.ECG, None),
            "F3-M2": self.Mapping(self.TTRef.F3, self.TTRef.RPA),
            "F4-M1": self.Mapping(self.TTRef.F4, self.TTRef.LPA),
            "FP1": self.Mapping(self.TTRef.Fp1, None),
            "FP2": self.Mapping(self.TTRef.Fp2, None),
            "LOC": self.Mapping(self.TTRef.EL, None),
            "LOC_A2": self.Mapping(self.TTRef.EL, self.TTRef.RPA),
            "ROC": self.Mapping(self.TTRef.ER, None),
            "ROC_A2": self.Mapping(self.TTRef.ER, self.TTRef.RPA),
            "A1": self.Mapping(self.TTRef.LPA, None),
            "A2": self.Mapping(self.TTRef.RPA, None),
            "BodyPosition": self.Mapping(self.TTRef.POSITION, None),
            "C3": self.Mapping(self.TTRef.C3, None),
            "C4": self.Mapping(self.TTRef.C4, None),
            "CPAP": self.Mapping(self.TTRef.CPAP, None),
            "Cz": self.Mapping(self.TTRef.Cz, None),
            "EEG_A1-A2": self.Mapping(self.TTRef.LPA, self.TTRef.RPA),
            "EEG_Fp1-A2": self.Mapping(self.TTRef.Fp1, self.TTRef.RPA),
            "EEG_Fp2-A1": self.Mapping(self.TTRef.Fp2, self.TTRef.LPA),
            "O1-M2": self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            "O2-M1": self.Mapping(self.TTRef.O2, self.TTRef.LPA),
            "EEG_P3-A2": self.Mapping(self.TTRef.P3, self.TTRef.RPA),
            "EEG_P4-A1": self.Mapping(self.TTRef.P4, self.TTRef.LPA),
            "EEG_T5-A2": self.Mapping(self.TTRef.P7, self.TTRef.RPA),
            "EEG_T6-A1": self.Mapping(self.TTRef.P8, self.TTRef.LPA),
            "F1M2": self.Mapping(self.TTRef.F1, self.TTRef.RPA),
            "F2M1": self.Mapping(self.TTRef.F2, self.TTRef.LPA),
            "F3": self.Mapping(self.TTRef.F3, None),
            "F4": self.Mapping(self.TTRef.F4, None),
            "F7": self.Mapping(self.TTRef.F7, None),
            "EEG_F7-A2": self.Mapping(self.TTRef.F7, self.TTRef.RPA),
            "EEG_F8-A1": self.Mapping(self.TTRef.F8, self.TTRef.LPA),
            "F8": self.Mapping(self.TTRef.F8, None),
            "Fpz": self.Mapping(self.TTRef.Fpz, None),
            "Fz": self.Mapping(self.TTRef.Fz, None),
            "T3-M2": self.Mapping(self.TTRef.T7, self.TTRef.RPA),
            "T3": self.Mapping(self.TTRef.T7, None),
            "T4": self.Mapping(self.TTRef.T8, None),
            "T4-M1": self.Mapping(self.TTRef.T8, self.TTRef.LPA),
            "T5": self.Mapping(self.TTRef.P7, None),
            "T6": self.Mapping(self.TTRef.P8, None),
            "RAT": self.Mapping(self.TTRef.EMG_RLEG, None),
            "LAT": self.Mapping(self.TTRef.EMG_LLEG, None),
        }

        self.intra_dataset_mapping = {'Abdomen': [ 'ABD', 'ABDM', 'ABDOMEN', 'Abd', 'Abdomen'],
                                    'A1': ['A1', 'M1'],
                                    'A2': ['A2', 'M2'],
                                    'Battery': ['Accu','Battery'],
                                    'Airflow': ['AIRFLOW','Airflow'],
                                    'Flow': ['Flow', 'FLOW',],
                                    'Cflow': ['CFLO', 'CFLOW', 'CPAP_Flow',],
                                    'CNEP_pressure': ['cNEP_pressure','CNEP_pressure'],
                                    'EtCo2': [ 'EtCO2', 'ETCO2', 'ETC2'],
                                    'EtCo2_trend': [ 'ETCO2_TREND', 'ETCO2_Trend',],
                                    'TcCo2': ['TcCO2', 'TcpCO2', 'TCCO2_Digital', ],
                                    'C3-M2': ['C3M2','EEG_C3-A2','C3-M2'],
                                    'C4-M1': ['EEG_C4-A1','C4-M1','C4M1'],
                                    'Chest': ['Chest','CHEST','THOR'],
                                    'Chin': ['Chin','CHIN','EMG_Chin','EMG1','EMG_#1','CHIN1'],
                                    'Chin2': ['Chin2','EMG_Chin2','EMG2','CHIN2','EMG_#2'],
                                    'Chin3': ['CHIN3','EMG3', 'EMG_#3'],
                                    'ECG': ['ECG','EKG','ECG1','ECG_I','EKG_#1'],
                                    'ECG1': ['EKG1'],
                                    'ECG2': ['ECG2','ECG_2','ECG_II','EKG2','EKG_#2'],
                                    'F3-M2': ['F3-M2','F3:M2','F3M2','EEG_F3-A2'],
                                    'F4-M1': ['F4-M1', 'F4M1','EEG_F4-A1'],
                                    'FP1': ['FP1','Fp1'],
                                    'FP2': ['FP2','Fp2'],
                                    'Heartrate': ['HR', 'Heartrate', ],
                                    'L_arm': ['Arms-L','L-Arm1'],
                                    'Leak': ['Leak','LEAK','Leak_Total'],
                                    'LAT1': ['L-Leg1','L-LEG1','L-LEG_1','LegsL-Leg1'],
                                    'LAT2':['L-Leg2','L-LEG2','L-LEG_2'],
                                    "LOC_A2": ['EOG_LOC-A2','E1M2'],
                                    'LOC': ['E1','E1_(LEOG)','L-EOG','EOG1'],
                                    'Masseter_1': ['Massater_1',  'Masseter_1',],
                                    'Masseter_2': ['Massater_2', 'Masseter_2',],
                                    'Right-Masseter': ['Massater_1',  'Masseter_1', 'Mass-R', 'Right-Masseter1','Right_Masseter_1'],
                                    'Left-Masseter': ['Mass-L','Left_Masseter_1'],
                                    'Movement': ['Move.', 'Mvmt',],
                                    'NasalOr':['NasOr', 'NasalOr',],
                                    'Nasal_pressure': ['NASAL_PRESSURE','Nasal_Pressure',],
                                    'O2-M1': ['O2-M1', 'O2M1','EEG_O2-A1'],
                                    'O1-M2':['O1-M2', 'O1M2','EEG_O1-A2'],
                                    'CPAP_leak':['CPAP_Leak', 'CPAP_leak'],
                                    'Pleth': ['Pleth', 'Plth','Plesmo',  ],
                                    'Pressure': ['Press', 'Press_Patient', 'Pressure'],
                                    'BodyPosition': [ 'BPOS', 'BPos', 'Body',],
                                    'Pos': ['POS','Pos','Pos.'],
                                    'Pulse': ['Pulse','PULSE','PulseRate'],
                                    'LAT': ['LAT','PLMl','PLMl.','Leg_1','LLEG'],
                                    'RAT': ['RAT','PLMr','PLMr.','Leg_2','RLEG'],
                                    'Resp_Rate': ['RR', 'Resp_Rate'],
                                    'RAT1': ['R-LEG1', 'R-LEG_1', 'R-Leg1'],
                                    'RAT2': ['R-LEG2','R-LEG_2', 'R-Leg2', ],
                                    'R_arm': ['Arms-R','Right_arm','R-Arm1'],
                                    'R_arm2': ['R-Arm2', 'Right_Arm_2'],
                                    "ROC_A2": ['EOG_ROC-A2', 'E2M2'],
                                    'ROC': ['E2', 'E2_(REOG)', 'R-EOG', 'EOG2'],
                                    'Sum': ['Sum','SUM'],
                                    'Snore': ['Snore','SNOR', 'SNORE',],
                                    'Snore_press': ['Pressure_Snore', 'P-Snore',],
                                    'SpO2': [ 'SpO2','SPO2', 'SAO2', 'Sa02', 'SaO2'],
                                    'T3-M2': ['T3M2','EEG_T3-A2'],
                                    'T4-M1': ['T4M1','EEG_T4-A1'],
                                    'TidVol': ['Tidal_Vol','TV','T-Volume'],
                                    'xPAP': ['xPAP_CPAP']
        }
    
        
        self.channel_names =  ['16', '20', '21', '32', 'A1', 'A2', 'ABD', 'ABDM', 'ABDOMEN', 'AIRFLOW', 'Abd', 'AbdDC', 'Abdomen', 'Accu', 'Airflow', 'Arms-L', 'Arms-R', 'BPOS', 'BPos', 
                               'Battery', 'Body', 'BreathRate', 'C-LEAK', 'C3', 'C3-M2', 'C3-P3', 'C3:C4', 'C3M2', 'C4', 'C4-M1', 'C4-P4', 'C4M1', 'CFLO', 'CFLOW', 'CHEST', 'CHIN', 
                               'CHIN1', 'CHIN2', 'CHIN3', 'CNEP_pressure', 'CO2', 'CO2_EndTidal', 'CO2_EndTidal2', 'CPAP_Flow', 'CPAP_Leak', 'CPAP_Pressure', 'CPAP_leak', 'CPAP_raw_flow', 
                               'Cannula', 'Chest', 'ChestDC', 'Chin', 'Chin2', 'Cz', 'DC08', 'DC_Nasal_Canual', 'DIA', 'E1', 'E1M2', 'E1_(LEOG)', 'E2', 'E2M2', 'E2_(REOG)', 'ECG', 'ECG1', 
                               'ECG2', 'ECG_2', 'ECG_I', 'ECG_I2', 'ECG_II', 'ECG_II2', 'ECG_IIHF', 'EEG', 'EEG_A1-A2', 'EEG_A1-A22', 'EEG_C3-A1', 'EEG_C3-A2', 'EEG_C3-A22', 'EEG_C4-A1', 
                               'EEG_C4-A12', 'EEG_C4-A2', 'EEG_F3-A1', 'EEG_F3-A2', 'EEG_F3-A22', 'EEG_F4-A1', 'EEG_F4-A12', 'EEG_F4-A2', 'EEG_F7-A2', 'EEG_F8-A1', 'EEG_Fp1-A2', 
                               'EEG_Fp1-A22', 'EEG_Fp2-A1', 'EEG_Fp2-A12', 'EEG_O1-A1', 'EEG_O1-A2', 'EEG_O1-A22', 'EEG_O2-A1', 'EEG_O2-A12', 'EEG_O2-A2', 'EEG_P3-A2', 'EEG_P4-A1', 
                               'EEG_T3-A2', 'EEG_T4-A1', 'EEG_T5-A2', 'EEG_T6-A1', 'EKG', 'EKG1', 'EKG2', 'EKG_#1', 'EKG_#2', 'EMG', 'EMG1', 'EMG2', 'EMG3', 'EMG_#1', 'EMG_#2', 'EMG_#3', 
                               'EMG_Aux1', 'EMG_Aux12', 'EMG_Aux2', 'EMG_Chin', 'EMG_Chin2', 'EOG1', 'EOG2', 'EOG_LOC-A2', 'EOG_LOC-A22', 'EOG_ROC-A1', 'EOG_ROC-A12', 'EOG_ROC-A2', 
                               'EOG_ROC-A22', 'EPAP', 'ETC2', 'ETCO2', 'ETCO2_Digital', 'ETCO2_TREND', 'ETCO2_Trend', 'ETCO2_Wave', 'Effort_ABD', 'Effort_THO', 'EtCO2', 'ExOb', 'F1M2', 
                               'F2M1', 'F3', 'F3-C3', 'F3-M2', 'F3:M2', 'F3M2', 'F4', 'F4-C4', 'F4-M1', 'F4:F3', 'F4M1', 'F7', 'F7-T3', 'F8', 'F8-T4', 'FLOW', 'FP1', 'FP1-F3', 'FP1-F7', 
                               'FP2', 'FP2-F4', 'FP2-F8', 'Flow', 'Flow_Aux4', 'Flow_Patient', 'Flow_Patient2', 'Flow_Patient3', 'Foot-L', 'Foot-R', 'Fp1', 'Fp2', 'Fp2:Fp1', 'Fpz', 'Fz', 
                               'Graphical_Aux2', 'HR', 'Heartrate', 'IC1-IC2', 'IPAP', 'Impedan', 'L-Arm1', 'L-Arm2', 'L-EOG', 'L-LEG1', 'L-LEG2', 'L-LEG_1', 'L-LEG_2', 'L-Leg1', 'L-Leg2', 
                               'LA1-LA2', 'LAT', 'LAT1-LAT2', 'LEAK', 'LLEG', 'LLeg3', 'LLeg4', 'LOC', 'Leak', 'Leak_Total', 'Left_Masseter_1', 'Left_Masseter_2', 'Leg_1', 'Leg_12', 
                               'Leg_2', 'Leg_22', 'LegsL-Leg1', 'Light', 'M1', 'M2', 'MV', 'Marker', 'Mass-L', 'Mass-R', 'Massater_1', 'Massater_2', 'Masseter_1', 'Masseter_2', 'Min_Vent', 
                               'Move.', 'Mvmt', 'NASAL_PRESSURE', 'NCPT', 'NasOr', 'NasOr2', 'Nasal', 'NasalDC', 'NasalOr', 'NasalSn', 'Nasal_Pressure', 'Nasal_Therm', 'Nonin_sat', 'O1', 
                               'O1-M2', 'O1M2', 'O2', 'O2-M1', 'O2M1', 'Oral-CO2', 'Other', 'Oz', 'P-Snore', 'P3', 'P3-O1', 'P4', 'P4-O2', 'PAP_Leak', 'PAP_Pres', 'PAP_Pt_Flo', 'PAP_TV', 
                               'PLM3.', 'PLM4.', 'PLMl', 'PLMl.', 'PLMr', 'PLMr.', 'POS', 'PPG', 'PTAF', 'PULSE', 'Pes-L1', 'Pes-L2', 'Pes-L3', 'Pes-L4', 'Plesmo', 'Pleth', 'Plth', 'Pos', 
                               'Pos.', 'Press', 'PressCheck', 'Press_Patient', 'Pressure', 'Pressure.1', 'Pressure_Flow', 'Pressure_Snore', 'Pulse', 'PulseRate', 'Pulse_Amp', 'Pz', 'R-Arm1', 
                               'R-Arm2', 'R-EOG', 'R-LEG1', 'R-LEG2', 'R-LEG_1', 'R-LEG_2', 'R-Leg1', 'R-Leg2', 'RAT', 'RAT1-RAT2', 'RIC', 'RLEG', 'RLeg5', 'RLeg6', 'ROC', 'RR', 'ResMed_Flow', 
                               'ResMed_Pressure', 'Resp_Rate', 'Right-Masseter1', 'Right_Arm_2', 'Right_Masseter_1', 'Right_Masseter_2', 'Right_arm', 'SAO2', 'SCM', 'SNOR', 'SNORE', 'SPO2', 
                               'SUB-R_-_V5', 'SUBR-SUBL', 'SUM', 'Sa02', 'SaO2', 'Scalene', 'Snore', 'Snore2', 'SpO2', 'SpO2Sta', 'Sum', 'T-Volume', 'T3', 'T3-T5', 'T3M2', 'T4', 'T4-T6', 'T4M1', 
                               'T5', 'T5-O1', 'T6', 'T6-O2', 'TCCO2_Digital', 'THOR', 'TV', 'TcCO2', 'TcPPG', 'TcSpO2', 'Tcm4CO2', 'TcpCO2', 'Technical', 'TidVol_Instant', 'TidVol_Target', 
                               'Tidal_Vol', 'WAVE', 'WPLM_ST', 'WPLM_ST.1', 'Winx-Oral', 'Winx-Pump', 'cNEP_pressure', 'unused', 'wPLMl', 'wPLMl_Sta', 'wPLMr', 'wPLMr_Sta', 'x', 'x.1', 'x.2', 
                               'xPAP_CPAP', 'xPAP_EPAP', 'xPAP_EPAPMax', 'xPAP_EPAPMin', 'xPAP_IPAP', 'xPAP_IPAPMax', 'xPAP_IPAPMin', 'xPAP_MaxPress', 'xPAP_PSMax', 'xPAP_PSMin']
        
        # ACHTUNG EEG in MSTR00200 nur digital??
        self.channel_types = {'analog': ['NasOr2', 'Pes-L1', 'RAT1-RAT2', 'PAP_Leak', '21', 'F3-M2', 'Fp2:Fp1', 'T4M1', 'NasalSn', 'Chin', 'F4-C4', 'NasalOr', 'ECG_II', 'Leg_12', 'wPLMl_Sta', 
                                         'E2M2', 'Chin2', 'Tidal_Vol', 'Right-Masseter1', 'RLEG', 'Arms-R', 'Snore2', 'SNORE', 'C4-M1', 'C-LEAK', 'O1-M2', 'CO2_EndTidal2', 'ROC', 'Flow_Patient2', 
                                         'EEG_F3-A22', 'LLeg4', 'Press_Patient', 'FP2-F4', 'EEG_T5-A2', 'R-LEG2', 'LAT', 'RAT', 'C3:C4', 'Fp1', 'EEG_T6-A1', 'CPAP_Leak', 'E2_(REOG)', 'Flow', 
                                         'Left_Masseter_1', 'R-LEG_2', 'T3M2', 'F4:F3', 'ABDOMEN', 'Pressure_Flow', 'CHIN3', 'Mass-R', 'CFLO', 'wPLMl', 'IC1-IC2', 'F3-C3', 'CO2_EndTidal', 
                                         'L-Arm2', 'FLOW', 'A1', 'EEG_A1-A2', 'Effort_ABD', 'Leak_Total', 'EEG_A1-A22', 'PLMl.', 'ABD', 'P4', 'E1M2', 'L-LEG_2', 'P3', 'THOR', 'EKG2', 'PLM3.', 
                                         'Winx-Pump', 'CHIN', 'FP1-F3', 'M2', 'L-EOG', 'PTAF', 'Leak', 'Right_Arm_2', 'CO2', 'T4-T6', 'O2M1', 'ECG_I', 'Abd', 'LEAK', 'ECG2', 'EEG_C3-A2', 
                                         'EOG_LOC-A22', 'EEG_P3-A2', 'Flow_Patient', 'Move.', 'F4M1', 'F7-T3', 'F7', 'EMG1', 'MV', 'EMG_Chin2', 'Technical', 'PLMr', 'Foot-L', 'EEG_F3-A2', 
                                         'TV', 'T5', 'TcPPG', 'C3', 'C3-M2', 'SUM', 'Leg_1', '32', 'Right_Masseter_1', 'T3', 'TcSpO2', 'R-Arm1', 'C3M2', 'M1', 'RLeg5', 'R-LEG1', 'EMG_#1', 
                                         'Oral-CO2', 'Plth', 'F3:M2', 'Oz', 'PLM4.', 'RLeg6', 'Cannula', 'O1M2', 'EEG_F7-A2', 'Accu', 'NasOr', 'RIC', 'L-Leg2', 'Fz', 'R-EOG', 'ECG_II2', 'T6', 
                                         'O2', 'EEG_O1-A22', 'SCM', 'Pes-L3', 'Massater_1', 'EMG3', 'F4-M1', 'R-Leg2', 'ECG_2', 'ECG', 'L-LEG_1', 'Heartrate', 'T3-T5', 'Impedan', 'EEG_F4-A12', 
                                         'PLMl', 'SNOR', 'Foot-R', 'P-Snore', 'EEG_Fp1-A2', 'Masseter_2', 'Leg_22', 'L-Arm1', 'LLeg3', 'EOG2', 'Leg_2', 'FP2-F8', 'F4', 'Sum', 'EMG_Chin', 'AIRFLOW', 
                                         'LOC', 'EEG_P4-A1', 'EEG_F4-A1', 'EOG_ROC-A2', 'Cz', 'Left_Masseter_2', 'EEG_T4-A1', 'wPLMr', 'Scalene', 'NasalDC', 'Masseter_1', 'PAP_Pt_Flo', 'CHIN2', 
                                         'EEG_Fp2-A1', 'ECG_IIHF', 'CHEST', 'FP1', 'LAT1-LAT2', 'EEG_C3-A1', 'Mass-L', 'EEG_O2-A2', 'PLMr.', 'EEG_Fp2-A12', 'Pressure', 'EPAP', 'C4-P4', 
                                         'EOG_ROC-A1', 'Fpz', 'A2', 'EMG_#3', 'EEG_O1-A2', 'Snore', 'Fp2', 'PAP_Pres', 'R-LEG_1', 'EMG_Aux1', 'LLEG', 'EOG1', 'Pz', 'Pleth', 'C4', 'F8', 'NCPT', 
                                         'EOG_ROC-A12', 'EEG_O1-A1', 'Plesmo', 'PAP_TV', 'EKG1', 'F2M1', 'CFLOW', 'R-Arm2', 'ECG_I2', 'EEG_C4-A1', 'Nasal_Pressure', 'Chest', 'EMG_Aux12', 
                                         'EMG_#2', '16', 'WPLM_ST', 'P4-O2', 'EEG_O2-A12', 'EEG_F4-A2', 'LA1-LA2', 'PPG', 'F1M2', 'TidVol_Instant', 'R-Leg1', 'Nasal_Therm', 'Massater_2', 'EKG', 
                                         'L-LEG1', 'EMG', 'Press', 'EKG_#1', 'EEG_C3-A22', 'Tcm4CO2', 'E1', 'unused', 'O2-M1', 'T4', 'Arms-L', 'SUB-R_-_V5', 'EEG_C4-A2', 'L-LEG2', 'Right_Masseter_2', 
                                         'ChestDC', 'T6-O2', 'CPAP_Flow', 'EOG_LOC-A2', 'EEG_O2-A1', 'ABDM', 'CHIN1', 'Mvmt', 'Effort_THO', 'EMG_Aux2', 'L-Leg1', 'EEG_F3-A1', 'EEG_F8-A1', 'Pressure_Snore', 
                                         'T5-O1', 'Abdomen', 'NASAL_PRESSURE', 'EOG_ROC-A22', 'Flow_Aux4', 'Pressure.1', 'EEG_Fp1-A22', '20', 'EKG_#2', 'DIA', 'FP1-F7', 'C4M1', 'E2', 'EEG_C4-A12', 'O1', 
                                         'Flow_Patient3', 'LegsL-Leg1', 'ExOb', 'FP2', 'Right_arm', 'Battery', 'ECG1', 'C3-P3', 'AbdDC', 'Airflow', 'SUBR-SUBL', 'Nasal', 'Light', 'P3-O1', 'F3', 'Pes-L2', 
                                         'E1_(LEOG)', 'Pes-L4', 'EMG2', 'Winx-Oral', 'EEG_T3-A2', 'F8-T4', 'RR', 'TcCO2', 'F3M2'],
                            'digital': ['PressCheck', 'CPAP_raw_flow', 'Sa02', 'xPAP_IPAP', 'ETC2', 'T-Volume', 'ETCO2_TREND', 'Pulse', 'Other', 'SpO2Sta', 'ETCO2_Digital', 'Marker', 'TidVol_Target', 
                                        'xPAP_MaxPress', 'CPAP_leak', 'ResMed_Flow', 'x.2', 'PULSE', 'DC_Nasal_Canual', 'BPOS', 'POS', 'xPAP_EPAP', 'BreathRate', 'WPLM_ST.1', 'Graphical_Aux2', 'Pos', 'Body', 
                                        'EtCO2', 'SaO2', 'Min_Vent', 'Resp_Rate', 'SPO2', 'Pos.', 'ResMed_Pressure', 'DC08', 'xPAP_IPAPMax', 'xPAP_EPAPMax', 'WAVE', 'xPAP_PSMin', 'SAO2', 'x.1', 'IPAP', 'EEG', 
                                        'CNEP_pressure', 'Pulse_Amp', 'SpO2', 'TCCO2_Digital', 'xPAP_IPAPMin', 'wPLMr_Sta', 'cNEP_pressure', 'xPAP_PSMax', 'HR', 'xPAP_EPAPMin', 'x', 'Nonin_sat', 'ETCO2', 
                                        'PulseRate', 'TcpCO2', 'xPAP_CPAP', 'ETCO2_Wave', 'ETCO2_Trend', 'CPAP_Pressure', 'BPos']}
        
        
        self.channel_groups = {'eeg_eog': ['C3', 'C3-M2', 'C3-P3', 'C3:C4', 'C3M2', 'C4', 'C4-M1', 'C4-P4', 'C4M1','E1', 'E1M2', 'E1_(LEOG)', 'E2', 'E2M2', 'E2_(REOG)', 'EEG', 'EEG_A1-A2', 'EEG_A1-A22', 'EEG_C3-A1', 'EEG_C3-A2', 'EEG_C3-A22', 'EEG_C4-A1', 
                                            'EEG_C4-A12', 'EEG_C4-A2', 'EEG_F3-A1', 'EEG_F3-A2', 'EEG_F3-A22', 'EEG_F4-A1', 'EEG_F4-A12', 'EEG_F4-A2', 'EEG_F7-A2', 'EEG_F8-A1', 'EEG_Fp1-A2', 
                                            'EEG_Fp1-A22', 'EEG_Fp2-A1', 'EEG_Fp2-A12', 'EEG_O1-A1', 'EEG_O1-A2', 'EEG_O1-A22', 'EEG_O2-A1', 'EEG_O2-A12', 'EEG_O2-A2', 'EEG_P3-A2', 'EEG_P4-A1', 
                                            'EEG_T3-A2', 'EEG_T4-A1', 'EEG_T5-A2', 'EEG_T6-A1', 'EOG1', 'EOG2', 'EOG_LOC-A2', 'R-EOG','L-EOG','EOG_LOC-A22', 'EOG_ROC-A1', 'EOG_ROC-A12', 'EOG_ROC-A2', 
                                            'EOG_ROC-A22', 'F1M2', 'Fp1', 'Fp2', 'Fp2:Fp1', 'Fpz', 'Fz', 'O1', 'Oz', 'P3', 'P3-O1', 'P4', 'P4-O2', 'O2', 'O2-M1', 'O2M1','O1-M2', 'O1M2',
                                            'F2M1', 'F3', 'F3-C3', 'F3-M2', 'F3:M2', 'F3M2', 'F4', 'F4-C4', 'F4-M1', 'F4:F3', 'F4M1', 'F7', 'F7-T3', 'F8', 'F8-T4', 'FP1', 'FP1-F3', 'FP1-F7', 
                                            'FP2', 'FP2-F4', 'FP2-F8', 'LOC','ROC', 'T3', 'T3-T5', 'T3M2', 'T4', 'T4-T6', 'T4M1',  'Pz','Cz','M1', 'M2','T5', 'T5-O1', 'T6', 'T6-O2', ],
                                'emg': ['CHIN', 'CHIN1', 'CHIN2', 'CHIN3', 'Chin', 'Chin2', 'EMG', 'EMG1', 'EMG2', 'EMG3', 'EMG_#1', 'EMG_#2', 'EMG_#3', 'EMG_Aux1', 'EMG_Aux12', 'EMG_Aux2', 
                                        'EMG_Chin', 'EMG_Chin2','Foot-L', 'Foot-R', 'L-Arm1', 'L-Arm2',  'L-LEG1', 'L-LEG2', 'L-LEG_1', 'L-LEG_2', 'L-Leg1', 'L-Leg2', 'LA1-LA2', 
                                        'LAT', 'LAT1-LAT2', 'PLMl', 'PLMl.','PLMr','PLMr.','LLEG', 'LLeg3', 'LLeg4', 'Leg_1', 'Leg_12', 'Leg_2', 'Leg_22', 'LegsL-Leg1','R-Arm1', 'Right_Arm_2',  'Right_arm','Arms-L', 
                                        'Arms-R', 'R-Arm2',   'R-LEG2',  'R-LEG_2','R-Leg2', 'RAT', 'RAT1-RAT2', 'RLEG', 'RLeg5', 'RLeg6','R-LEG1','R-LEG_1', 'R-Leg1','Mass-L', 
                                        'Mass-R', 'Massater_1',  'Masseter_1', 'Massater_2', 'Masseter_2','Left_Masseter_1', 'Left_Masseter_2','Right-Masseter1','Right_Masseter_1', 'Right_Masseter_2',],
                                'ecg': ['ECG', 'ECG1', 'ECG2', 'ECG_2', 'ECG_I', 'ECG_I2', 'ECG_II', 'ECG_II2', 'ECG_IIHF', 'EKG', 'EKG1', 'EKG2', 'EKG_#1', 'EKG_#2',],
                                'thoraco_abdo_resp': ['THOR','ABD', 'ABDM', 'ABDOMEN', 'Abd', 'AbdDC', 'Abdomen','Chest', 'ChestDC','CHEST','Effort_ABD', 'Effort_THO',],
                                'snoring': ['Snore', 'Snore2','SNOR', 'SNORE','Pressure_Snore', 'P-Snore',],
                                'nasal_pressure': ['NASAL_PRESSURE','Nasal_Pressure',],
                                }
        
        
        self.file_extensions = {
                                'psg_ext': '**/*.edf',
                                'ann_ext': '**/*.csv'
                                }
        

    def dataset_paths(self) -> Tuple[str, str]:
        return [
            os.path.join("original", "STAGES PSGs"),
            os.path.join("original", "STAGES PSGs")
        ]
        
    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        function to parse the annotation file of the dataset into sleep stage events with start and duration

        """

        ann_stage_events = []
        ann_df = pd.read_csv(ann_fname,header = 0, sep=',',on_bad_lines='skip', index_col=False)
        ann_Startdatetime = None

        ann_stage_events = []
        for i,row in ann_df.iterrows():
            if not pd.notna(row['Event']):
                continue
            event = row['Event'].strip()
            if event in self.ann2label:
                start = datetime.combine(date(1985,1,1),datetime.strptime(row['Start Time'],"%H:%M:%S").time())

                if ann_Startdatetime == None:
                    ann_Startdatetime = start

                start = int((start - ann_Startdatetime).seconds)
                duration = row['Duration (seconds)']

                # In some subfolder are some epochs with duration of 0 that do not fit into the timeline -> exclude them
                if any(subfolder in os.path.basename(ann_fname) for subfolder in ['GSLH', 'GSSW', 'MSQW','MSTR']) and len(ann_stage_events) > 0:
                    if duration == 0 and start != (ann_stage_events[-1]['Start'] + ann_stage_events[-1]['Duration']):
                        print(row)
                        continue
                
                # In MSTR some epochs missing -> fill with unknown
                if 'MSTR' in ann_fname and len(ann_stage_events) > 0:
                    if start != (ann_stage_events[-1]['Start'] + ann_stage_events[-1]['Duration']):
                        ann_stage_events.append({'Stage':'UnknownStage','Start':ann_stage_events[-1]['Start'] + ann_stage_events[-1]['Duration'],'Duration':start - (ann_stage_events[-1]['Start'] + ann_stage_events[-1]['Duration'])})

                ann_stage_events.append({'Stage': event,
                                        'Start': start,
                                        'Duration': duration})
                
        # In some subfolders/files the duration column is empty for all sleep stages -> calc manually
        if any(part in os.path.basename(ann_fname) for part in ['BOGN', 'STNF00191', 'STNF00233','STNF00261']):
            for i, event in enumerate(ann_stage_events[:-1]):
                # in STNF all REM epochs have a duration of 2592000 -> modify to real duration
                if ann_stage_events[i]['Duration'] == 0 or ann_stage_events[i]['Duration'] == 2592000:
                    ann_stage_events[i]['Duration'] = ann_stage_events[i+1]['Start'] - event['Start']


        return ann_stage_events, ann_Startdatetime
    
    def align_front(self, logger, alignment, pad_values, epoch_duration, delay_sec, signal, labels, fs):

        return self.base_align_front(logger, delay_sec, alignment, pad_values, epoch_duration, signal, labels,fs)

    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(labels) > len(signals):
            return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)

        if len(signals) > len(labels):
            return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)        
    