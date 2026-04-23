import os
import numpy as np

from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("CHAT")
class CHAT(BaseDataset):
    """CHAT (CHILDHOOD ADENOTONSILLECTOMY TRIAL) dataset"""
    
    def __init__(self):
        super().__init__("CHAT","CHAT - Childhood Adenotonsillectomy Trial")
  
    def _setup_dataset_config(self):
        self.ann2label = {
            "Wake": "W",
            "Stage 1 sleep": "N1",
            "Stage 2 sleep": "N2",
            "Stage 3 sleep": "N3",
            "Stage 4 sleep": "N3", # Follow AASM Manual
            "REM sleep": "REM",
            "Unscored": "UNK",
        }

        self.intra_dataset_mapping = {
            "Airflow": ['AIR', 'AIRFLOW', 'Airflow', 'Airlfow', 'Airflow1'],
            "Cchin": ['CCHIN', 'CChin','Cchin', 'cchin',],
            "Chin": ['CHIN', 'Chin'],
            "CAP": ['CAP', 'Cap', 'cap'],
            "CPAP": ['CFLOW', 'CFlow', 'CPAP Flow'],
            "CPRESS": ['C-Pres', 'CPress','CPAP Pressure'],
            "CannulaFlow": ['CANNULAFLOW', 'CannnulaFlow', 'CannulaFLow', 'CannulaFlow', 'CannulaFow', 'Cannulaflow'],
            "Chest": ['CHEST','Chest', ],
            "ECG_": ['ECG','EKG'],  # have to map these uncommon labels to different name because ECG1 (much more common)will be saved as ECG
            "ECG2": ['ECG2', 'ECg2'],
            "ETCO2": ['ETC02', 'ETCO2','EtC02', 'EtCO2', 'EtCo2',],
            "F7": ['F7', 'F7-'],
            "HEARTRATE": ['HEARTRATE', 'HR', 'Hr'],     # have to map these uncommon labels to different name because DHR (much more common) will be saved as HEARTRATE
            "Lchin": ['LCHIN', 'LChin', 'Lchin'],
            "Leak": ['LEAK', 'Leak'],
            "Lleg": ['LLEG','LLeg','Lleg','LLEG1','LLeg1','Lleg1','lleg1'],
            "Lleg2": ['LLEG2','LLeg2','Lleg2'],
            "Masimo": ['MASIMO', 'Masimo'],
            "M2": ['M2', 'm2'],
            "Oral": ['ORAL', 'Oral','oral'],
            "Pap": ['PAP', 'Pap'],
            "Pleth": ['PLETHMASIMO','PletMasimo', 'PlethMasimo', 'Plethmasimo','PlethMasino', 'PlethNellcor','Pleth','Plesmo'],
            "ProTechPosition": ['Pro Tech Positio', 'ProTech Positio', 'ProTechPositione'],
            "PHOT": ['PHOT', 'PHOT-NoRef'],
            "Pulse": ['PULSE', 'Pulse','PulseMasimo', 'Pulsemasimo', 'pulsemasimo',  'PulseNonin'],
            "Position": ['Position', 'Positon', 'Postion', 'Postition' ],
            "POSdc": ['POS','POSdc', 'POSdc1'],
            "Rchin": ['RCHIN', 'RChin', 'Rchin','Rchiin'],
            "Rleg": ['RLEG','RLeg','Rleg','RLEG1','RLeg1','Rleg1'],
            "Rleg2": ['RLEG2','RLeg2','Rleg2'],
            "Snore 2": ['SNORE','SNORE 2', 'Snore 2','SNORE2','Snore2'],
            "SaO2": ['SA02nonin', 'SAO2', 'SAO2External', 'SAO2ninon', 'SAO2noin', 'SAO2nonin', 'SaO2', 'SaO2Nonin'],
            "SaO2Masimo": ['SAO2masimo', 'Sa02masimo','SA02masimo'],
            "SaO2Nellcor": ['SAO2Nellcor', 'SaO2Nellcor'],
            "SpO2": ['SpO2', 'SpO2 BB'],
            "SUM": ['SUM', 'Sum'],
            "TcCO2": ['TCCO2', 'TcCO2'],
            "NotUsed": ['NotUsed', 'notused'],
        }

        self.inter_dataset_mapping = {
            "Airflow": self.Mapping(self.TTRef.AIRFLOW, None),
            "ABD": self.Mapping(self.TTRef.ABDOMINAL, None),
            "Cchin": self.Mapping(self.TTRef.EMG_CHIN, self.TTRef.Fpz),
            "Lchin": self.Mapping(self.TTRef.EMG_LCHIN, self.TTRef.Fpz),
            "Rchin": self.Mapping(self.TTRef.EMG_RCHIN, self.TTRef.Fpz),
            "Lleg": self.Mapping(self.TTRef.EMG_LLEG, self.TTRef.Fpz),
            "Rleg": self.Mapping(self.TTRef.EMG_RLEG, self.TTRef.Fpz),
            "DHR": self.Mapping(self.TTRef.HR, None),
            "E1": self.Mapping(self.TTRef.EL, self.TTRef.Fpz),
            "E2": self.Mapping(self.TTRef.ER, self.TTRef.Fpz),
            "F3": self.Mapping(self.TTRef.F3, self.TTRef.Fpz),
            "F4": self.Mapping(self.TTRef.F4, self.TTRef.Fpz),
            "C3": self.Mapping(self.TTRef.C3, self.TTRef.Fpz),
            "C4": self.Mapping(self.TTRef.C4, self.TTRef.Fpz),
            "T3": self.Mapping(self.TTRef.T7, self.TTRef.Fpz),
            "T4": self.Mapping(self.TTRef.T8, self.TTRef.Fpz),
            "O1": self.Mapping(self.TTRef.O1, self.TTRef.Fpz),
            "O2": self.Mapping(self.TTRef.O2, self.TTRef.Fpz),
            "M1": self.Mapping(self.TTRef.LPA, self.TTRef.Fpz),
            "M2": self.Mapping(self.TTRef.RPA, self.TTRef.Fpz),
            "F7": self.Mapping(self.TTRef.F7, self.TTRef.Fpz),
            "F8": self.Mapping(self.TTRef.F8, self.TTRef.Fpz),
            "Fz": self.Mapping(self.TTRef.Fz, self.TTRef.Fpz),
            "Fp2": self.Mapping(self.TTRef.Fp2, self.TTRef.Fpz),
            "P3": self.Mapping(self.TTRef.P3, self.TTRef.Fpz),
            "P4": self.Mapping(self.TTRef.P4, self.TTRef.Fpz),
            "T5": self.Mapping(self.TTRef.P7, self.TTRef.Fpz),
            "T6": self.Mapping(self.TTRef.P8, self.TTRef.Fpz),
            "Fp1": self.Mapping(self.TTRef.Fp1, self.TTRef.Fpz),
            "Fpz": self.Mapping(self.TTRef.Fpz, None),
            "FZ": self.Mapping(self.TTRef.Fz, self.TTRef.Fpz),
            "ECG1": self.Mapping(self.TTRef.ECG, None),
            "Snore": self.Mapping(self.TTRef.SNORE, None),
            "Position": self.Mapping(self.TTRef.POSITION, None),
            "SaO2": self.Mapping(self.TTRef.SPO2, None),
            "Chest": self.Mapping(self.TTRef.THORACIC, None),
            "CPAP Flow": self.Mapping(self.TTRef.CPAP, None),
        }
        

        self.channel_names = ['ABD', 'AIR', 'AIRFLOW', 'Airflow', 'Airflow1', 'Airflow2', 'Airlfow', 'BPOSdc1', 'BPOSdc2', 'BciCap', 'BciEtCO2', 'Braebon', 'Braebon Bipolar', 
                              'Braebon Body P', 'C-Pres', 'C3', 'C3-A2', 'C4', 'C4-A1', 'CANNULAFLOW', 'CAP', 'CCHIN', 'CChin', 'CFLOW', 'CFlow', 'CHEST', 'CHIN', 'CPAP', 'CPAP Flow', 
                              'CPAP Leak', 'CPAP Pressure', 'CPAP Volume', 'CPress', 'CannnulaFlow', 'CannulaFLow', 'CannulaFlow', 'CannulaFow', 'Cannulaflow', 'Cap', 'Cchin', 'Chest', 
                              'Chin', 'Compumedics', 'Compumedics Body', 'Cz', 'DC1', 'DC10', 'DC2', 'DC2-DCRef', 'DC3', 'DC4', 'DC4-DCRef', 'DC5', 'DC5-DCRef', 'DC6', 'DC6-DCRef', 
                              'DC7', 'DC7-DCRef', 'DC8', 'DC9', 'DHR', 'E1', 'E2', 'ECG', 'ECG1', 'ECG2', 'ECG3', 'ECg2', 'EKG', 'ETC02', 'ETCO2', 'EXTSAT', 'Echin', 'EtC02', 'EtCO2', 
                              'EtCo2', 'Event', 'F3', 'F3-A2', 'F4', 'F4-A1', 'F7', 'F7-', 'F8', 'FLOW', 'FZ', 'Flow DC7', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'Gravity', 'Gravity X', 'Gravity Y', 
                              'HEARTRATE', 'HR', 'Hr', 'LAT', 'LCHIN', 'LChin', 'LChin-Rchin', 'LEAK', 'LLEG', 'LLEG1', 'LLEG2', 'LLeg', 'LLeg1', 'LLeg2', 'Lchin', 'Leak', 'Light', 
                              'Lleg', 'Lleg1', 'Lleg2', 'M1', 'M2', 'MASIMO', 'MaO2', 'ManPos', 'Masimo', 'NPAF', 'NotUsed', 'O1', 'O1-A2', 'O2', 'O2-A1', 'ORAL', 'OXER', 'OXER-DCRef', 
                              'Oral', 'OxSTAT', 'P3', 'P4', 'PAP', 'PHOT', 'PHOT-NoRef', 'PLETHMASIMO', 'PN', 'PNEUMFLOW', 'POS', 'POSdc', 'POSdc1', 'PR', 'PULSE', 'PWF', 'PaO2', 'Pap', 
                              'Plesmo', 'PletMasimo', 'Pleth', 'PlethMasimo', 'PlethMasino', 'PlethNellcor', 'Plethmasimo', 'Pluse', 'Position', 'Positon', 'Postion', 'Postition', 'Pro', 
                              'Pro Tech Positio', 'ProTech Positio', 'ProTechPositione', 'Pulse', 'PulseMasimo', 'PulseNellcor', 'PulseNonin', 'Pulsemasimo', 'Pz', 'RAT', 'RCHIN', 
                              'RChin', 'REF', 'REF X1', 'REFX1', 'RLEG', 'RLEG1', 'RLEG2', 'RLeg', 'RLeg1', 'RLeg2', 'RR', 'Rchiin', 'Rchin', 'Rleg', 'Rleg1', 'Rleg2', 'SA02masimo', 
                              'SA02nonin', 'SAO2', 'SAO2External', 'SAO2Nellcor', 'SAO2masimo', 'SAO2ninon', 'SAO2noin', 'SAO2nonin', 'SNORE', 'SNORE 2', 'SNORE2', 'SUM', 'Sa02masimo', 
                              'SaO2', 'SaO2Nellcor', 'SaO2Nonin', 'SenTec', 'SenTec CO2', 'Snore', 'Snore 2', 'Snore2', 'SpO2', 'SpO2 BB', 'Sum', 'T3', 'T3-A2', 'T4', 'T4-A1', 'T5', 'T6', 
                              'TCCO2', 'TcCO2', 'X1 DC11', 'X1 DC12', 'X1 DC5', 'X1 DC6', 'X1 DC8', 'X4-Y4', 'X8-Y8', 'XFlow', 'XSum', 'cap', 'cchin', 'lleg1', 'm2', 'maskflow', 'notused', 
                              'oral', 'pulsemasimo']
        
        
        self.channel_types = {'analog': ['ABD', 'AIR', 'AIRFLOW', 'Airflow', 'Airflow1', 'Airflow2', 'Airlfow', 'BPOSdc1', 'BPOSdc2', 'BciCap', 'BciEtCO2', 'C-Pres', 'C3', 'C3-A2', 'C4', 
                                         'C4-A1', 'CANNULAFLOW', 'CAP', 'CCHIN', 'CChin', 'CFLOW', 'CFlow', 'CHEST', 'CHIN', 'CPAP', 'CPAP Flow', 'CPAP Pressure', 'CPAP Volume', 'CPress', 
                                         'CannnulaFlow', 'CannulaFLow', 'CannulaFlow', 'CannulaFow', 'Cannulaflow', 'Cap', 'Cchin', 'Chest', 'Chin', 'Compumedics', 'Compumedics Body', 'Cz', 
                                         'DC1', 'DC10', 'DC2', 'DC2-DCRef', 'DC3', 'DC4', 'DC4-DCRef', 'DC5-DCRef', 'DC6', 'DC6-DCRef', 'DC7', 'DC7-DCRef', 'DC8', 'DC9', 'DHR', 'E1', 'E2', 
                                         'ECG', 'ECG1', 'ECG2', 'ECG3', 'ECg2', 'EKG', 'ETC02', 'ETCO2', 'EXTSAT', 'Echin', 'EtC02', 'EtCO2', 'EtCo2', 'F3', 'F3-A2', 'F4', 'F4-A1', 'F7', 
                                         'F7-', 'F8', 'FLOW', 'FZ', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'Gravity', 'Gravity X', 'Gravity Y', 'LAT', 'LCHIN', 'LChin', 'LChin-Rchin', 'LEAK', 'LLEG', 
                                         'LLEG1', 'LLEG2', 'LLeg', 'LLeg1', 'LLeg2', 'Lchin', 'Leak', 'Lleg', 'Lleg1', 'Lleg2', 'M1', 'M2', 'MASIMO', 'MaO2', 'Masimo', 'NPAF', 'NotUsed', 
                                         'O1', 'O1-A2', 'O2', 'O2-A1', 'ORAL', 'Oral', 'P3', 'P4', 'PAP', 'PHOT', 'PHOT-NoRef', 'PLETHMASIMO', 'PN', 'PNEUMFLOW', 'POS', 'POSdc', 'POSdc1', 
                                         'PWF', 'PaO2', 'Pap', 'PletMasimo', 'Pleth', 'PlethMasimo', 'PlethMasino', 'PlethNellcor', 'Plethmasimo', 'Position', 'Positon', 'Postition', 'Pro', 
                                         'Pro Tech Positio', 'ProTech Positio', 'ProTechPositione', 'Pulse', 'PulseNonin', 'Pz', 'RAT', 'RCHIN', 'RChin', 'REF', 'REF X1', 'REFX1', 'RLEG', 
                                         'RLEG1', 'RLEG2', 'RLeg', 'RLeg1', 'RLeg2', 'RR', 'Rchiin', 'Rchin', 'Rleg', 'Rleg1', 'Rleg2', 'SA02masimo', 'SA02nonin', 'SAO2', 'SAO2External', 
                                         'SAO2masimo', 'SAO2nonin', 'SNORE', 'SNORE 2', 'SNORE2', 'SUM', 'Sa02masimo', 'SaO2', 'SaO2Nonin', 'Snore', 'Snore 2', 'Snore2', 'Sum', 'T3', 'T3-A2', 
                                         'T4', 'T4-A1', 'T5', 'T6', 'TCCO2', 'TcCO2', 'X4-Y4', 'X8-Y8', 'cap', 'cchin', 'lleg1', 'm2', 'maskflow', 'notused', 'oral'], 
                              'digital': ['Braebon', 'Braebon Bipolar', 'Braebon Body P', 'CPAP Leak', 'DC5', 'Event', 'Flow DC7', 'HEARTRATE', 'HR', 'Hr', 'Light', 'ManPos', 'OXER', 
                              'OXER-DCRef', 'OxSTAT', 'PR', 'PULSE', 'Plesmo', 'Pluse', 'Postion', 'PulseMasimo', 'PulseNellcor', 'Pulsemasimo', 'SAO2Nellcor', 'SAO2ninon', 'SAO2noin', 
                              'SaO2Nellcor', 'SenTec', 'SenTec CO2', 'SpO2', 'SpO2 BB', 'X1 DC11', 'X1 DC12', 'X1 DC5', 'X1 DC6', 'X1 DC8', 'XFlow', 'XSum', 'pulsemasimo']}
    
        self.channel_groups = {
            'eeg_eog': ['C3', 'C3-A2', 'C4', 'C4-A1', 'F3', 'F3-A2', 'F4', 'F4-A1', 'F7', 'F8', 'FZ', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'M1', 'M2','m2','O1', 'O1-A2', 'O2', 'O2-A1','P3','P4','T3','T3-A2','T4','T4-A1','T5','T6','Cz'],
            'emg': ['CCHIN', 'CChin','Cchin', 'cchin','LCHIN', 'LChin', 'Lchin','LLEG','LLeg','Lleg','LLEG1','LLeg1','Lleg1','lleg1','LLEG2','LLeg2','Lleg2','RCHIN', 'RChin', 'Rchin','Rchiin','RLEG','RLeg','Rleg','RLEG1','RLeg1','Rleg1','RLEG2','RLeg2','Rleg2' ],
            'ecg':  ['ECG', 'ECG1', 'ECG2', 'ECG3', 'EKG','ECg2'],
            'thoraco_abdo_resp': ['ABD', 'AIR', 'AIRFLOW', 'Airflow', 'Airlfow','Airflow1','Airflow2','CHEST','Chest',],
            'nasal_pressure': ['CANNULAFLOW', 'CannnulaFlow', 'CannulaFLow', 'CannulaFlow', 'CannulaFow', 'Cannulaflow'],
            'snoring': ['SNORE', 'Snore','SNORE 2', 'Snore 2','SNORE2','Snore2']
        }        
        
        self.file_extensions = {
            'psg_ext': '*/*.edf',
            'ann_ext': '*/*-nsrr.xml'
        }

    def get_light_times(self, logger, psg_fname):
        if "Light" in self._file_handler.get_channels(logger, psg_fname):

            light_data = self._file_handler.get_signal_data(logger, psg_fname, "Light")

            light_signal = light_data["signal"]
                
            # Lights Off when light signal is 1
            light_off_indices = np.flatnonzero(light_signal == 1)

            if light_off_indices.size > 0:
                # First occurrence of light off (1)
                light_off_idx = light_off_indices[0]
                lights_off_sec = light_off_idx / light_data["sampling_rate"]

                # Last occurrence of light off (1) 
                light_on_idx = light_off_indices[-1] + 1
                lights_on_sec = light_on_idx / light_data["sampling_rate"]
            else:
                lights_off_sec = None
                lights_on_sec = None
            
            return lights_off_sec, lights_on_sec
        
        else:
            logger.info(f"Light channel not found in {os.path.basename(psg_fname)}. Cannot determine light on/off times")
            return None, None