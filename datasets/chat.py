from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("CHAT")
class CHAT(BaseDataset):
    """CHAT (CHILDHOOD ADENOTONSILLECTOMY TRIAL) dataset"""
    
    def __init__(self):
        super().__init__("CHAT","CHAT - Childhood Adenotonsillectomy Trial")
  
    def _setup_dataset_config(self):
        self.ann2label = {
            "Wake": 0,
            "Stage 1 sleep": 1,
            "Stage 2 sleep": 2,
            "Stage 3 sleep": 3,
            # "Stage 4 sleep": 3, # Follow AASM Manual
            "REM sleep": 4,
            # "Unscored": 6,
        }

        self.intra_dataset_mapping = {
            "Airflow": ['AIR', 'AIRFLOW', 'Airflow', 'Airlfow'],
            "Cchin": ['CCHIN', 'CChin','Cchin', 'cchin',],
            "Cflow": ['CFLOW', 'CFlow'],
            "CannulaFlow": ['CANNULAFLOW', 'CannnulaFlow', 'CannulaFLow', 'CannulaFlow', 'CannulaFow', 'Cannulaflow'],
            "Chest": ['CHEST','Chest', ],
            "ECG": ['ECG','EKG'],
            "ECG2": ['ECG2', 'ECg2'],
            "ETCO2": ['ETC02', 'ETCO2','EtC02', 'EtCO2', 'EtCo2',],
            "HR": ['HEARTRATE', 'HR', 'Hr'],
            "Lchin": ['LCHIN', 'LChin', 'Lchin'],
            "Lleg": ['LLEG','LLeg','Lleg'],
            "Lleg1": ['LLEG1','LLeg1','Lleg1','lleg1'],
            "Lleg2": ['LLEG2','LLeg2','Lleg2'],
            "Masimo": ['MASIMO', 'Masimo'],
            "M2": ['M2', 'm2'],
            "Oral": ['ORAL', 'Oral','oral'],
            "PlethMasimo": ['PLETHMASIMO','PletMasimo', 'PlethMasimo', 'Plethmasimo'],
            "Pulse": ['PULSE', 'Pulse'],
            "Position": ['Position', 'Positon', 'Postion', 'Postition','POS', ],
            "Rchin": ['RCHIN', 'RChin', 'Rchin','Rchiin'],
            "Rleg": ['RLEG','RLeg','Rleg'],
            "Rleg1": ['RLEG1','RLeg1','Rleg1'],
            "Rleg2": ['RLEG2','RLeg2','Rleg2'],
            "Snore": ['SNORE', 'Snore'],
            "Snore 2": ['SNORE 2', 'Snore 2','SNORE2','Snore2'],
            "SpO2": ['SpO2', 'SaO2'],
            "TcCO2": ['TCCO2', 'TcCO2'],
            "NotUsed": ['NotUsed', 'notused'],
        }


        self.inter_dataset_mapping = {
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
            'nasal_pressure': [],
            'snoring': ['SNORE', 'Snore','SNORE 2', 'Snore 2','SNORE2','Snore2']
        }        
        
        self.file_extensions = {
            'psg_ext': '*/*.edf',
            'ann_ext': '*/*-nsrr.xml'
        }