from datasets.base import BaseDataset
from datasets.registry import register_dataset
from datasets.file_handlers import BRAINVISIONHandler

@register_dataset("SIEFERT")
class SIEFERT(BaseDataset):
    """Siefert2024 (ds005121) dataset"""

    def __init__(self):
        super().__init__("SIEFERT","Siefert2024 (ds005121)", keep_folder_structure=False)

        self._file_handler = BRAINVISIONHandler()
    
    def _setup_dataset_config(self):
        self.ann2label = {} # no annotations available
        
        self.channel_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 
                              'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 
                              'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 
                              'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8', 'SW filtered']
        
        # See https://doi.org/10.1523/JNEUROSCI.0022-24.2024 and _eeg.json files
        self.inter_dataset_mapping = {
            'Fp1': self.Mapping(self.TTRef.Fp1, self.TTRef.FCz),
            'Fz': self.Mapping(self.TTRef.Fz, self.TTRef.FCz),
            'F3': self.Mapping(self.TTRef.F3, self.TTRef.FCz),
            'F7': self.Mapping(self.TTRef.F7, self.TTRef.FCz),
            'FT9': self.Mapping(self.TTRef.FT9, self.TTRef.FCz),
            'FC5': self.Mapping(self.TTRef.FC5, self.TTRef.FCz),
            'FC1': self.Mapping(self.TTRef.FC1, self.TTRef.FCz),
            'C3': self.Mapping(self.TTRef.C3, self.TTRef.FCz),
            'T7': self.Mapping(self.TTRef.EMG_LCHIN, self.TTRef.FCz),   # not sure if left or right
            'TP9': self.Mapping(self.TTRef.LPA, self.TTRef.FCz),    # not sure if left or right
            'CP5': self.Mapping(self.TTRef.CP5, self.TTRef.FCz),
            'CP1': self.Mapping(self.TTRef.CP1, self.TTRef.FCz),
            'Pz': self.Mapping(self.TTRef.Pz, self.TTRef.FCz),
            'P3': self.Mapping(self.TTRef.P3, self.TTRef.FCz),
            'P7': self.Mapping(self.TTRef.P7, self.TTRef.FCz),
            'O1': self.Mapping(self.TTRef.O1, self.TTRef.FCz),
            'Oz': self.Mapping(self.TTRef.Oz, self.TTRef.FCz),
            'O2': self.Mapping(self.TTRef.O2, self.TTRef.FCz),
            'P4': self.Mapping(self.TTRef.P4, self.TTRef.FCz),
            'P8': self.Mapping(self.TTRef.P8, self.TTRef.FCz),
            'TP10': self.Mapping(self.TTRef.RPA, self.TTRef.FCz),   #  not sure if left or right
            'CP6': self.Mapping(self.TTRef.CP6, self.TTRef.FCz),
            'CP2': self.Mapping(self.TTRef.CP2, self.TTRef.FCz),
            'Cz': self.Mapping(self.TTRef.Cz, self.TTRef.FCz),
            'C4': self.Mapping(self.TTRef.C4, self.TTRef.FCz),
            'T8': self.Mapping(self.TTRef.EMG_RCHIN, self.TTRef.FCz),   # not sure if left or right
            'FT10': self.Mapping(self.TTRef.FT10, self.TTRef.FCz),
            'FC6': self.Mapping(self.TTRef.FC6, self.TTRef.FCz),
            'FC2': self.Mapping(self.TTRef.FC2, self.TTRef.FCz),
            'F4': self.Mapping(self.TTRef.F4, self.TTRef.FCz),
            'F8': self.Mapping(self.TTRef.F8, self.TTRef.FCz),
            'Fp2': self.Mapping(self.TTRef.Fp2, self.TTRef.FCz),
            'AF7': self.Mapping(self.TTRef.AF7, self.TTRef.FCz),
            'AF3': self.Mapping(self.TTRef.AF3, self.TTRef.FCz),
            'AFz': self.Mapping(self.TTRef.AFz, self.TTRef.FCz),
            'F1': self.Mapping(self.TTRef.F1, self.TTRef.FCz),
            'F5': self.Mapping(self.TTRef.F5, self.TTRef.FCz),
            'FT7': self.Mapping(self.TTRef.FT7, self.TTRef.FCz),
            'FC3': self.Mapping(self.TTRef.FC3, self.TTRef.FCz),
            'FCz': self.Mapping(self.TTRef.FCz, None),
            'C1': self.Mapping(self.TTRef.C1, self.TTRef.FCz),
            'C5': self.Mapping(self.TTRef.C5, self.TTRef.FCz),
            'TP7': self.Mapping(self.TTRef.TP7, self.TTRef.FCz),
            'CP3': self.Mapping(self.TTRef.CP3, self.TTRef.FCz),
            'P1': self.Mapping(self.TTRef.P1, self.TTRef.FCz),
            'P5': self.Mapping(self.TTRef.P5, self.TTRef.FCz),
            'PO7': self.Mapping(self.TTRef.PO7, self.TTRef.FCz),
            'PO3': self.Mapping(self.TTRef.PO3, self.TTRef.FCz),
            'POz': self.Mapping(self.TTRef.POz, self.TTRef.FCz),
            'PO4': self.Mapping(self.TTRef.PO4, self.TTRef.FCz),
            'PO8': self.Mapping(self.TTRef.PO8, self.TTRef.FCz),
            'P6': self.Mapping(self.TTRef.P6, self.TTRef.FCz),
            'P2': self.Mapping(self.TTRef.P2, self.TTRef.FCz),
            'CPz': self.Mapping(self.TTRef.CPz, self.TTRef.FCz),
            'CP4': self.Mapping(self.TTRef.CP4, self.TTRef.FCz),
            'TP8': self.Mapping(self.TTRef.TP8, self.TTRef.FCz),
            'C6': self.Mapping(self.TTRef.C6, self.TTRef.FCz),
            'C2': self.Mapping(self.TTRef.C2, self.TTRef.FCz),
            'FC4': self.Mapping(self.TTRef.FC4, self.TTRef.FCz),
            'FT8': self.Mapping(self.TTRef.FT8, self.TTRef.FCz),
            'F6': self.Mapping(self.TTRef.F6, self.TTRef.FCz),
            'F2': self.Mapping(self.TTRef.F2, self.TTRef.FCz),
            'AF4': self.Mapping(self.TTRef.AF4, self.TTRef.FCz),
            'AF8': self.Mapping(self.TTRef.AF8, self.TTRef.FCz),
            'SW filtered': self.Mapping(self.TTRef.Fz, None),   #Fz referenced to average of linked mastoids and filtered, used for SO detection
        }
        
        self.channel_types = {'analog': ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 
                                         'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 
                                         'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 
                                         'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8', 'SW filtered'], 
                            'digital': []}
        
        
        self.channel_groups = {'eeg_eog': ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 
                                         'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 
                                         'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 
                                         'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8', 'SW filtered'],
                                'emg': ['T7','T8',]}
                
        self.file_extensions = {'psg_ext': '**/*.vhdr',
                                'ann_ext': '.'} # no annotations
        

    def dataset_paths(self):
        return ['', '']