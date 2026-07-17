from datasets.base import BaseDataset
from datasets.registry import register_dataset
from datasets.file_handlers import BRAINVISIONHandler

@register_dataset("RCEEG")
class RCEEG(BaseDataset):
    """RCEEG - A test-retest resting and cognitive state EEG dataset (ds004148) dataset"""

    def __init__(self):
        super().__init__("RCEEG","RCEEG - A test-retest resting and cognitive state EEG dataset (ds004148)", keep_folder_structure=False)

        self._file_handler = BRAINVISIONHandler()
    
    def _setup_dataset_config(self):
        self.ann2label = {
                            "Wake": "W",
                          }
        
        self.channel_names = ['Fp1', 'AF3', 'AF7', 'Fz', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3', 'FC5', 'FT7', 'Cz', 'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3', 
        'CP5', 'TP7', 'TP9', 'Pz', 'P1', 'P3', 'P5', 'P7', 'PO3', 'PO7', 'Oz', 'O1', 'Fpz', 'Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4', 
        'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P2', 'P4', 'P6', 'P8', 'POz', 'PO4', 'PO8', 'O2', 'Cpz', 'FPz']
        
        # FCz was used as online reference (doi: 10.1038/s41597-022-01607-9)
        self.inter_dataset_mapping = {
            "Fp1": self.Mapping(self.TTRef.Fp1, self.TTRef.FCz),
            "AF3": self.Mapping(self.TTRef.AF3, self.TTRef.FCz),
            "AF7": self.Mapping(self.TTRef.AF7, self.TTRef.FCz),
            "Fz": self.Mapping(self.TTRef.Fz, self.TTRef.FCz),
            "F1": self.Mapping(self.TTRef.F1, self.TTRef.FCz),
            "F3": self.Mapping(self.TTRef.F3, self.TTRef.FCz),
            "F5": self.Mapping(self.TTRef.F5, self.TTRef.FCz),
            "F7": self.Mapping(self.TTRef.F7, self.TTRef.FCz),
            "FC1": self.Mapping(self.TTRef.FC1, self.TTRef.FCz),
            "FC3": self.Mapping(self.TTRef.FC3, self.TTRef.FCz),
            "FC5": self.Mapping(self.TTRef.FC5, self.TTRef.FCz),
            "FT7": self.Mapping(self.TTRef.FT7, self.TTRef.FCz),
            "Cz": self.Mapping(self.TTRef.Cz, self.TTRef.FCz),
            "C1": self.Mapping(self.TTRef.C1, self.TTRef.FCz),
            "C3": self.Mapping(self.TTRef.C3, self.TTRef.FCz),
            "C5": self.Mapping(self.TTRef.C5, self.TTRef.FCz),
            "T7": self.Mapping(self.TTRef.T7, self.TTRef.FCz),
            "CP1": self.Mapping(self.TTRef.CP1, self.TTRef.FCz),
            "CP3": self.Mapping(self.TTRef.CP3, self.TTRef.FCz),
            "CP5": self.Mapping(self.TTRef.CP5, self.TTRef.FCz),
            "TP7": self.Mapping(self.TTRef.TP7, self.TTRef.FCz),
            "TP9": self.Mapping(self.TTRef.TP9, self.TTRef.FCz),
            "Pz": self.Mapping(self.TTRef.Pz, self.TTRef.FCz),
            "P1": self.Mapping(self.TTRef.P1, self.TTRef.FCz),
            "P3": self.Mapping(self.TTRef.P3, self.TTRef.FCz),
            "P5": self.Mapping(self.TTRef.P5, self.TTRef.FCz),
            "P7": self.Mapping(self.TTRef.P7, self.TTRef.FCz),
            "PO3": self.Mapping(self.TTRef.PO3, self.TTRef.FCz),
            "PO7": self.Mapping(self.TTRef.PO7, self.TTRef.FCz),
            "Oz": self.Mapping(self.TTRef.Oz, self.TTRef.FCz),
            "O1": self.Mapping(self.TTRef.O1, self.TTRef.FCz),
            "Fpz": self.Mapping(self.TTRef.Fpz, self.TTRef.FCz),
            "Fp2": self.Mapping(self.TTRef.Fp2, self.TTRef.FCz),
            "AF4": self.Mapping(self.TTRef.AF4, self.TTRef.FCz),
            "AF8": self.Mapping(self.TTRef.AF8, self.TTRef.FCz),
            "F2": self.Mapping(self.TTRef.F2, self.TTRef.FCz),
            "F4": self.Mapping(self.TTRef.F4, self.TTRef.FCz),
            "F6": self.Mapping(self.TTRef.F6, self.TTRef.FCz),
            "F8": self.Mapping(self.TTRef.F8, self.TTRef.FCz),
            "FC2": self.Mapping(self.TTRef.FC2, self.TTRef.FCz),
            "FC4": self.Mapping(self.TTRef.FC4, self.TTRef.FCz),
            "FC6": self.Mapping(self.TTRef.FC6, self.TTRef.FCz),
            "FT8": self.Mapping(self.TTRef.FT8, self.TTRef.FCz),
            "C2": self.Mapping(self.TTRef.C2, self.TTRef.FCz),
            "C4": self.Mapping(self.TTRef.C4, self.TTRef.FCz),
            "C6": self.Mapping(self.TTRef.C6, self.TTRef.FCz),
            "T8": self.Mapping(self.TTRef.T8, self.TTRef.FCz),
            "CPz": self.Mapping(self.TTRef.CPz, self.TTRef.FCz),
            "CP2": self.Mapping(self.TTRef.CP2, self.TTRef.FCz),
            "CP4": self.Mapping(self.TTRef.CP4, self.TTRef.FCz),
            "CP6": self.Mapping(self.TTRef.CP6, self.TTRef.FCz),
            "TP8": self.Mapping(self.TTRef.TP8, self.TTRef.FCz),
            "TP10": self.Mapping(self.TTRef.TP10, self.TTRef.FCz),
            "P2": self.Mapping(self.TTRef.P2, self.TTRef.FCz),
            "P4": self.Mapping(self.TTRef.P4, self.TTRef.FCz),
            "P6": self.Mapping(self.TTRef.P6, self.TTRef.FCz),
            "P8": self.Mapping(self.TTRef.P8, self.TTRef.FCz),
            "POz": self.Mapping(self.TTRef.POz, self.TTRef.FCz),
            "PO4": self.Mapping(self.TTRef.PO4, self.TTRef.FCz),
            "PO8": self.Mapping(self.TTRef.PO8, self.TTRef.FCz),
            "O2": self.Mapping(self.TTRef.O2, self.TTRef.FCz),
            "Cpz": self.Mapping(self.TTRef.Cpz, self.TTRef.FCz),
            "FPz": self.Mapping(self.TTRef.FPz, self.TTRef.FCz),
        }
        
        self.channel_types = {'analog': ['Fp1', 'AF3', 'AF7', 'Fz', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3', 'FC5', 'FT7', 'Cz', 'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3', 
        'CP5', 'TP7', 'TP9', 'Pz', 'P1', 'P3', 'P5', 'P7', 'PO3', 'PO7', 'Oz', 'O1', 'Fpz', 'Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4', 
        'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P2', 'P4', 'P6', 'P8', 'POz', 'PO4', 'PO8', 'O2', 'Cpz', 'FPz'], 
                            'digital': []}
        
        
        self.channel_groups = {'eeg_eog': ['Fp1', 'AF3', 'AF7', 'Fz', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3', 'FC5', 'FT7', 'Cz', 'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3', 
        'CP5', 'TP7', 'TP9', 'Pz', 'P1', 'P3', 'P5', 'P7', 'PO3', 'PO7', 'Oz', 'O1', 'Fpz', 'Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4', 
        'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P2', 'P4', 'P6', 'P8', 'POz', 'PO4', 'PO8', 'O2', 'Cpz', 'FPz'],}
                
        self.file_extensions = {'psg_ext': '**/*_eeg.vhdr',
                                'ann_ext': '**/*_eeg.vhdr'} # no annotations, but known that all epochs are Wake
    
    def dataset_paths(self):
        return ['', '']
    
    def ann_parse(self, ann_fname):

        file_info = self.get_file_info(logger=None, filepath=ann_fname)
        start_time = file_info["start_datetime"]
        duration = file_info["file_duration"]
        
        n_epochs = int(duration // 30)  # Assuming 30-second epochs

        ann_stage_events = [{"Stage": "Wake", "Start": idx * 30, "Duration": 30} for idx in range(n_epochs)]

        return ann_stage_events, start_time, None, None
    

