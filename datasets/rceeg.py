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
        
        self.inter_dataset_mapping = {
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
    

