import os
import shutil
import moabb
from moabb.datasets import Rodrigues2017
from pathlib import Path

from datasets.base import BaseDataset
from datasets.registry import register_dataset

print(moabb.__version__)

@register_dataset("RODRIGUES2017")
class RODRIGUES2017(BaseDataset):
    """Rodrigues2017 dataset"""
    
    def __init__(self):
        super().__init__("RODRIGUES2017","MNE-BIDS-rodrigues2017", keep_folder_structure=False)
  
    def _setup_dataset_config(self):
        self.ann2label = {} # no annotations

        # https://hal.science/hal-02086581
        self.inter_dataset_mapping = {
            "Fp1": self.Mapping(self.TTRef.Fp1, self.TTRef.RPA),
            "Fp2": self.Mapping(self.TTRef.Fp2, self.TTRef.RPA),
            "Fc5": self.Mapping(self.TTRef.FC5, self.TTRef.RPA),
            "Fz": self.Mapping(self.TTRef.Fz, self.TTRef.RPA),
            "Fc6": self.Mapping(self.TTRef.FC6, self.TTRef.RPA),
            "T7": self.Mapping(self.TTRef.T7, self.TTRef.RPA),
            "Cz": self.Mapping(self.TTRef.Cz, self.TTRef.RPA),
            "T8": self.Mapping(self.TTRef.T8, self.TTRef.RPA),
            "P7": self.Mapping(self.TTRef.P7, self.TTRef.RPA),
            "P3": self.Mapping(self.TTRef.P3, self.TTRef.RPA),
            "Pz": self.Mapping(self.TTRef.Pz, self.TTRef.RPA),
            "P4": self.Mapping(self.TTRef.P4, self.TTRef.RPA),
            "P8": self.Mapping(self.TTRef.P8, self.TTRef.RPA),
            "O1": self.Mapping(self.TTRef.O1, self.TTRef.RPA),
            "Oz": self.Mapping(self.TTRef.Oz, self.TTRef.RPA),
            "O2": self.Mapping(self.TTRef.O2, self.TTRef.RPA),
        }
        

        self.channel_names = ['Fp1', 'Fp2', 'Fc5', 'Fz', 'Fc6', 'T7', 'Cz', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
        
        
        self.channel_types = {'analog': ['Fp1', 'Fp2', 'Fc5', 'Fz', 'Fc6', 'T7', 'Cz', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2'], 
                              'digital': []}
    
        
        
        self.channel_groups = {
            'eeg_eog':  ['Fp1', 'Fp2', 'Fc5', 'Fz', 'Fc6', 'T7', 'Cz', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2'],
        }        
        
        self.file_extensions = {'psg_ext': '**/*_eeg.edf',
                                'ann_ext': '**/*_stages.tsv'} # no annotations

    def dataset_paths(self):
        return [
            "",
            ""
        ]

    def preprocess(self, n_workers, data_dir, ann_dir):
        
        execute_preprocess = input(f"Do you want to download the Rodrigues2017 dataset now to your specified base directory ({data_dir})? (Y/N) ")
        
        if str(execute_preprocess).lower() == "y":

            dataset = Rodrigues2017()
            dataset.convert_to_bids(path=Path(data_dir).parent)

            # Clean-up mne cache
            mat_cache = os.path.join(os.path.expanduser("~"), "mne_data", "MNE-alphawaves-data")
            shutil.rmtree(mat_cache, ignore_errors=True)

            if str(input("Do you want to continue with processing now? (Y/N) ")).lower() == "n":
                return False
        return True
    
