import os
from pathlib import Path
import pandas as pd
import glob

from datasets.base import BaseDataset
from datasets.registry import register_dataset
from datasets.file_handlers import BRAINVISIONHandler

@register_dataset("SEF")
class SEF(BaseDataset):
    """SEF - Simultaneous EEG and fMRI signals during sleep from humans (ds003768) dataset"""

    def __init__(self):
        super().__init__("SEF","SEF - Simultaneous EEG and fMRI signals during sleep from humans (ds003768)", keep_folder_structure=False)

        self._file_handler = BRAINVISIONHandler()
    
    def _setup_dataset_config(self):
        self.ann2label = {"W": "W",
                          "W (uncertain)": "W",
                          "1": "N1",
                          "1 (uncertain)": "N1",
                          "2": "N2",
                          "2 (uncertain)": "N2",
                          "3": "N3",
                          "3 (uncertain)": "N3",
                          "Unscorable": "UNK",
                          "unscorable": "UNK",
                          "1 (unscorable)": "UNK",
                          "2 (unscorable)": "UNK",
                          "3 (unscorable)": "UNK",
                          "2 or 3 (unscorable)": "UNK",
                          }
        
        self.channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 
                              'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'EOG', 'ECG']
        
        # see https://doi.org/10.1016/j.neuroimage.2022.119720
        self.inter_dataset_mapping = {
            'Fp1': self.Mapping(self.TTRef.Fp1, self.TTRef.FCz),
            'Fp2': self.Mapping(self.TTRef.Fp2, self.TTRef.FCz),
            'F3': self.Mapping(self.TTRef.F3, self.TTRef.FCz),
            'F4': self.Mapping(self.TTRef.F4, self.TTRef.FCz),
            'C3': self.Mapping(self.TTRef.C3, self.TTRef.FCz),
            'C4': self.Mapping(self.TTRef.C4, self.TTRef.FCz),
            'P3': self.Mapping(self.TTRef.P3, self.TTRef.FCz),
            'P4': self.Mapping(self.TTRef.P4, self.TTRef.FCz),
            'O1': self.Mapping(self.TTRef.O1, self.TTRef.FCz),
            'O2': self.Mapping(self.TTRef.O2, self.TTRef.FCz),
            'F7': self.Mapping(self.TTRef.F7, self.TTRef.FCz),
            'F8': self.Mapping(self.TTRef.F8, self.TTRef.FCz),
            'T7': self.Mapping(self.TTRef.T7, self.TTRef.FCz),
            'T8': self.Mapping(self.TTRef.T8, self.TTRef.FCz),
            'P7': self.Mapping(self.TTRef.P7, self.TTRef.FCz),
            'P8': self.Mapping(self.TTRef.P8, self.TTRef.FCz),
            'Fz': self.Mapping(self.TTRef.Fz, self.TTRef.FCz),
            'Cz': self.Mapping(self.TTRef.Cz, self.TTRef.FCz),
            'Pz': self.Mapping(self.TTRef.Pz, self.TTRef.FCz),
            'Oz': self.Mapping(self.TTRef.Oz, self.TTRef.FCz),
            'FC1': self.Mapping(self.TTRef.FC1, self.TTRef.FCz),
            'FC2': self.Mapping(self.TTRef.FC2, self.TTRef.FCz),
            'CP1': self.Mapping(self.TTRef.CP1, self.TTRef.FCz),
            'CP2': self.Mapping(self.TTRef.CP2, self.TTRef.FCz),
            'FC5': self.Mapping(self.TTRef.FC5, self.TTRef.FCz),
            'FC6': self.Mapping(self.TTRef.FC6, self.TTRef.FCz),
            'CP5': self.Mapping(self.TTRef.CP5, self.TTRef.FCz),
            'CP6': self.Mapping(self.TTRef.CP6, self.TTRef.FCz),
            'TP9': self.Mapping(self.TTRef.TP9, self.TTRef.FCz),
            'TP10': self.Mapping(self.TTRef.TP10, self.TTRef.FCz),
            'EOG': self.Mapping(self.TTRef.EL, self.TTRef.FCz), # not sure
            'ECG': self.Mapping(self.TTRef.ECG, None),
        }
        
        self.channel_types = {'analog': ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 
                                         'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'EOG', 'ECG'], 
                            'digital': []}
        
        
        self.channel_groups = {'eeg_eog': ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'EOG'],
                                'ecg': ['ECG']}
                
        self.file_extensions = {'psg_ext': '**/*_eeg.vhdr',
                                'ann_ext': '**/*_stages.tsv'} # no annotations
    
    def dataset_paths(self):
        return ['', '']
    
    def ann_parse(self, ann_fname):
        ann_df = pd.read_csv(ann_fname, sep='\t', header=0)

        ann_stage_events = []
        start_time = None
        epoch_duration = 30

        for i, row in ann_df.iterrows():
            try:
                start = int(row['epoch_start_time_sec'])
                stage = str(row['30-sec_epoch_sleep_stage'])
            except (ValueError, TypeError):
                # File sub-32_task-sleep_run-1_eeg.tsv has wrong column split (space instead of tab)
                start, stage = row['epoch_start_time_sec'].split(" ")
                start = int(start)

            if start_time is None:
                start_time = start

            ann_stage_events.append({'Stage': stage,
                                        'Start': start - start_time,
                                        'Duration': epoch_duration})

        return ann_stage_events, start_time, None, None
    
    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(labels) > len(signals):
            return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)

        if len(signals) > len(labels):
            return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)
        
    def preprocess(self, n_workers, data_dir, ann_dir):
        print("\n SEF annotation files originally are stored in an unsupported way and therefor need to be preprocessed/splitted ... \n \
              This will not modify the original file content")
        
        execute_preprocess = input("Do you want to perform the preprocessing now? (Y/N) ")
        
        if str(execute_preprocess).lower() == "y":
            organizer = SEFFileOrganizer(data_dir, self.file_extensions)
            organizer.organize_files()

            if str(input("Do you want to continue with processing now? (Y/N) ")).lower() == "n":
                return False
        return True
    
class SEFFileOrganizer:
    def __init__(self, base_dir, file_extensions):
        self.base_dir = base_dir
        self.file_extensions = file_extensions

    def organize_files(self):
        # annot_files = glob.glob(os.path.join(self.base_dir, 'sourcedata/*_sleep-stage.tsv'))
        print(os.path.join(self.base_dir, self.file_extensions['psg_ext']))
        psg_files = glob.glob(os.path.join(self.base_dir, self.file_extensions['psg_ext']), recursive=True)

        print(f"Found {len(psg_files)} PSG files to organize.")

        for psg_file in psg_files:
            if "sub-16_task-rest_run-1_eeg.vhdr" in psg_file or "sub-24_task-rest_run-2_eeg.vhdr" in psg_file:
                print(f"File {os.path.basename(psg_file)} is known to have a wrong filename stored inside. It will be modified to match the .eeg file.")
                with open(psg_file, "r", encoding="utf-8") as f:
                    content = f.read()

                old_value = "File=rsub-"
                new_value = "File=sub-"

                if old_value in content:
                    content = content.replace(old_value, new_value)
                    with open(psg_file, "w", encoding="utf-8") as f:
                        f.write(content)
                    print("Fixed: DataFile entry updated successfully.")

            psg_parts = Path(psg_file).stem.split("_")
            psg_id = "_".join(psg_parts[:3])
            task_id = "_".join(psg_parts[1:3])
            annot_file = os.path.join(self.base_dir, "sourcedata", Path(psg_file).stem.split("_task")[0] + "-sleep-stage.tsv")
            target_annot_path = os.path.join(Path(psg_file).parent, f"{psg_id}_stages.tsv")

            if os.path.exists(annot_file):
                all_annot_subj = pd.read_csv(annot_file, sep='\t', header=0)
                annot_task = all_annot_subj[all_annot_subj['session'] == task_id]
                annot_task.to_csv(target_annot_path, sep='\t', index=False)
        print("Finished organizing files.")



