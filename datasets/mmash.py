import os
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("MMASH")
class MMASH(BaseDataset):
    """MMASH - Multilevel Monitoring of Activity and Sleep in Healthy People dataset."""
    
    def __init__(self):
        super().__init__("MMASH","MMASH - Multilevel Monitoring of Activity and Sleep in Healthy People")

    unit_mapping = {
        'ibi_s': 's',
        'Axis1': 'Nm',
        'Axis2': 'Nm',
        'Axis3': 'Nm',
        'Steps': 'count',
        'HR': 'bpm',
        'Inclinometer Off': 'a.u.',
        'Inclinometer Standing': 'a.u.',
        'Inclinometer Sitting': 'a.u.',
        'Inclinometer Lying': 'a.u.',
        'Vector Magnitude': 'Nm'}


    def get_channels(self, logger, filepath):
        """available channels from file."""
        channels = []

        rr_file = filepath.replace('user_info', 'RR')
        if os.path.exists(rr_file):
            channels.append('ibi_s')

        actigraph_file = filepath.replace('user_info', 'Actigraph')
        if os.path.exists(actigraph_file):
            channels.extend(['Axis1', 'Axis2', 'Axis3', 'Steps', 'HR', 'Inclinometer Off', 'Inclinometer Standing', 
                             'Inclinometer Sitting', 'Inclinometer Lying', 'Vector Magnitude'])

        return channels

    def read_signal(self, logger, filepath, channel):
        """Read signal data for a specific channel."""
        file_info = self.get_file_info(logger, filepath)
        start_time = file_info["start_datetime"].replace(tzinfo=timezone.utc)
        duration = file_info["file_duration"]
        start_timestamp = int(start_time.timestamp())
        end_timestamp = start_timestamp + int(duration)

        if channel == 'ibi_s':
            file = filepath.replace('user_info', 'RR')
            df = pd.read_csv(file, header=0)
            df.loc[df["day"]!= 1, 'day'] = 2   # Adjust days to be either 1 or 2, because some files have values like -29 for second day
            df_seconds = df.groupby(['day','time'])['ibi_s'].mean().reset_index()
            times = (
                    pd.to_datetime(df_seconds['time'], format='%H:%M:%S', utc=True)   # vectorized parse
                    + pd.to_timedelta(df_seconds['day'] - 1, unit='D')       # add day offset
                ).map(pd.Timestamp.timestamp).astype(int).values
            signal = df_seconds['ibi_s'].values
            f = interp1d(times, signal, kind='nearest', fill_value='extrapolate')
            signal_interpl = f(np.arange(start_timestamp, end_timestamp, 1))
        else:
            file = filepath.replace('user_info', 'Actigraph')
            df = pd.read_csv(file, header=0)
            df.loc[df["day"]!= 1, 'day'] = 2   # Adjust days to be either 1 or 2, because some files have values like -29 for second day
            times = (
                    pd.to_datetime(df['time'], format='%H:%M:%S', utc=True)   # vectorized parse
                    + pd.to_timedelta(df['day'] - 1, unit='D')       # add day offset
                ).map(pd.Timestamp.timestamp).astype(int).values
            signal = df[channel].values
            f = interp1d(times, signal, kind='nearest', fill_value='extrapolate')
            signal_interpl = f(np.arange(start_timestamp, end_timestamp, 1))
        return signal_interpl
    
    def get_file_info(self, logger, filepath):
        """Get start datetime and file duration."""
        rr_file = filepath.replace('user_info', 'RR')
        rr_start_time, rr_end_time = None, None
        if os.path.exists(rr_file):
            rr = pd.read_csv(rr_file, header=0)
            rr_start_time = datetime.strptime(rr['time'][0], '%H:%M:%S')
            rr_end_time = datetime.strptime(rr['time'].iloc[-1], '%H:%M:%S')
            if rr['day'].iloc[-1] != rr['day'][0]:  # handle day rollover
                rr_end_time += pd.Timedelta(days=1)


        actigraph_file = filepath.replace('user_info', 'Actigraph')
        actigraph_start_time, actigraph_end_time = None, None
        if os.path.exists(actigraph_file):
            actigraph = pd.read_csv(actigraph_file)
            actigraph_start_time = datetime.strptime(actigraph['time'][0], '%H:%M:%S')
            actigraph_end_time = datetime.strptime(actigraph['time'].iloc[-1], '%H:%M:%S')
            if actigraph['day'].iloc[-1] != actigraph['day'][0]:  # handle day rollover
                actigraph_end_time += pd.Timedelta(days=1)

        if rr_start_time and actigraph_start_time:
            start_time = max(rr_start_time, actigraph_start_time)   # take the later start time to ensure both signals are covered
            end_time = min(rr_end_time, actigraph_end_time) # take the earlier end time to ensure both signals are covered
        else:
            start_time = rr_start_time or actigraph_start_time
            end_time = rr_end_time or actigraph_end_time

        if start_time is None:
            raise ValueError(f"No valid start time found in either RR or Actigraph files for {filepath}")
        

        file_duration = (end_time - start_time).total_seconds()

        return {"start_datetime": start_time,
                "file_duration": file_duration}
    
    def get_signal_data(self, logger, filepath, channel):
        """Get complete signal information for specific channel."""
        # Get start time to align signals from different files
        file_info = self.get_file_info(logger, filepath)
        start_time = file_info["start_datetime"].replace(tzinfo=timezone.utc)
        duration = file_info["file_duration"]
        start_timestamp = int(start_time.timestamp())
        end_timestamp = start_timestamp + int(duration)

        if channel == 'ibi_s':
            file = filepath.replace('user_info', 'RR')
            df = pd.read_csv(file, header=0)

            # Adjust days to be either 1 or 2, because some files have values like -29 for second day
            df.loc[df["day"]!= 1, 'day'] = 2   
            # Some rows have the same timestamp because there is one row per heart beat, so ibi_s values are averaged per second
            df_seconds = df.groupby(['day','time'])['ibi_s'].mean().reset_index()
            
            # Convert 'time' and 'day' to timestamps
            times = (
                    pd.to_datetime(df_seconds['time'], format='%H:%M:%S', utc=True)
                    + pd.to_timedelta(df_seconds['day'] - 1, unit='D')       # add day offset
                ).map(pd.Timestamp.timestamp).astype(int).values
            signal = df_seconds['ibi_s'].values
            # Interpolate signal to reference timestamps (1Hz) to align with other signals
            f = interp1d(times, signal, kind='nearest', fill_value='extrapolate')
            signal_interpl = f(np.arange(start_timestamp, end_timestamp, 1))
            unit = 's'
        else:
            file = filepath.replace('user_info', 'Actigraph')
            df = pd.read_csv(file, header=0)
            df.loc[df["day"]!= 1, 'day'] = 2   
            times = (
                    pd.to_datetime(df['time'], format='%H:%M:%S', utc=True)
                    + pd.to_timedelta(df['day'] - 1, unit='D')       # add day offset
                ).map(pd.Timestamp.timestamp).astype(int).values
            signal = df[channel].values
            f = interp1d(times, signal, kind='nearest', fill_value='extrapolate')
            signal_interpl = f(np.arange(start_timestamp, end_timestamp, 1))
            unit = self.unit_mapping.get(channel, 'a.u.')

        return {
            "signal": signal_interpl,
            "sampling_rate": 1,
            "unit": unit
        }


    def _setup_dataset_config(self):
        self.ann2label = {"W": "W",
                          "Sleep": "SLEEP"}

        
        self.channel_names =  ['Axis1', 'Axis2', 'Axis3', 'Steps', 'HR', 'Inclinometer Off', 'Inclinometer Standing', 
                             'Inclinometer Sitting', 'Inclinometer Lying', 'Vector Magnitude','ibi_s']
        
        
        self.channel_types = {'analog': ['ibi_s', 'Axis1', 'Axis2', 'Axis3', 'HR', 'Vector Magnitude'], 
                            'digital': ['Steps', 'Inclinometer Off', 'Inclinometer Standing', 'Inclinometer Sitting', 'Inclinometer Lying']}
        
        
        self.channel_groups = {}
        
        
        self.file_extensions = {
                                'psg_ext': 'DataPaper/**/user_info.csv',    # used as common reference per user (channels are derived from RR.csv and Actigraph.csv)
                                'ann_ext': 'DataPaper/**/sleep.csv' # contains only sleep statistics, no annotations
                                }
        

    def dataset_paths(self):
        return ['', '']
    
    def ann_parse(self, ann_fname):
        """ MMASH does not have sleep stage annotations, but sleep statistics (TIB, WASO, Sleep Onset, etc.)
        This function will create annotations based on these statistics, assigning Wake (W) for non-sleep periods
        and Unknown (UNK) for sleep periods, as there are no specific sleep stages like N1, N2, etc.
        """
        # Get start and end time from recording
        user_info_file = ann_fname.replace('sleep.csv', 'user_info.csv')
        file_info = self.get_file_info(None, user_info_file)
        start_time = file_info["start_datetime"]
        duration = file_info["file_duration"]
        end_time = start_time + pd.Timedelta(seconds=duration)

        sleep_df = pd.read_csv(ann_fname, sep=',', header=0)
        if len(sleep_df) == 0:
            return [], start_time, None, None  # No sleep data available (e.g. user 11)
        
        lights_off = datetime.strptime(sleep_df['In Bed Time'][0], '%H:%M') + pd.Timedelta(days=0 if sleep_df['In Bed Date'][0] == 1 else 1) 
        lights_on = datetime.strptime(sleep_df['Out Bed Time'].iloc[-1], '%H:%M') + pd.Timedelta(days=0 if sleep_df['Out Bed Date'][0] == 1 else 1)

        ann_stage_events = []
        # annotate with wake from recording start till recording stop
        n_epochs_pre_sleep = int((end_time - start_time).total_seconds() // 30)
        ann_stage_events.extend([{'Stage': 'W', 'Start': idx*30, 'Duration': 30} for idx in range(0,n_epochs_pre_sleep)])
        
        # annotate sleep phases with label 'Sleep'
        for _, row in sleep_df.iterrows():
            phase_start = datetime.strptime(row['Onset Time'], '%H:%M') + pd.Timedelta(days=0 if row['Onset Date'] == 1 else 1)
            phase_end = phase_start + pd.Timedelta(minutes=row['Total Minutes in Bed'])
            start_epoch = int((phase_start - start_time).total_seconds() // 30)
            end_epoch = int((phase_end - start_time).total_seconds() // 30)
            ann_stage_events[start_epoch:end_epoch] = [{'Stage': 'Sleep', 'Start': idx*30, 'Duration': 30} for idx in range(start_epoch, end_epoch)]

        return ann_stage_events, start_time, lights_off, lights_on
    
    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(labels) > len(signals):
            # happens at least once (user_9) where time out of bed is after the end of recording
            return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)


    def preprocess(self, n_workers, data_dir, ann_dir):
        print("\n MMASH files do not have unique filenames ... \n")
        
        execute_preprocess = input("Do you want to perform the processing on copied and renamed files (if not, the results will be stored in user_id subfolders)? (Y/N) ")
        
        if str(execute_preprocess).lower() == "y":
            organizer = MMASHRenaming(data_dir)
            organizer.rename_files(n_workers)
            self.file_extensions = {
                                'psg_ext': 'DataRenamed/*_user_info.csv',
                                'ann_ext': 'DataRenamed/*_sleep.csv'
                                }
            self.keep_folder_structure = False
        else:
            self.keep_folder_structure = True
        
        return True
            

class MMASHRenaming:
    """Handles renaming of MMASH dataset files."""

    def __init__(self, base_dir: str):
        """
        Initialize the organizer.

        Args:
            base_dir: Path to MMASH dataset root directory
        """
        self.base_dir = Path(base_dir)

    def rename_files(self, n_workers):
        """Copy and rename all files by adding user_id
        """

        # Process each user
        data_dir = Path(self.base_dir) / 'DataPaper'
        user_folders = [str(f) for f in data_dir.iterdir() if f.is_dir()]

        # Create new directory for renamed files
        renamed_dir = Path(self.base_dir) / 'DataRenamed'
        renamed_dir.mkdir(exist_ok=True)

        for user in tqdm(user_folders, desc="Processing users"):

            files = glob.glob(user+"/*.csv")

            try:
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = [executor.submit(self._copy_file, renamed_dir, file, user) for file in files]
                    for future in as_completed(futures):
                        future.result()  
            except KeyboardInterrupt:
                print("Preprocessing interrupted by user. Shutting down...")
                executor.shutdown(cancel_futures=True)            
                
        print('Resorting done, if you want you can delete the duplicates at their original location.')

    def _copy_file(self, renamed_dir, file, user):
        new_filename = os.path.basename(user)+'_'+str(os.path.basename(file))

        if not os.path.exists(renamed_dir / new_filename):  
            # print(f"Copy file {file} to {subgroup_folder+'/'+new_filename}")
            shutil.copy2(file, renamed_dir / new_filename) 