"""
Main dataset processor for PSG data preparation.
"""

import logging
import os
import re
import copy
from math import ceil, floor
from datetime import datetime, date, time, timedelta
from pyedflib import EdfWriter
import h5py
import glob
from pathlib import Path
from decimal import Decimal
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from ..utils import LoggingManager
from .dataset_explorer import Dataset_Explorer
from .signal_processor import SignalProcessor

logging.captureWarnings(True)

# Final sleep stage labels mapping (did not yet find a better place for this, maybe in config?)
# labels will appear like this in output 
STAGE_DICT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "MOVE": 5, "UNK": 6}

SLEEP_STAGES = [STAGE_DICT["N1"],STAGE_DICT["N2"],STAGE_DICT["N3"],STAGE_DICT["REM"]]


class DatasetProcessor:
    """
    Main processor for PSG dataset preparation and signal processing.

    This class orchestrates the entire dataset processing pipeline,
    handling file processing, signal cleaning, and output generation.
    """

    def __init__(self, dataset, config):

        self.logging_manager = LoggingManager(console_level=config.logging_level)
        self.dataset = dataset
        self.config = config

    def process_files(self):

        try:
            if self.config.overwrite:
                log_files = glob.glob(os.path.join(self.config.output_dir, "**", "*.log"), recursive=True)
                for f in log_files:
                    os.remove(f)

            self.pipeline_logger = self.logging_manager.create_pipeline_logger()

            self.pipeline_logger.info("Starting dataset processing")

            # Get all PSG and Annot files names
            psg_fnames, ann_fnames  = Dataset_Explorer(
                self.pipeline_logger,
                self.dataset,
                self.config.psg_dir,
                self.config.ann_dir,
                log_level=self.config.logging_level,
            ).get_files()

            if self.config.use_annot:
                annotation_map = self._build_annot_lookup(psg_fnames, ann_fnames)
                if len(annotation_map) < len(psg_fnames):
                    self.pipeline_logger.info(f"{len(psg_fnames)-len(annotation_map)}/{len(psg_fnames)} PSG files have no matching annotation file and will be skipped.")
                n_files_to_process = len(annotation_map)
            else:
                n_files_to_process = len(psg_fnames)

            # Process psg files in parallel
            max_workers = self.config.num_workers or os.cpu_count() or 1
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                tasks = []
                psg_iter = iter(psg_fnames)

                # Progress Bar
                with tqdm(total=n_files_to_process, desc="Processing files") as pbar:
                    more_files_to_process = True
                    while True:
                        if not more_files_to_process and not tasks:
                           break
                        
                        # Fill up worker tasks until max_workers is reached or no more files to process
                        # Do not enqueue all files at once to handle Errors and stop processing immediatly
                        while len(tasks) < max_workers:
                            try:
                                psg_fname = next(psg_iter)
                            except StopIteration:
                                more_files_to_process = False
                                break

                            # Find matching annotation file
                            if self.config.use_annot:
                                ann_fname = annotation_map.get(psg_fname, None)
                                if ann_fname is None:
                                    continue # No matching annotation file found, skip this PSG file
                            else:
                                ann_fname = None

                            file_processor = FileProcessor(self.config, self.dataset, psg_fname, ann_fname)

                            future = executor.submit(file_processor._process_file)
                            tasks.append(future)
                            
                        # Get notified when at least one file is finished and resume this one
                        # Filling with new file will happen in the next loop iteration -> handle errors before enqueuing new files
                        # Maximal "n_worker" files will still get finished before stopping pipeline if an error occurs
                        # Ensures that capacity of workers is used as good as possible, even if some files take much longer to process than others
                        done,_ = wait(tasks, return_when=FIRST_COMPLETED)   

                        for future in done:
                            try:
                                future.result()  # Check for exceptions
                                pbar.update(1)
                            except Exception as e:
                                # self.pipeline_logger.error(f"Processing failed: {e}")
                                executor.shutdown(cancel_futures=True)
                                break
                            finally:
                                tasks.remove(future)
                        else:
                            continue
                        break  # Break outer loop if an exception occurred in the inner loop
                    
            # Finalize processing
            self.pipeline_logger.info("=" * 60)
            self.pipeline_logger.info("DATASET PROCESSING COMPLETED")
        except KeyboardInterrupt:
            for task in tasks:
                task.cancel()
            executor.shutdown(cancel_futures=True)
            self.pipeline_logger.info("=" * 60)
            self.pipeline_logger.info("Stopped processing")

    def _build_annot_lookup(self, psg_fnames, ann_fnames):
        annotation_map = {}

        # Build lookup dict by identifier
        ann_lookup = {}

        for ann_fname in ann_fnames:
            ann_base = str(Path(ann_fname).relative_to(self.config.ann_dir))
            ann_id = self.dataset.get_file_identifier(None, ann_base)[1]
            ann_lookup[ann_id] = ann_fname

        for psg_fname in psg_fnames:
            psg_base = str(Path(psg_fname).relative_to(self.config.psg_dir))
            psg_id = self.dataset.get_file_identifier(psg_base, None)[0]
            match_ann_id = ann_lookup.get(psg_id, None)
            if match_ann_id:
                annotation_map[psg_fname] = match_ann_id
            # else:
            #     self.pipeline_logger.warning(
            #         f"No matching annotation file found for PSG: "
            #         f"{Path(psg_fname).relative_to(self.config.psg_dir)}. Skipping file."
            #     )
        return annotation_map


class FileProcessor:
    """Class to process a single PSG file with one or multiple channels and save the processed data to file."""

    def __init__(self, config, dataset, psg_fname, ann_fname):
        self.logging_manager = LoggingManager(console_level=config.logging_level)
        self.config = config
        self.dataset = dataset
        self.psg_fname = psg_fname
        self.ann_fname = ann_fname

    def _process_file(self, ):
        """Process a single PSG file for all specified channels."""
     
        try:
            # Start buffering logs for this file
            file_id = Path(self.psg_fname).stem
            file_logger, buffer_handler = self.logging_manager.create_file_logger(file_identifier=file_id)

            file_logger.info(f"Signal file: {Path(self.psg_fname).relative_to(self.config.psg_dir)}")
            if self.config.use_annot:
                file_logger.info(f"Annotation file: {Path(self.ann_fname).relative_to(self.config.ann_dir)}")

            # Initialize signal data dictionary which holds all necessary info for processing and saving
            file_data = {
                "psg_fname": self.psg_fname,
                "ann_fname": self.ann_fname,
            }

            if self.config.output_format in ["hdf5", "edf"]:
                file_output_path, log_path = self._setup_file_output()
                # Skip if file already exists and overwrite is False
                if os.path.exists(file_output_path) and not self.config.overwrite:
                    file_logger.info(f"File already exists: {file_output_path}, skipping file.")
                    return
            else:
                log_paths = {}  # Store log paths for each channel separately

            # Get Start datetime of polysomnography data
            start_datetime = self.dataset.get_start_datetime(file_logger, self.psg_fname)
            file_data["start_datetime"] = start_datetime
            file_logger.info(f"Start datetime: {start_datetime}")

            if self.config.use_annot:
                # Parse annotations (is same for all channels)
                ann_stage_events, ann_Startdatetime, lights_off, lights_on = self.dataset.ann_parse(self.ann_fname)
                file_data["ann_start_datetime"] = ann_Startdatetime
                file_data.update({
                    "lights_off": lights_off,
                    "lights_on": lights_on
                })

                if ann_stage_events == []:
                    file_logger.warning(f"No sleep stage annotations found in {os.path.basename(self.ann_fname)}, skipping file.")
                    return
                
                # Map dataset-labels to standardized labels and check consistency
                labels = self.dataset.ann_label(file_logger, ann_stage_events, STAGE_DICT, self.config.epoch_duration)

                # Check how many sleep epochs are in the file
                sleep_mask = np.isin(labels, SLEEP_STAGES)
                sleep_idx = np.where(sleep_mask)[0]

                if len(sleep_idx) < self.config.min_sleep_epochs:
                    file_logger.warning("File contains less sleep epochs than required, skipping file.")
                    return
                
                file_data["labels"] = labels
 
                # Check if annotations and signal start at the same timestamp and pad/crop if necessary and configured
                file_data["start_delay"] = self._get_start_delay(file_logger, ann_Startdatetime, start_datetime)

            else:
                file_data.update({
                    "labels": None,
                    "ann_start_datetime": None,
                    "lights_off": None,
                    "lights_on": None,
                    "start_delay": 0,
                })

            if self.config.select_epochs == "lights":
                # Some datasets have lights marker not embedded in annotation but in a separate file or as a psg channel
                lights_off, lights_on = self.dataset.get_light_times(file_logger, self.psg_fname)
                if lights_off is not None:
                    file_data["lights_off"] = lights_off
                if lights_on is not None:
                    file_data["lights_on"] = lights_on     

                # Calculate the epochs to select later between lights off and lights on
                file_data["lights_off"], file_data["lights_on"] = self._get_lights_epochs(file_logger, file_data)    

            # Intersection of available channels in psg file and configured channels to process
            channels = list(
                set(self.config.channels) & set(self.dataset.get_channels(file_logger, self.psg_fname))
            )
            if len(channels)==0:
                file_logger.warning(f"No selected channels found in this file. Skipping.")
                return 

            # Process each channel, then save after all channels are processed
            all_channel_data = {}
            for channel in sorted(channels):
                # Set current channel for logging
                buffer_handler.set_channel(channel)
                channel_harm = self._harmonize_channel_name(file_logger, channel)
                if self.config.output_format == "npz":
                    file_output_path, log_path = self._setup_channel_output(channel_harm)
                    log_paths[channel] = log_path

                # all_channel_data["file_output_path"] = file_output_path
                # if os.path.exists(file_output_path):
                #     raise FileExistsError(f"Output file already exists: {file_output_path}.")
                # np.savetxt(file_output_path, [])
                # continue

                # Skip if file already exists and overwrite is False (only for output format npz on this level)
                if os.path.exists(file_output_path) and not self.config.overwrite:
                    file_logger.info(f"File already exists: {file_output_path}, skipping channel {channel}.")
                    continue 

                channel_processor = ChannelProcessor(file_logger, self.config, self.dataset, channel)
                proc_channel_data = channel_processor._process_channel(copy.deepcopy(file_data))

                if proc_channel_data is not None:
                    # Store all relevant info for saving
                    all_channel_data[channel_harm] = {
                        **proc_channel_data,
                        "ch_name": channel_harm,
                        "file_output_path": file_output_path,
                    }
                file_logger.info("=" * 40)

            # Save all channel data (output format handled inside _save_processed_data)
            if len(all_channel_data) > 0:
                self._save_processed_data(all_channel_data)
                file_logger.info(f"Successfully processed: {Path(self.psg_fname).name}")
        
        except Exception as e:
            # Log the exception
            file_logger.error(f"Error processing file {Path(self.psg_fname).name}: {str(e)}", exc_info=True)
            raise  # Re-raise after logging        
        finally:

            # Flush buffered logs to console and file(s), even if an exception occurred
            if self.config.output_format == "npz":
                if log_paths == {}:
                    # Error happened before log paths could be set up, flush to console for this case
                    buffer_handler.flush_to_console()
                # For npz, flush to each channel's log file with channel filtering
                for channel, log_path in log_paths.items():
                    buffer_handler.flush_to_console_and_file(log_path, channel=channel)
            elif self.config.output_format in ["edf", "hdf5"]:
                # For edf/hdf5, flush all channels' logs to single file
                buffer_handler.flush_to_console_and_file(log_path)

    def _harmonize_channel_name(self, logger, channel):
        """Harmonize channel name based on dataset-specific mapping."""
        if self.config.map_channel_names:
            channel_harm = self.dataset.map_channel(channel)
            logger.info(f"Mapped channel name {channel} to {channel_harm}")
            return channel_harm
        else:
            return channel
        
    def _setup_file_output(self):
        """ Setup output for edf and hdf5 output format where all channels of one file are stored together and share the same log file. """
        # Create output directory (including subfolders of original data structure if keep_folder_structure is True)
        if self.dataset.keep_folder_structure:
            relative_path = os.path.split(Path(self.psg_fname).relative_to(self.config.psg_dir))[0]
        else:
            relative_path = ""

        # Output is generated per PSG file containing all channels, log files are saved per PSG file separately
        output_dir = os.path.join(self.config.output_dir, relative_path)
        log_dir = os.path.join(output_dir, "log_files")
        log_filename = f"{Path(self.psg_fname).stem}.log"

        # Generate output file name
        filename = f"{Path(self.psg_fname).stem}.{self.config.output_format}"
        file_output_path = os.path.join(output_dir, filename)

        # Prepare log file path (will be written after each file is processed)
        log_file_path = os.path.join(log_dir, log_filename)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        return file_output_path, log_file_path

    def _setup_channel_output(self, channel):
        """Setup output for npz output format where each channel is stored separately and has its own log file."""

        # Create output directory (including subfolders of original data structure if keep_folder_structure is True)
        if self.dataset.keep_folder_structure:
            relative_path = os.path.split(Path(self.psg_fname).relative_to(self.config.psg_dir))[0]
        else:
            relative_path = ""

        # Output is generated per channel and sorted into channel folders with corresponding log file
        # replace slash in folder names to avoid nested output structure and colon because it is often not accepted in folder names
        channel_clean = re.sub(r"[:/]", "_", channel)
        output_dir = os.path.join(self.config.output_dir, relative_path, channel_clean)
        log_dir = output_dir
        log_filename = channel_clean + ".log"

        # Generate output file name
        filename = f"{Path(self.psg_fname).stem}.{self.config.output_format}"
        file_output_path = os.path.join(output_dir, filename)

        # Prepare log file path (will be written after each file is processed)
        log_file_path = os.path.join(log_dir, log_filename)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        return file_output_path, log_file_path

    def _get_start_delay(self, logger, ann_start_datetime, signal_start_datetime):
        """ Check if annotation and signal start datetime are aligned and calculate the delay between annotation and signal start if necessary.
        """
        if ann_start_datetime != None:
            delay = 0

            # If annotation start datetime is given as a datetime object, compare with signal start datetime and calculate start delay
            if isinstance(ann_start_datetime, datetime) and signal_start_datetime is not None:
                # Strip timezone info before comparison/subtraction
                ann_dt = ann_start_datetime.replace(tzinfo=None)
                sig_dt = signal_start_datetime.replace(tzinfo=None)
                if ann_dt.time() != sig_dt.time():
                    delay = (ann_dt - sig_dt).total_seconds()
                    
            # If annotation start datetime is given as a numeric value, it indicates a delay in seconds or samples (depending on dataset)
            elif isinstance(ann_start_datetime, (int, float, Decimal)):
                delay = ann_start_datetime
            else:
                raise Exception(f"Unsupported format of annotation start datetime: {ann_start_datetime}")
            if delay != 0:
                logger.info(f"Start of signal: {signal_start_datetime}, Start of labels: {ann_start_datetime}")

        return delay

    def _get_lights_epochs(self, logger, channel_data):
        """Get the epochs where lights off and lights on happen based on the configured lights marker in annotation or PSG data.
        - The epoch in which lights off event happens is return as lights_off_epoch
        - The epoch AFTER which lights on event happens is returned as lights_on_epoch, 
            meaning that the epoch in which lights on event happens is still included in the selected data. 

        Args:
            logger (_type_): logger
            channel_data (_type_): Dictionary containing minimum the following keys: "psg_fname", "start_datetime", "lights_off", "lights_on"

        Returns:
            _type_: epochs of lights off and lights on
        """
        psg_fname = os.path.basename(channel_data["psg_fname"])
        
        startdatetime = channel_data["start_datetime"]
        lights_off_epoch = 0
        lights_on_epoch = None

        if channel_data["lights_off"] is not None:
            lights_off = channel_data["lights_off"]
            if isinstance(lights_off, (datetime,time)):
                if isinstance(lights_off, time):
                    # if date of marker timestamp is not available, take date of signal start for easier comparison
                    lights_off = datetime.combine(startdatetime.date(),lights_off)

                lights_off_sec = (lights_off - startdatetime).total_seconds()

                if lights_off_sec != 0:
                    if lights_off_sec < 0 and lights_off_sec > -3600:       
                        # Between -3600 and 0, before signal start but in 1 hour range
                        logger.info(f"Lights Off time {lights_off} is before signal start time {startdatetime.time()}. No epoch selection applied.")
                        # maybe padding with wake epochs until lights Off time is reached? For now, just keep all epochs and do not select (keep lights_off_epoch at 0)
                    else:
                        if lights_off_sec < 0:
                            # Assume lights Off is after and signal start before midnight
                            lights_off_sec += 24*3600 # add 24h
                        
                        # Round to full epoch
                        lights_off_epoch = self._round_marker_time(logger, "lights_off", lights_off_sec, self.config.epoch_duration, lights_off.time())
                        
                        logger.info(f"Select only epochs after lights Off at {(startdatetime + timedelta(seconds=lights_off_sec)).time()} (epoch {lights_off_epoch})")
                 
                else:
                    logger.info("Lights Off time is at the start of the signal, no need of epoch selection based on lights Off time.")

            elif isinstance(lights_off, (int,float)):
                lights_off_sec = lights_off
                if lights_off_sec != 0:
                    if lights_off_sec < 0 and lights_off_sec > -3600:   # Lights Off probably starts before PSG Data (1 hour range before signal start)
                        logger.warning(f"{psg_fname}: Lights Off time {lights_off_sec} is before signal start time {startdatetime.time()}. No epoch selection applied.")
                        # front_padding = 
                        # maybe padding with wake epochs until lights Off time is reached? For now, just keep all epochs and do not select
                    elif lights_off_sec < 0:
                        raise Exception(f"{psg_fname}: Lights Off time ({lights_off_sec}) is more than 1 hour before signal start ({startdatetime.time()})")
                    else:
                        # Round to full epoch
                        lights_off_epoch = self._round_marker_time(logger, "lights_off", lights_off_sec, self.config.epoch_duration)
                        logger.info(f"Select only epochs after lights Off at second {lights_off_sec} (epoch {lights_off_epoch})")
                else:
                    logger.info("Lights Off time is at the start of the signal, no need of epoch selection based on lights Off time.")
            else:
                raise Exception(f"{psg_fname}: Lights Off time has unsupported format: {lights_off}.")
        
        else:
            logger.warning(f"{psg_fname}: Lights Off time not available, keeping all wake epochs at start.")
            
        if channel_data["lights_on"] is not None:
            lights_on = channel_data["lights_on"]
            # Info: Lights On marker can happen to be after signal end, cannot check this here because signal length is not determined yet
            if isinstance(lights_on, (datetime,time)):
                if isinstance(lights_on, time):
                    # if date of marker timestamp is not available, take date of signal start for easier comparison
                    lights_on = datetime.combine(startdatetime.date(),lights_on)
                
                # seconds from recording start until lights On
                lights_on_sec = (lights_on - startdatetime).total_seconds()

                if lights_on_sec < 0:
                    # Assume lights On is after and signal start before midnight
                    lights_on_sec += 24 * 3600  # add 24h

                lights_on_epoch = self._round_marker_time(logger, "lights_on", lights_on_sec, self.config.epoch_duration, lights_on.time())
                logger.info(f"Select only epochs before lights On at {(startdatetime + timedelta(seconds=lights_on_sec)).time()} (epoch {lights_on_epoch})")

            elif isinstance(lights_on, (int,float)):
                # seconds from recording start until lights On
                lights_on_sec = lights_on

                if lights_on_sec < 0:
                    raise Exception(f"{psg_fname}: Lights On time ({lights_on_sec}) is before signal start ({startdatetime.time()})")
                        
                lights_on_epoch = self._round_marker_time(logger, "lights_on", lights_on_sec, self.config.epoch_duration)
                logger.info(f"Select only epochs before lights On at second {lights_on_sec} (epoch {lights_on_epoch})")
            else:
                raise Exception(f"{psg_fname}: Lights On time has unsupported format: {lights_on}.")
        else:
            logger.warning(f"{psg_fname}: Lights On time not available.")

        return lights_off_epoch, lights_on_epoch
   
    def _round_marker_time(self, logger, marker, marker_sec, epoch_duration, marker_time=None):
        """Helper function to round lights Off/On time to the next (or previous) epoch if it is not exactly at the end of an epoch and log this behavior.
        
        Args:
            marker (_type_): Type of marker which is processed ("lights_off" or "lights_on")
            marker_sec (_type_): timestamp of marker in seconds from signal start
            marker_time (_type_): timestamp of marker as datetime.time object for logging purposes (if not available, )
            epoch_duration (_type_): epoch duration in seconds to round the marker time to

        Returns:
            _type_: epoch number corresponding to the rounded marker time
        """
        if marker_time is None:
            marker_time = marker_sec
        if marker_sec % epoch_duration != 0:              
            if marker == "lights_off":
                round_marker_epoch = floor(marker_sec / epoch_duration)
                logger.info(f"Lights Off time {marker_time} is not exactly at the start of an epoch. Keep data from second{round_marker_epoch * epoch_duration} (epoch {round_marker_epoch}) on to avoid cutting epochs.")
            elif marker == "lights_on":
                round_marker_epoch = ceil(marker_sec / epoch_duration)
                logger.info(f"Lights On time {marker_time} is not exactly at the end of an epoch. Keep data until second {round_marker_epoch * epoch_duration} (epoch {round_marker_epoch}) to avoid cutting epochs.")
        else:
            round_marker_epoch = marker_sec / epoch_duration
        return int(round_marker_epoch)
 
    def _save_processed_data(self, all_channel_data):
        """Save processed data to file for npz, edf, hdf5 formats."""
        output_format = self.config.output_format
        channels_sorted = sorted(all_channel_data.keys())
        channel_dicts = [all_channel_data[ch] for ch in channels_sorted]

        if output_format == "npz":
            for channel_data in channel_dicts:
                save_dict = {
                    "x": channel_data["signal"],
                    "fs": channel_data["sampling_rate"],
                    "ch_label": channel_data["ch_name"],
                    "ch_label_orig": channel_data["ch_name_orig"],
                    "file_duration": len(channel_data["signal"]) * self.config.epoch_duration,
                    "epoch_duration": self.config.epoch_duration,
                    "n_epochs": len(channel_data["signal"]),  # after cleaning
                }

                # Write Annotations
                if self.config.use_annot:
                    # Handle multiple scorers
                    labels = channel_data["labels"]
                    if labels.ndim == 1:
                        save_dict["y"] = labels
                    elif labels.ndim == 2:
                        save_dict["y"] = labels[:, 0]
                        save_dict["y2"] = labels[:, 1]
                save_dict["unit"] = channel_data.get("unit", "a.u.")
                np.savez(channel_data["file_output_path"], **save_dict)

        elif output_format == "edf":
            file_output_path = channel_dicts[0]["file_output_path"]
            with EdfWriter(file_output_path, n_channels=len(channel_dicts)) as edf_writer:
                for i, channel_data in enumerate(channel_dicts):
                    signal = channel_data["signal"].flatten()
                    scale = 10**3  # to get 3 decimals for physical min and max
                    if len(signal) == 0:
                        print("here")
                    phys_min = floor(np.nanmin(signal) * scale) / scale
                    phys_max = ceil(np.nanmax(signal) * scale) / scale
                    if phys_min == phys_max:
                        phys_min -= 1.0
                        phys_max += 1.0
                    channel_info = {
                        "label": channel_data["ch_name"],
                        "dimension": channel_data.get("unit", "a.u."),
                        "sample_frequency": float(channel_data["sampling_rate"]),
                        "physical_min": phys_min,
                        "physical_max": phys_max,
                        "digital_min": -32768,
                        "digital_max": 32767,
                        "transducer": "",
                        "prefilter": "",
                    }
                    edf_writer.setSignalHeader(i, channel_info)
                if isinstance(channel_dicts[0]["start_datetime"], datetime):
                    if channel_dicts[0]["start_datetime"].date() > date(1985, 1, 1):
                        Startdatetime = channel_dicts[0]["start_datetime"]
                    else:
                        Startdatetime = datetime.combine(date(1985, 1, 1), channel_dicts[0]["start_datetime"].time())
                else:
                    Startdatetime = datetime(1985, 1, 1, 0, 0, 0)

                # Write samples
                edf_writer.setStartdatetime(Startdatetime)
                all_signals = [channel_data["signal"].flatten() for channel_data in channel_dicts]
                edf_writer.writeSamples(all_signals)

                # Write annotations
                if self.config.use_annot:
                    all_labels = np.array([channel_data["labels"] for channel_data in channel_dicts])
                    duration = self.config.epoch_duration
                    for epoch_idx, label in enumerate(all_labels[0]):
                        edf_writer.writeAnnotation(
                            onset_in_seconds=epoch_idx * duration,
                            duration_in_seconds=duration,
                            description=str(label),
                        )

        elif output_format == "hdf5":
            file_output_path = channel_dicts[0]["file_output_path"]
            with h5py.File(file_output_path, "w") as h5f:
                # Metadata
                h5f.attrs["epoch_duration"] = self.config.epoch_duration
                h5f.attrs["file_duration"] = len(channel_dicts[0]["signal"]) * self.config.epoch_duration
                h5f.attrs["n_epochs"] = len(channel_dicts[0]["signal"])
                grp_signals = h5f.create_group("signals")
                for channel_data in channel_dicts:
                    signal = channel_data["signal"].flatten()

                    # if channel_name came from h5 originally, keep only the last part after slash
                    if "h5" in self.dataset.file_extensions["psg_ext"] and "/" in channel_data["ch_name"]:
                        group_name = channel_data["ch_name"].split("/")[-1]
                    else:
                        group_name = channel_data["ch_name"]
                    ch_grp = grp_signals.create_group(group_name)
                    ch_grp.create_dataset("data", data=signal, compression="gzip", shuffle=True)
                    ch_grp.attrs["ch_label"] = channel_data["ch_name"]
                    ch_grp.attrs["ch_label_orig"] = channel_data["ch_name_orig"]
                    ch_grp.attrs["unit"] = channel_data.get("unit", "a.u.")
                    ch_grp.attrs["sampling_rate"] = channel_data["sampling_rate"]

                # Annotations
                if self.config.use_annot:
                    labels = np.asarray(channel_dicts[0]["labels"])
                    if labels.ndim == 1:
                        h5f.create_dataset("y", data=labels, compression="gzip")
                    elif labels.ndim == 2:
                        h5f.create_dataset("y", data=labels[:, 0], compression="gzip")
                        h5f.create_dataset("y2", data=labels[:, 1], compression="gzip")


class ChannelProcessor:

    def __init__(self, logger, config, dataset, channel):
        self.logger = logger
        self.config = config
        self.dataset = dataset
        self.channel = channel

    def _process_channel(
        self,
        data,
    ):
        """Process a single channel from a single file."""

        data["ch_name_orig"] = self.channel
        self.logger.info(f"Channel selected: {data['ch_name_orig']}")

        # Extract data from psg file and add to data dictionary
        psg_data = self.dataset.get_signal_data(self.logger, data["psg_fname"], data["ch_name_orig"])
        if psg_data == {}:
            return None
        data.update(psg_data)
        del psg_data  # free memory

        self.logger.info(f"Select channel samples: {len(data['signal'])}")
        self.logger.info(f"File duration: {data['file_duration']} sec, {data['file_duration']/3600:.2f} h")

        # Process the signal (resample, filter, clean)
        signal = data["signal"].astype(np.float64)
        labels = data["labels"]
        fs = data["sampling_rate"]

        if self.config.resample is not None or self.config.filter:
            signal_processor = SignalProcessor(self.logger, signal, data["ch_name_orig"], self.config.filter_freq, self.dataset.channel_types)

            if self.config.resample is not None:
                # Resample signal
                signal_processor.resample_signal(fs, self.config.resample)
                fs = self.config.resample

            if self.config.filter:
                # Filter signal according to AASM
                signal_processor.filter_signal(fs, self.dataset.channel_groups)
            
            # Clip signal to original value range
            signal_processor.clip_signal()
            signal = signal_processor.signal

        if self.config.use_annot:
            # Apply start alignment of signal and annotations 
            start_time_shift, signal, labels = self._apply_start_shift(data, signal, labels, fs)

        # Reshape into epochs
        n_epoch_samples = self.config.epoch_duration * fs
        if not n_epoch_samples.is_integer():
            raise ValueError(f"Epoch duration {self.config.epoch_duration} sec with sampling rate {fs} Hz "
                              "does not yield an integer number of samples per epoch.")
        n_epoch_samples = int(n_epoch_samples)

        # Check signal length (at least one epoch required)
        n_epochs, remainder = divmod(len(signal), n_epoch_samples)
        if n_epochs < 1:
            self.logger.warning(
                f"{os.path.basename(data['psg_fname'])}: Channel {data['ch_name_orig']} does not hold at least one epoch, only {len(signal)} samples.")
            return None

        if remainder > 0:
            self.logger.info(f"Signal is cropped to full epochs ({remainder / fs:.2f} sec cropped).")

        signal_epoched = signal[: n_epochs * n_epoch_samples].reshape(n_epochs, -1)

        if self.config.use_annot:
            if len(signal_epoched) != len(labels):
                # Align end of signal and labels (some datasets have different length of signal and annotation data)
                signal_epoched, labels = self.dataset.align_end(
                    self.logger,
                    self.config.alignment,
                    self.config.pad_values,
                    data["psg_fname"],
                    data["ann_fname"],
                    signal_epoched,
                    labels,
                )

            assert len(signal_epoched) == len(labels), \
            f"Length mismatch: signal ({os.path.basename(data['psg_fname'])})={len(signal_epoched)}, \
                 labels({os.path.basename(data['ann_fname'])})={len(labels)} TODO: adapt alignment function"

        # Select only configured epochs (based on annotation, lights marker)
        new_startdatetime, signal_epoched, labels = self._select_epochs(
                    data["start_datetime"], start_time_shift, signal_epoched, labels,
                    data["lights_off"], data["lights_on"]
                    )
        if signal_epoched is None:
            return None

        # update signal, labels and sampling_rate after the processing
        data.update({"start_datetime": new_startdatetime, "signal": signal_epoched, "labels": labels, "sampling_rate": fs})

        return data

    def _apply_start_shift(self, data, signal, labels, fs):
        delay = data["start_delay"]
        start_time_shift = 0
        if delay != 0:
            # Align the start of signals and labels based on configuration
            start_time_shift, signal, labels = self.dataset.align_front(
                self.logger,
                self.config.alignment,
                self.config.pad_values,
                self.config.epoch_duration,
                delay,
                signal,
                labels,
                fs,
            )

            if start_time_shift:
                self.logger.info(f"Applied start time shift of {start_time_shift} seconds to align signal with annotation start time.")
                if isinstance(data["start_datetime"], datetime):
                    data["start_datetime"] = data["start_datetime"] + timedelta(seconds=start_time_shift)
                    self.logger.info(f"Adjusted start datetime after alignment: {data['start_datetime']}")
        return start_time_shift, signal, labels

    def _select_epochs(self, startdatetime, start_time_shift, signal_epoched, labels, lights_off_epoch, lights_on_epoch):
        """
        Clean signal by removing movement/unknown epochs and selecting sleep periods.

        Args:
            signal_epoched: Signal epochs array
            labels: Stage labels array

        Returns:
            Tuple of (cleaned_signals, cleaned_labels)
        """

        # 1. Get start and end indices if configured
        start_idx = 0
        end_idx = len(signal_epoched)

        sleep_mask = np.isin(labels, SLEEP_STAGES)
        sleep_idx = np.where(sleep_mask)[0]

        if self.config.select_epochs == "all":
            pass
        elif self.config.select_epochs == "lights":

            # Check for lights_off marker epoch
            if lights_off_epoch is not None:
                lights_off_epoch += start_time_shift // self.config.epoch_duration  # adjust lights off epoch based on start time shift if applied
                start_idx = lights_off_epoch

            # Check for lights_on marker epoch
            if lights_on_epoch is not None:
                lights_on_epoch += start_time_shift // self.config.epoch_duration  # adjust lights on epoch based on start time shift if applied
                if lights_on_epoch <= len(signal_epoched):
                    # valid lights on
                    end_idx = lights_on_epoch 
                elif lights_on_epoch == len(signal_epoched) + 1:
                    # lights on happened happenend in last unfilled epoch that was cropped -> keep all epochs
                    pass
                else:
                    self.logger.warning(f"Lights On is {len(signal_epoched) - lights_on_epoch} epochs after signal ends.")
                    pass
                    # maybe padding with wake epochs until lights On time is reached? For now, just keep all epochs and do not select based on lights On time
                    # raise Exception
            else:
                # if not lights on marker available, truncate all wake at end of file if ocnfigured
                if self.config.use_annot and self.config.truncate_non_sleep_end:
                    if len(sleep_idx) > 0:
                        end_idx = sleep_idx[-1] + 1
                        self.logger.info(f"Removed {len(signal_epoched)-end_idx} Non-Sleep epochs at the end of the night based on annotation data.")
            
        elif isinstance(self.config.select_epochs, int):
            # Remove extensive wake epochs at start and end as given in config
            n_select_epochs = self.config.select_epochs
            start_idx = max(0, sleep_idx[0] - n_select_epochs)
            end_idx = min(len(signal_epoched), sleep_idx[-1] + n_select_epochs + 1)

            if start_idx > 0 or end_idx != len(signal_epoched):
                n_crop = len(signal_epoched) - (end_idx - start_idx)
                self.logger.info(f"  Cropped {n_crop} epochs ({n_crop * self.config.epoch_duration / 60:.1f}min) of extensive wake")

        # 2. Get indices of movement and unknown epochs if configured
        if self.config.use_annot:
            if self.config.rm_move:
                move_idx = np.where(labels == STAGE_DICT["MOVE"])[0]
            else:
                move_idx = []
            if len(move_idx) > 0:
                self.logger.info(f"  Removing {len(move_idx)} Movement epochs")

            if self.config.rm_unk:
                unk_idx = np.where(labels == STAGE_DICT["UNK"])[0]
            else:
                unk_idx = []
            if len(unk_idx) > 0:
                self.logger.info(f"  Removing {len(unk_idx)} Unknown epochs")

            remove_idx = np.union1d(move_idx, unk_idx)
        else:
            remove_idx = []

        select_idx = np.setdiff1d(np.arange(start_idx, end_idx), remove_idx)

        signal_epoched = signal_epoched[select_idx]
        if labels is not None:
            labels = labels[select_idx]

        if signal_epoched.shape[0] == 0:
            self.logger.warning(f"No data left after selection of epochs. \
                            \n Start idx: {start_idx}, End idx: {end_idx}, Remove idx (Mov/Unk): {remove_idx}")
            return None, None, None

        self.logger.info(f"Data after selection: {signal_epoched.shape}, {labels.shape}")

        # Adapt start time
        new_startdatetime = startdatetime + timedelta(seconds=float(select_idx[0] * self.config.epoch_duration))

        return new_startdatetime, signal_epoched, labels

