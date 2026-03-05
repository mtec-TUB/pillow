
# PILLOW - Polysomnography Integrated Large-scale Library Of Waveforms

A comprehensive toolkit for processing and storing polysomnography (PSG) datasets in a harmonized way, supporting various file formats and automated signal processing.

---

## **Table of Contents**
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Supported Datasets](#supported-datasets)
- [Processing steps](#processing-steps)
- [Output](#output)
- [Adding New Datasets](#adding-new-datasets)

---

## **Architecture**

The codebase is organized into three main components:

```
├── datasets/                      # Dataset-specific logic
|   ├── file_handlers/             # Support for generic file formats
│   ├── base.py                    # Abstract base class for dataset
│   ├── registry.py                # Dataset registration and management
│   └── <dataset>.py               # One file per dataset
├── psg_processing/                # Core modules for processing
│   ├── core/
│   │   ├── dataset_explorer.py    # File search & channel analysis
│   │   ├── processor.py           # Main processing pipeline
│   │   └── signal_processor.py    # Filtering & resampling 
│   └── utils/
│       ├── config.py              # Helper class for process configuration
│       └── logging_manager.py     # Helper class for logging
├── config.yaml                    # Configuration file
└── process_dataset.py             # Main command-line interface

```

---

## **Quickstart**

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set configuration parameters**
    - Check default configuration in [config.yaml](/config.yaml)
    - modify them according to your needs (espacially the name of the dataset you want to process and where it is located (base_data_dir))

3. **Start the processing pipeline**
   ```bash
   python process_dataset.py
   ```
   or 
    ```bash
   python process_dataset.py --config /path/to/your/config.yaml
   ```

---

## **Supported Datasets**

We do not provide any data. Please request the data from the original authors and cite accordingly.

| Name        | Supported Version | Source                             |
| :---        | :----             | :---                               |
| ABC         | 0.4.0 (Feb 11, 2022)| https://doi.org/10.25822/nx52-bc11 |
| ANPHY       | Mar 19, 2025      | https://osf.io/r26fh/              |
| APOE        | 0.2.1 (Mar 1, 2024)| https://doi.org/10.25822/6ssj-2157 |
| APPLES      | 0.1.0 (Mar 15, 2023)| https://doi.org/10.25822/63pr-a591 |
| BESTAIR     | 0.6.1 (Oct 11, 2022)     | https://doi.org/10.25822/f656-yg39 |
| CAP         | 1.0.0             | https://doi.org/10.13026/C2VC79    |
| CCSHS       | 0.8.0 (Nov 14, 2023) | https://doi.org/10.25822/cg2n-4y91      |
| CFS         | Jun 12, 2020      | https://doi.org/10.25822/jmyx-mz90      |
| CHAT        | 0.14.0 (Nov 30, 2023) | https://doi.org/10.25822/d68d-8g03      |
| CPS         | 1.0.0             | https://doi.org/10.13026/sxs0-h317      |
| DCSM        | Mar 15, 2021      | https://doi.org/10.17894/ucph.282d3c1e-9b98-4c1e-886e-704afdfa9179 |
| DOD-H       | v1 (Jul 15, 2025) | https://doi.org/10.5281/zenodo.15900394      |
| DOD-O       | v1 (Jul 15, 2025) | https://doi.org/10.5281/zenodo.15900394      |
| DREAMT      | 1.0.0 (2.1.0 in prep)| https://doi.org/10.13026/dztc-dv77      |
| EESM17      | 1.0.5             | https://doi.org/10.18112/openneuro.ds004348.v1.0.5      |
| EESM19      | 1.0.2             | https://doi.org/10.18112/openneuro.ds005185.v1.0.2      |
| EESM23      | 1.0.0             | https://doi.org/10.18112/openneuro.ds005178.v1.0.0      |
| FDCSR       | 0.1.0 (Oct 12, 2023) | https://doi.org/10.25822/cxjt-6585      |
| HEARTBEAT   | 0.5.0 (Jun 8, 2022)| https://doi.org/10.25822/njzh-dk37      |
| HMC         | 1.1               | https://doi.org/10.13026/t4w7-3k21      |
| HOMEPAP     | 0.2.0 (Jul 19, 2022) | https://doi.org/10.25822/xmwv-yz91      |
| ISRUC       | (accessed May 3, 2024)   | https://sleeptight.isr.uc.pt/      |
| MESA        | 0.7.0 (Feb 21, 2024) | https://doi.org/10.25822/n7hq-c406      |
| MIT-BIH     | 1.0.0             | https://doi.org/10.13026/C23K5S      |
| MNC         | Jan 22, 2021      | https://doi.org/10.25822/00tc-zz78      |
| MrOS        | 0.6.0 (Apr 5, 2022) | https://doi.org/10.25822/kc27-0425      |
| MSP         | Apr, 2023         | https://doi.org/10.25822/gc7q-g526      |
| MWT         | v1 (Sep 27, 2019) | https://doi.org/10.5281/zenodo.3251716      |
| NCHSDB      | Jul, 2022         | https://doi.org/10.25822/jpdr-vz50      |
| Physio2018  | 1.0.0             | https://doi.org/10.13026/1q9b-ge17      |
| SLEEPBRL    | 1.0.0             | https://doi.org/10.13026/C29H4K      |
| SHHS        | 0.21.0 (Jul 2, 2024) | https://doi.org/10.25822/ghy8-ks59      |
| Sleep-EDF   | 1.0.0             | https://doi.org/10.13026/C2X676      |
| SOF         | Jun 12, 2020      | https://doi.org/10.25822/e1cf-rx65      |
| STAGES      | 0.3.0 (Jul 25, 2022) | https://doi.org/10.25822/me0d-xs45      |
| UCDDB       | 1.0.1             | https://doi.org/10.13026/C26C7D      |
| WSC         | 0.7.0 (Jul 17, 2024) | https://doi.org/10.25822/js0k-yh52      |

---

## **Processing steps**

The steps that are performed during processing depend heavily on your [config.yaml](/config.yaml). If you keep it as the default, the following will be applied (see [process_dataset.py](/process_dataset.py) and [processor.py](/psg_processing/core/processor.py)):

- the configuration file [config.yaml](/config.yaml) is loaded and checked for correct parameters
- all PSG files with matching extension (dataset.file_extension) are automatically detected and processed sequentially 
- the annotations are loaded (param: use_annot) either from a seperate corresponding annotation file or from the input PSG file itself (if no matching annotation is found, the PSG file is skipped)
- annotation file is checked for timing consistency, label violations and start time
- all available channels inside PSG file are processed sequentially
- channel name is harmonized across datasets (dataset.map_channel)
- output path is generated (see also [Output](#output))
- current file/channel is skipped, if the output path already exists (param: overwrite)
- signal is extracted from PSG file using defined file handler (dataset._file_handler)
- signal is resampled to 100Hz (param: resample):
    - digital channels (like Oxygen Saturation) are resampled with a nearest neighbour interpolation 
    - analog channels (like EEG) are resampled with polyphase filtering (mse.filter)
- signal is filtered based on AASM recomendation (param: filter)
- signal is clipped to the orignal range (because filtering may introduce overshootings)
- start of signal and labels/annotation is compared and handled to align them (param: alignment and pad_values)
- signal is truncated to whole epochs and reshaped to [num_epochs, epoch_dur * fs] (param: epoch_duration)
- end of signal and labels/annotations is compared and handled to align them (param: alignment and pad_values)
- signal is cleaned based on annotations
    - max. 30min of wake epochs are kept at beginning and ending of signal
    - epochs labeled with 'Movement' or 'Artifact' are removed
- signal, labels and other information is saved to the output_path (see [Output](#output))


---

## **Output**

You can choose the desired output format inside the [config.yaml](/config.yaml). Possible formats are: 'npz', 'edf' and 'hdf5'.

**Format `.npz`:**
The pipeline generates one .npz file per processed channel and per file in a channel based folder structure. Thus for example three input .edf files with each five recorded channels, result in five output folders named after the channels with each containing three .npz files. It is possible that not all output folders contain the number of input files, which only means that some input files recorded only a subset of channels. 

This structure is espacially helpful if you plan to use the processed data as input for [SleePyCo](https://github.com/gist-ailab/SleePyCo).

The .npz-files can be extracted afterwards with numpy.load() and contain the following entries:

- **x**: Processed signal data array with dim [num_epochs, epoch_dur * fs]
- **y**: Sleep stage annotations
- **fs**: Sampling rate
- **ch_label**: Harmonized channel name
- **ch_label_orig**: Channel name like it was originally found in the input file
- **file_duration**: Total duration of the signal in seconds
- **epoch_duration**: Duration of each epoch in seconds
- **n_epochs**: Number of epochs

If there is a second annotation entry called **y2**, this results from a second scorer (for example in ISRUC or EESM19 dataset)

**Format `.edf`:**
The pipeline generates one edf file per input psg file, including all chosen channels (with harmonized channel name). The corresponding annotations are stored inside the edf annotation channel (two dimensional, if more than one scorer).

**Format `.hdf5`:**
The pipeline generates one h5 file per input psg file, including all chosen channels (with harmonized channel name). You will find the metadata as h5 attributes and the annotations as a h5 dataset called 'y' (and possibly 'y2' for a second scorer). The signals are stored inside the h5 group 'signals', with each channel stored as a h5 dataset inside this group, with its corresponding metadata (channel name, sampling rate etc.) as attributes. You can use https://hdfviewer.com/ to check the structure before trying to load it programatically.

---

## **Adding New Datasets**

1. **Create a new dataset descriptive file:**
    - Add a file `datasets/<your_dataset>.py`.
    - Inherit from [BaseDataset](datasets/base.py) and implement the following methods:
        - `_setup_dataset_config` (specify file extensions as this is the only known property at the beginning)
        - `ann_parse`
        - `dataset_paths`
        - optionally `preprocess` (see [FDCSR dataset](/datasets/fdcsr.py))
        
    - register the dataset using the decorator `@register_dataset("YOUR_DS_NAME")`.
    - Check if the polysomnography file extension is already covered by one of the [file_handlers](/datasets/file_handlers/). For standardized formats (e.g. EDF, EEGLAB, WFDB) you can use the existing generic handlers by specifying them inside the __init__ function. For all others you have to implement the following functions yourself inside the dataset script to be able to use the pipeline:
        - get_channels()
        - read_signals()
        - get_signal_data()

    - Or you create a new generic handler inside [file_handlers](/datasets/file_handlers/), if the format could be reused across other datasets)

2. **Explore dataset:**
    - search for existing channel names inside your dataset by setting the parameter `action` inside [config.yaml](/config.yaml) to 'get_channel_names' and run
    ```bash
    python process_dataset.py
    ```
    - if you want to apply resampling and filtering to your data you need to know which channels are analog (e.g. EEG channels) and which digital (e.g. Oxygen saturation). You can find it out by setting the parameters `action` inside [config.yaml](/config.yaml) to 'get_channel_types' and run the main script again (Check the results manually to prevent wrong processing)
3. **Define all pending dataset properties:**
    - Now you are good to go to define all pending properties inside your script to make use of the whole processing pipeline. Inside `_setup_dataset_config`, specify the collected channel names, channel types, channel groups (to apply filtering according to AASM) and optional intra-dataset- and inter-dataset-channel-mappings (can be used to harmonize channel names, see [BESTAIR dataset](/datasets/bestair.py))
4. **Perform processing:**
    - set the `action` parameter back to 'process', choose all other configurations, start the pipeline and lay back :) 


---
