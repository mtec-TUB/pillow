import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import firwin, filtfilt
import tkinter as tk
from tkinter import filedialog


def visualize_multiple_signals(signals):
    """
    Visualizes multiple polysomnography signals from .npz files on a single plot.


    Each .npz file is expected to contain 'x.npy' (signal).
    The function takes a list of dictionaries, each specifying a file path
    and its corresponding sample rate.


    Args:
        signal_configs (list): A list of dictionaries, where each dictionary
                               has 'file_path' (str) and 'sample_rate' (int).
                               Example: [{'file_path': 'ch1.npz', 'sample_rate': 100}, ...]
    """


    if not signals:
        print("No signal configurations provided. Please provide a list of dictionaries.")
        return


    plt.figure(figsize=(15, 6)) # Create a single figure for all plots
    ax = plt.gca() # Get the current axes to plot all signals on


    max_time = 0 # To determine the maximum time extent for the x-axis


    # Iterate through each provided signal configuration
    for i, file_path in enumerate(signals):

        try:
            # Load the signal (x) from the .npz file
            data = np.load(file_path)
            signal = data['x']
            sample_rate = data['fs']
            print(sample_rate)
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}. Skipping this signal.")
            continue

        # signal = signal[int(len(signal)/2):]
        signal = signal.flatten()
        
        # Calculate the time axis for the current signal data
        signal_time = np.arange(len(signal)) / sample_rate
        
        # if 'orig' in file_path:
            # b = firwin(501,50,fs=sample_rate,pass_zero='lowpass')
            # filt_signal = filtfilt(b,1,signal)
            # ax.plot(signal_time, filt_signal,'o--',lw=2, label=f'{file_path} ({sample_rate}Hz)_filt')
            # yf = np.abs(rfft(filt_signal))
            # yf = yf/len(yf)
            # yf = 20*np.log10(yf)
            
            # xf = rfftfreq(len(filt_signal), 1/sample_rate)
        
            # yf[1:] = yf[1:]*2
            
            # plt.figure()
            # plt.plot(xf,yf)
        
        yf = np.abs(rfft(signal))
        yf = yf/len(yf)
        yf = 20*np.log10(yf)
        
        xf = rfftfreq(len(signal), 1/sample_rate)
        
        yf[1:] = yf[1:]*2
        
        plt.figure()
        plt.title(f'{os.path.basename(file_path)} ({sample_rate}Hz)')
        plt.plot(xf,yf)
        
        print(data['y'].shape)

        # Update max_time to ensure the x-axis covers all signals
        if len(signal_time) > 0:
            max_time = max(max_time, signal_time[-1])


        # Plot the signal data on the same axes
        # Use a label for the legend to identify each signal
        if 'orig' in file_path:
            ax.plot(signal_time, signal,label=f'{os.path.basename(file_path)} ({sample_rate}Hz)')
        else:
            ax.plot(signal_time, signal,label=f'{os.path.basename(file_path)} ({sample_rate}Hz)')


    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Signal')
    ax.grid(True, linestyle='--', alpha=0.7) 
    ax.legend(loc='upper right') # Display the legend to identify each signal
    ax.set_xlim(0, max_time) # Ensure x-axis spans the full duration of all signals


    plt.tight_layout()
    plt.show()

# #'/media/linda/Elements/sleep_data/100Hz/EEG3_mesa-sleep-0002.npz',
signals = [
            # '/media/linda/Elements/sleep_data/APPLES - Apnea Positive Pressure Long-term Efficacy Study/APPLES_harmonized/100Hz_filt/npz/ROC/ROC_apples-460164.npz',
            # '/media/linda/Elements/sleep_data/APPLES - Apnea Positive Pressure Long-term Efficacy Study/APPLES_harmonized/orig/npz/ROC/ROC_apples-460164.npz',
            # '/media/linda/Elements/sleep_data/CPS - Comprehensive Polysomnography Dataset (A Resource for Sleep-Related Arousal Research)/CPS_harmonized/100Hz_filt/npz/EOGl/EOGl_0Ah95Qw18puf1JsnrKBA6u8XXZLlMIQJ.npz',
            # '/media/linda/Elements/sleep_data/CPS - Comprehensive Polysomnography Dataset (A Resource for Sleep-Related Arousal Research)/CPS_harmonized/orig/npz/EOGl/EOGl_0Ah95Qw18puf1JsnrKBA6u8XXZLlMIQJ.npz'

            '/media/linda/Elements/sleep_data/MESA - Multi-Ethnic Study of Atherosclerosis/MESA_harmonized/orig/npz/EMG/EMG_mesa-sleep-0002.npz',
            '/media/linda/Elements/sleep_data/MESA - Multi-Ethnic Study of Atherosclerosis/MESA_harmonized/100Hz_filt/npz/EMG/EMG_mesa-sleep-0002.npz',
            # '/media/linda/Elements/sleep_data/MESA - Multi-Ethnic Study of Atherosclerosis/MESA_harmonized/100Hz_filt/npz/EEG1/EEG1_mesa-sleep-0001_after.npz',
            #'/media/linda/Elements/sleep_data/MESA - Multi-Ethnic Study of Atherosclerosis/MESA_harmonized/100Hz_filt/npz/EEG1/EEG1_mesa-sleep-0001_before.npz',
            ]


# # signals = [
# #     '/media/linda/Elements/sleep_data/SHHS - Sleep Heart Health Study/SHHS_harmonized/100Hz_filt/shhs1/npz/ECG/ECG_shhs1-200043.npz',
# #     '/media/linda/Elements/sleep_data/SHHS - Sleep Heart Health Study/SHHS_harmonized/orig/shhs1/npz/ECG/ECG_shhs1-200043.npz',
# #     ]
visualize_multiple_signals(signals)

def select_npz_files():
    """
    Opens a file dialog to allow user to select multiple .npz files.
   
    Returns:
        list: List of selected file paths, or empty list if canceled
    """
    # Create a root window and hide it
    root = tk.Tk()
    root.withdraw()  # Hide the main window
   
    # Open file dialog for multiple .npz files
    file_paths1 = filedialog.askopenfilenames(
        title="Select .npz files to visualize",
        filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")],
        multiple=True,
        initialdir="/media/linda/Elements/sleep_data"
    )

    # Open file dialog for multiple .npz files
    file_paths2 = filedialog.askopenfilenames(
        title="Select .npz files to visualize",
        filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")],
        multiple=True,
        initialdir="/media/linda/Elements/sleep_data"
    )
   
    root.destroy()  # Clean up the root window
   
    # Convert tuple to list
    return list(file_paths1) + list(file_paths2)




# # Select files using file dialog
# print("Please select the .npz files you want to visualize...")
# selected_files = select_npz_files()


# if selected_files:
#     print(f"Selected {len(selected_files)} files:")
#     for file_path in selected_files:
#         print(f"  - {os.path.basename(file_path)}")
   
#     # Visualize the selected signals
#     visualize_multiple_signals(selected_files)
# else:
#     print("No files selected. Exiting...")


