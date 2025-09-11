
import numpy as np
import matplotlib.pyplot as plt
import os


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

        print(signal.shape)
        
        signal = signal[-15:]
        signal = signal.flatten()
        print(signal.shape)
        
        print(data['y'])
        print(data['y'].shape)
        # Calculate the time axis for the current signal data
        signal_time = np.arange(len(signal)) / sample_rate


        # Update max_time to ensure the x-axis covers all signals
        if len(signal_time) > 0:
            max_time = max(max_time, signal_time[-1])


        # Plot the signal data on the same axes
        # Use a label for the legend to identify each signal
        if 'orig' in file_path:
            ax.plot(signal_time, signal, 'o',lw=2, label=f'{file_path} ({sample_rate}Hz)')
        else:
            ax.plot(signal_time, signal,'o', lw=2, label=f'{file_path} ({sample_rate}Hz)')


    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Signal')
    ax.grid(True, linestyle='--', alpha=0.7) # Add a subtle grid
    ax.legend(loc='upper right') # Display the legend to identify each signal
    ax.set_xlim(0, max_time) # Ensure x-axis spans the full duration of all signals


    plt.tight_layout() # Adjust layout to prevent labels/titles from overlapping
    plt.show() # Display the plot


signals = ['/media/linda/Elements/sleep_data/CPS - Comprehensive Polysomnography Dataset/CPS_harmonized/orig/npz/Akku/Akku_0Ah95Qw18puf1JsnrKBA6u8XXZLlMIQJ.npz',
           '/media/linda/Elements/sleep_data/CPS - Comprehensive Polysomnography Dataset/CPS_harmonized/orig/npz/Licht/Licht_0Ah95Qw18puf1JsnrKBA6u8XXZLlMIQJ.npz',
           ]
visualize_multiple_signals(signals)

