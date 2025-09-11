import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def visualize_polysomnography(folder, channels, ext):
    """
    Visualizes polysomnography data from multiple .npz files.

    Each .npz file is expected to contain 'x.npy' (signal) and 'y.npy' (labels).
    Signals are sampled at 100Hz. Labels correspond to 30-second epochs.

    Args:
        file_paths (list): A list of paths to .npz files, where each file
                           represents a single channel of recording.
    """
    file_paths = []
    for ch in channels:
        filename = '*/' + ch + ext
        file_paths.append(glob.glob(os.path.join(folder,filename))[0])
        
        
    num_channels = len(file_paths)
    if num_channels == 0:
        print("No file paths provided. Please provide a list of .npz file paths.")
        return

    # Create a figure and subplots. Each channel gets two subplots:
    # one for the signal and one for the sleep stage labels.
    # 'sharex="col"' links the x-axes of all subplots in the same column,
    # enabling synchronized zooming and panning.
    fig, axes = plt.subplots(num_channels * 2, 1, sharex='col', figsize=(15, 4 * num_channels))

    # Define the mapping from integer labels to sleep stage names
    sleep_stage_map = {
        0: 'Wake',
        1: 'NREM 1',
        2: 'NREM 2',
        3: 'NREM 3',
        4: 'REM',
    }

    # Define distinct colors for each sleep stage for better visualization
    plot_stage_value_map = {
        0: 5,
        1: 3,
        2: 2,
        3: 1,
        4: 4,
    }

    # Prepare handles for the legend, which will explain the color coding
    legend_handles = []
    for stage_name, color in sleep_stage_colors.items():
        # Create a dummy rectangle patch for each stage to be used in the legend
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, ec="k", label=stage_name))

    # Iterate through each provided file path (channel)
    for i, file_path in enumerate(file_paths):
        try:
            # Load the signal (x) and labels (y) from the .npz file
            data = np.load(file_path)
            signal = data['x']
            labels = data['y']
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}. Skipping this channel.")
            continue # Move to the next file if there's an error

        # Calculate the time axis for the signal data
        sampling_rate = 100 # Hz, as specified by the user
        signal_time = np.arange(len(signal)) / sampling_rate

        # Calculate the start and end times for each 30-second epoch of labels
        epoch_duration = 30 # seconds, as specified by the user
        label_time_start = np.arange(len(labels)) * epoch_duration
        label_time_end = label_time_start + epoch_duration

        # Get the specific axes for the current channel's signal and labels
        ax_signal = axes[i * 2] # Axis for the signal plot
        ax_labels = axes[i * 2 + 1] # Axis for the labels plot

        # Plot the signal data
        ax_signal.plot(signal_time, signal, lw=0.5, color='darkblue')
        ax_signal.set_ylabel(f'Signal (mV)')
        ax_signal.set_title(f'Channel {i+1}: {os.path.basename(file_path)}') # Display filename in title
        ax_signal.grid(True, linestyle='--', alpha=0.7) # Add a subtle grid

        # Plot the sleep stage labels as colored background regions
        for j, label_val in enumerate(labels):
            stage_name = sleep_stage_map.get(label_val, f'Unknown {label_val}')
            color = sleep_stage_colors.get(stage_name, 'purple') # Default color if stage is unknown
            # Use axvspan to draw a vertical span (rectangle) for each epoch
            ax_labels.axvspan(label_time_start[j], label_time_end[j], color=color, alpha=0.4)

        ax_labels.set_yticks([]) # Hide y-axis ticks for labels, as they are represented by colors
        ax_labels.set_ylabel('Sleep Stage')
        ax_labels.set_ylim(0, 1) # Set y-limits to ensure axvspan fills the height
        ax_labels.grid(True, linestyle='--', alpha=0.7) # Add a subtle grid

        # Hide x-axis labels and ticks for all but the very bottom plot
        # This keeps the plot clean and avoids redundancy since x-axes are linked
        if i < num_channels - 1:
            ax_signal.tick_params(labelbottom=False)
            ax_labels.tick_params(labelbottom=False)

    # Set the common x-axis label for the bottom-most plot
    axes[-1].set_xlabel('Time (seconds)')

    # Add a single legend to the entire figure to explain the sleep stage colors
    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 1), title="Sleep Stages")

    # Adjust the layout to prevent labels/titles from overlapping and make space for the legend
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # rect adjusts the plot area to leave space on the right for legend
    plt.show() # Display the plot
    
channels = ['C4-A1','C3-A2','X2']

visualize_polysomnography(folder='/media/linda/Elements/sleep_data/ISRUC/ISRUC_harmonized/Subgroup 1/100Hz/npz',channels=channels,ext='_1.npz')