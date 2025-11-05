import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
file_path = "/media/linda/Elements/sleep_data/STAGES - Stanford Technology Analytics and Genomics in Sleep/STAGES_harmonized/100Hz_filt/npz/ECG/EKG_BOGN00001.npz"
file_path = "/home/linda/Downloads/MNC - Mignot Nature Communications/MNC_harmonized/100Hz_filt/dhc/training/npz/ECG/ECG_N0020-nsrr.npz"


start_time_sec = None   # e.g., 300.0  to start at 5 minutes
duration_sec = None     # e.g., 120.0 to show 2 minutes; None -> show all

# ---------- LOAD ----------
data = np.load(file_path, allow_pickle=True)

# Required keys (adapt if different)
x = data["x"]                    # expected shape (n_epochs, samples_per_epoch)
y = data["y"]                    # expected length n_epochs
fs = float(data["fs"].tolist())  # sampling frequency (ensure python float)
epoch_duration = float(data["epoch_duration"].tolist()) if "epoch_duration" in data else None

print(f"Removed at beginning: {data['rm_start_epochs']*30*2/60}min")

# Channel label / unit (optional)
ch_label = data.get("ch_label", None)
unit = data.get("unit", None)

# Normalize channel label to string if present
if ch_label is not None:
    if isinstance(ch_label, np.ndarray):
        if ch_label.size == 1:
            ch_label = str(ch_label.flatten()[0])
        else:
            # if it's array of labels unexpectedly, take first
            ch_label = str(ch_label.flatten()[0])
    else:
        ch_label = str(ch_label)

# ---------- SANITY CHECKS ----------
if x.ndim != 2:
    raise ValueError(f"Expected x to be 2D (n_epochs, samples_per_epoch). Got shape {x.shape}")

n_epochs, samples_per_epoch = x.shape
total_samples = n_epochs * samples_per_epoch

# If epoch_duration is missing, compute from fs and samples_per_epoch
if epoch_duration is None or epoch_duration <= 0:
    epoch_duration = samples_per_epoch / fs
else:
    # minor check
    expected = samples_per_epoch / fs
    if abs(expected - epoch_duration) > 1e-3:
        print(f"Warning: epoch_duration ({epoch_duration}s) != samples_per_epoch/fs ({expected:.6f}s)")

# Ensure hypnogram length matches n_epochs
y = np.asarray(y).flatten()
if len(y) != n_epochs:
    raise ValueError(f"Hypnogram length ({len(y)}) does not match number of epochs ({n_epochs})")

print(f"File: {file_path}")
print(f"Channels: {ch_label or 'single-channel file'}")
print(f"fs = {fs} Hz; epoch_duration = {epoch_duration}s")
print(f"n_epochs = {n_epochs}; total_samples = {total_samples}; total_time = {total_samples/fs/60:.2f} min")

# ---------- BUILD CONTINUOUS SIGNAL ----------
# Flatten epochs in chronological order to one 1D array
sig = x.reshape(-1)   # row-major: epoch0 then epoch1...
time = np.arange(sig.size) / fs  # seconds

# If user requested a time window, slice both signal and hypnogram appropriately
if start_time_sec is not None:
    # Clip start
    if start_time_sec < 0 or start_time_sec > time[-1]:
        raise ValueError("start_time_sec out of bounds")
    start_sample = int(round(start_time_sec * fs))
else:
    start_sample = 0

if duration_sec is not None:
    end_sample = min(sig.size, start_sample + int(round(duration_sec * fs)))
else:
    end_sample = sig.size

sig = sig[start_sample:end_sample]
time = time[start_sample:end_sample]

# For hypnogram plotting, compute epoch start times for the portion shown
epoch_start_idx = start_sample // samples_per_epoch
n_epochs_shown = int(np.ceil((end_sample) / samples_per_epoch)) - epoch_start_idx
t_epochs = (np.arange(epoch_start_idx, epoch_start_idx + n_epochs_shown) - epoch_start_idx) * epoch_duration
y_window = y[epoch_start_idx: epoch_start_idx + n_epochs_shown]

# ---------- MAP SLEEP STAGE NAMES ----------
# Default mapping (AASM-like). Adjust if your dataset uses another encoding.
default_map = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "Movement",
    -1: "Unknown"
}

# Detect unique codes and create name list
unique_codes = np.unique(y)
# Build y_names using default_map where possible, otherwise fallback to str(code)
y_names = [default_map.get(int(code), str(int(code))) for code in y_window]

# Define display order so that Wake is at top and N3 at bottom
display_order = ["Wake", "REM", "N1", "N2", "N3"]
# Numeric positions: Wake -> highest value
pos_map = {stage: len(display_order) - 1 - i for i, stage in enumerate(display_order)}
# Convert mapped names to numeric positions for plotting; unknowns -> NaN
y_plot = [pos_map.get(name, np.nan) for name in y_names]

# ---------- PLOTTING ----------
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})

# Signal plot
axes[0].plot(time, sig, lw=0.5)
axes[0].set_ylabel(f"{ch_label or 'Signal'} ({unit or 'a.u.'})")
axes[0].set_title(f"{ch_label or 'Signal'}  —  {file_path}")
axes[0].grid(True)

# Hypnogram plot (step)
axes[1].step(t_epochs, y_plot, where="post", lw=2)
# Y tick positions and labels (reverse display_order so Wake at top)
yticks = [pos_map[s] for s in display_order]
yticklabels = display_order  # Wake ... N3
axes[1].set_yticks(yticks)
axes[1].set_yticklabels(yticklabels)
axes[1].set_ylim(min(yticks) - 0.5, max(yticks) + 0.5)
axes[1].set_xlabel("Time (s) relative to window start")
axes[1].set_ylabel("Stage ")
axes[1].grid(True)

plt.tight_layout()
plt.show()
