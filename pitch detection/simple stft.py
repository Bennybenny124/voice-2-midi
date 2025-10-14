import librosa
import numpy as np
import matplotlib.pyplot as plt
import struct
from config import *

y, S_db, num_frames, times, freqs = load()

with open("calibration/73dB.bin", "rb") as f:
    dB_73 = struct.unpack("f", f.read(4))[0]
rms = librosa.feature.rms(y=y, frame_length=4*FRAME_LEN, hop_length=HOP_LEN)[0]
volume = librosa.amplitude_to_db(rms, ref=dB_73)

f0_by_fft = np.zeros(num_frames)
for frame in range(num_frames):
    spectrum = S_db[:, frame]

    peak = 0
    for i in range(1, len(spectrum) - 1):
        if spectrum[i - 1] < spectrum[i] > spectrum[i + 1]:
            x = freqs[i - 1:i + 2]
            y = spectrum[i - 1:i + 2]
            a, b, _ = np.polyfit(x, y, 2)
            freq = -b / (2 * a)

            if freq > FMAX * 1.1:
                break
            if FMIN < freq and peak == 0:
                peak = freq
            elif FMIN < freq and (abs(1.0 - freq / f0_by_fft[frame-1]) < abs(1.0 - peak / f0_by_fft[frame-1])):
                peak = freq

    if volume[frame] < -40 or is_flat(spectrum[np.where((freqs>CUTOFF_L)&(freqs<CUTOFF_H))]):
        f0_by_fft[frame] = -1
    else:
        f0_by_fft[frame] = peak

# plt.figure(figsize=(10, 4))
# plt.plot(times, f0_by_fft, label='FFT Estimated Pitch (Hz)', color='tab:orange', alpha=0.7)
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.title("Pitch Estimation")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.ylim(100, 600)
# plt.show()


fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(times, f0_by_fft, label='FFT Estimated Pitch (Hz)', color='tab:orange', alpha=0.7)
axes[0].set_ylabel("Frequency (Hz)")
axes[0].set_title("Pitch Estimation")
axes[0].grid(True)
axes[0].legend()
axes[0].set_ylim(100, 600)

axes[1].plot(times, volume, label='Volume (dB)', color='tab:blue', alpha=0.7)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Volume (dB)")
axes[1].set_title("Volume Estimation")
axes[1].grid(True)
axes[1].legend()
# axes[1].set_ylim(-60, 0)

plt.tight_layout()
plt.show()
