import librosa 
import numpy as np
import matplotlib.pyplot as plt
import struct 
from config import * 

with open("calibration/73dB.bin", "rb") as f:
    dB_73 = struct.unpack("f", f.read(4))[0]

y, S_db, num_frames, times, freqs = load()
rms = librosa.feature.rms(y=y, frame_length=FRAME_LEN, hop_length=HOP_LEN)[0]
volume = librosa.amplitude_to_db(rms, ref=dB_73) + 73

valid_idx = np.where((freqs >= CUTOFF_L) & (freqs <= CUTOFF_H))[0]

f0_by_fft = np.full(num_frames, -1.0)

for frame in range(num_frames):
    spectrum = smooth_spectrum_blackman(S_db[:, frame])

    if frame == np.argmin(np.abs(times - DEBUG_TIME)):
        if DEBUG:
            breakpoint()

    peaks = []
    for idx in valid_idx[1:-2]:
        if spectrum[idx - 1] < spectrum[idx] > spectrum[idx + 1]:
            x = freqs[idx - 2:idx + 3]
            y = spectrum[idx - 2:idx + 3]
            a, b, _ = np.polyfit(x, y, 2)
            peak_freq = -b / (2 * a)

            if len(peaks) > 0 and spectrum[idx] < peaks[-1][1] - 20:
                continue
            elif len(peaks) == 1 and (peaks[-1][1] + 15 < spectrum[idx]) and (CUTOFF_L <= peak_freq <= CUTOFF_H) and (a < MAX_A):
                if abs(freqs[idx] - peak_freq) < PEAK_SHIFT_TOLERANCE:
                    peaks[-1] = [peak_freq, spectrum[idx]]
                else:
                    peaks[-1] = [freqs[idx], spectrum[idx]]
            elif (len(peaks) < MAX_PEAK) and (CUTOFF_L <= peak_freq <= CUTOFF_H) and (a < MAX_A):
                if abs(freqs[idx] - peak_freq) < PEAK_SHIFT_TOLERANCE:
                    peaks.append([peak_freq, spectrum[idx]])
                else:
                    peaks.append([freqs[idx], spectrum[idx]])

    peaks = np.array(peaks, dtype=float)

    i = 1
    while i < len(peaks) - 1:
        x = peaks[i-1:i+2, 0]
        y = peaks[i-1:i+2, 1] 
        a, _, _ = np.polyfit(x, y, 2)
        if a > 3e-4 and x[1] > 800:
            peaks = np.delete(peaks, i, axis=0)
        else:
            i += 1

    if (volume[frame] < 40) or (len(peaks) == 0):
        f0_by_fft[frame] = 0.0
        continue

    if len(peaks) < MIN_PEAK:
        f0_by_fft[frame] = float(peaks[0][0])
        continue

    f0s = np.stack([np.diff(peaks[:, 0]), np.minimum(peaks[:-1, 1], peaks[1:, 1])], axis=1)

    f0s = f0s[f0s[:, 0] < FMAX]
    median = np.median(f0s[:, 0])
    f0s = f0s[f0s[:, 0] < 3 * median]
    median = np.median(f0s[:, 0])
    f0s = f0s[f0s[:, 0] < 2 * median]
    median = np.median(f0s[:, 0])

    i = 0
    while i < len(f0s):
        val = f0s[i, 0]
        if near(val / 2, median):
            f0s[i, 0] = val / 2
            i += 1
        elif near(val / 3, median):
            f0s[i, 0] = val / 3
            i += 1
        elif (i < len(f0s) - 1) and near(val + f0s[i + 1, 0], median):
            f0s[i + 1, 0] = f0s[i + 1, 0] + val
            f0s = np.delete(f0s, i, axis=0)
        else:
            i += 1

    median = np.median(f0s[:, 0])

    near_mask = near(f0s[:, 0], median)
    small_mask = f0s[:, 0] < median
    big_mask = f0s[:, 0] > median
    small_f0s = f0s[near_mask | small_mask]
    big_f0s   = f0s[near_mask | big_mask]

    small_prom = safe_mean(small_f0s[:, 1]) if len(small_f0s) else -np.inf
    big_prom   = safe_mean(big_f0s[:, 1])   if len(big_f0s)   else -np.inf

    f0s = f0s[near_mask]
    if len(f0s) == 0:
        if small_prom >= big_prom and len(small_f0s):
            f0_by_fft[frame] = safe_mean(small_f0s[:, 0])
        elif len(big_f0s):
            f0_by_fft[frame] = safe_mean(big_f0s[:, 0])
        else:
            f0_by_fft[frame] = peaks[0][0]
    else:
        f0_by_fft[frame] = safe_mean(f0s[:, 0])

# 畫圖
plt.figure(figsize=(10, 4))
plt.plot(times, f0_by_fft, label='FFT Estimated Pitch (Hz)', alpha=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Pitch Estimation with Band-Limited Peak Validation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.ylim(0, FMAX)
plt.show()
