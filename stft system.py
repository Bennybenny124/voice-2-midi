import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time
from system_config import *
from midi.messages import *
from calibration.spl import load_reference

dB_73 = load_reference()
freqs = np.fft.rfftfreq(FRAME_LEN, d=1.0 / SR)
cut_freqs = freqs[np.where((freqs >= CUTOFF_L) & (freqs <= CUTOFF_H))[0]]

def find_f0(spectrum: np.ndarray, volume: float) -> float:
    '''
    Returns f0 based on the given spectrum and volume. 

    Args:
        spectrum (np.ndarray): the fft spectrum between two cutoff frequencies, in decibles and with a blackman filter applied.
        volume (float): calibrated volume, in decibles.
    
    Returns:
        float: estimated f0
    '''
    peaks = []
    for idx in range(2, len(spectrum) - 3):
        if spectrum[idx - 1] < spectrum[idx] > spectrum[idx + 1]:
            x = cut_freqs[idx - 2:idx + 3]
            y = spectrum[idx - 2:idx + 3]
            a, b, _ = np.polyfit(x, y, 2)
            peak_freq = -b / (2 * a)

            if len(peaks) > 0 and spectrum[idx] < peaks[-1][1] - 22:
                continue
            else:
                not_shifted = abs(cut_freqs[idx] - peak_freq) < PEAK_SHIFT_TOLERANCE
                if len(peaks) >= 1 and (peaks[0][1] + 10 < spectrum[idx]):# and (a < MAX_A):
                    peaks = [[peak_freq if not_shifted else cut_freqs[idx], spectrum[idx]]]
                elif len(peaks) < MAX_PEAK and CUTOFF_L <= peak_freq <= CUTOFF_H:# and a < MAX_A:
                        peaks.append([peak_freq if not_shifted else cut_freqs[idx], spectrum[idx]])
    peaks = np.array(peaks, dtype=float)

    # i = 1
    # while i < len(peaks) - 1:
    #     x = peaks[i-1:i+2, 0]
    #     y = peaks[i-1:i+2, 1]
    #     a, _, _ = np.polyfit(x, y, 2)
    #     if a > 3e-4 and x[1] > 800:
    #         peaks = np.delete(peaks, i, axis=0)
    #     else:
    #         i += 1

    if len(peaks) == 0 or volume < 30 or is_flat(spectrum):
        return 0.0
    else:
        return peaks[0][0]
    # elif len(peaks) < MIN_PEAK:
    #     return peaks[0][0]
    # else:
    #     f0s = np.stack([np.diff(peaks[:, 0]), np.minimum(peaks[:-1, 1], peaks[1:, 1])], axis=1)
    #     f0s = f0s[f0s[:, 0] < FMAX]
    #     median = np.median(f0s[:, 0])
    #     f0s = f0s[f0s[:, 0] < 3 * median]
    #     median = np.median(f0s[:, 0])
    #     f0s = f0s[f0s[:, 0] < 2 * median]
    #     median = np.median(f0s[:, 0])

    #     i = 0
    #     while i < len(f0s):
    #         val = f0s[i, 0]
    #         if near(val / 2, median):
    #             f0s[i, 0] = val / 2
    #             i += 1
    #         elif near(val / 3, median):
    #             f0s[i, 0] = val / 3
    #             i += 1
    #         elif (i < len(f0s) - 1) and near(val + f0s[i + 1, 0], median):
    #             f0s[i + 1, 0] = f0s[i + 1, 0] + val
    #             f0s = np.delete(f0s, i, axis=0)
    #         else:
    #             i += 1

    #     median = np.median(f0s[:, 0])
        
    #     near_mask = near(f0s[:, 0], median)
    #     small_mask = f0s[:, 0] < median
    #     big_mask = f0s[:, 0] > median
    #     small_f0s = f0s[near_mask | small_mask]
    #     big_f0s   = f0s[near_mask | big_mask]

    #     small_prom = safe_mean(small_f0s[:, 1]) if len(small_f0s) else -np.inf
    #     big_prom   = safe_mean(big_f0s[:, 1])   if len(big_f0s)   else -np.inf

    #     f0s = f0s[near(f0s[:, 0], median)] 

    #     if len(f0s) == 0:
    #         if small_prom >= big_prom:
    #             return safe_mean(small_f0s[:, 0])
    #         else:
    #             return safe_mean(big_f0s[:, 0])
    #     else:
    #         return safe_mean(f0s[:, 0])

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SR, input=True, frames_per_buffer=FRAME_LEN)

figure, plots = plt.subplots(3, 1)
figure.tight_layout()
a = plots[0]
b = plots[1]
c = plots[2]

ax = np.arange(0, TIME_WINDOW_SEC * SR // HOP_LEN)
ay = np.zeros(TIME_WINDOW_SEC * SR // HOP_LEN)
aline, = a.plot(ax, ay)
a.set_xlim([0, TIME_WINDOW_SEC * SR // HOP_LEN])
a.set_ylim([0, 500])

bx = np.arange(0, TIME_WINDOW_SEC * SR // HOP_LEN)
by = np.zeros(TIME_WINDOW_SEC * SR // HOP_LEN)
bline, = b.plot(bx, by)
b.set_xlim([0, TIME_WINDOW_SEC * SR // HOP_LEN])
b.set_ylim([20, 100])

cy = np.zeros(len(cut_freqs))
cline, = c.plot(cut_freqs, cy)
c.set_xlim([CUTOFF_L, CUTOFF_H])
c.set_ylim([-100, 0])

buffer_sample = np.zeros(TIME_WINDOW_SEC * SR, dtype=np.int16)
buffer_f0 = np.zeros(TIME_WINDOW_SEC * SR // HOP_LEN, dtype=np.float32)
buffer_dynamic = np.zeros(TIME_WINDOW_SEC * SR // HOP_LEN, dtype=np.float32)

for i in range(1000):
    START = time.time()
    for i in range(FRAME_LEN // HOP_LEN - 1, -1, -1):
        frame = buffer_sample[-i * HOP_LEN - FRAME_LEN : -i * HOP_LEN if i != 0 else None]
        raw_spectrum = dft(frame, FRAME_LEN)
        dB_spectrum = 20.0 * np.log10(raw_spectrum / dB_73) - 140 # normalize for visualization only
        smoothed_spectrum = smooth_spectrum_blackman(dB_spectrum)
        spectrum = smoothed_spectrum[np.where((freqs >= CUTOFF_L) & (freqs <= CUTOFF_H))[0]]

        volume = get_volume(frame, dB_73, 73)
        f0 = find_f0(spectrum, volume)

        buffer_f0 = np.roll(buffer_f0, -1)
        buffer_f0[-1] = f0
        buffer_dynamic = np.roll(buffer_dynamic, -1)
        buffer_dynamic[-1] = volume

        aline.set_ydata(buffer_f0)
        bline.set_ydata(buffer_dynamic)
        cline.set_ydata(spectrum)

    data = stream.read(FRAME_LEN, exception_on_overflow=False)
    new_samples = np.frombuffer(data, dtype=np.int16)

    buffer_sample = np.roll(buffer_sample, -FRAME_LEN)
    buffer_sample[-FRAME_LEN:] = new_samples

    END = time.time()
    plt.pause(max(0.0001, TICK + START - END))

stream.stop_stream()
stream.close()
p.terminate()
