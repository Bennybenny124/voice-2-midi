import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time
import struct
from system_config import *
from midi.messages import *

START = 0
END = 0

with open("calibration/73dB.bin", "rb") as f:
    dB_73 = struct.unpack("f", f.read(4))[0]

freqs = np.fft.rfftfreq(FRAME_LEN, d=1.0 / SR)

def dft(frame, frame_len = FRAME_LEN, window_fn=np.hanning):
    window = window_fn(frame_len)
    xw = frame * window
    spectrum = np.fft.rfft(xw)
    magnitude = np.abs(spectrum)
    return magnitude

def get_volume(frame):
    x = frame.astype(np.float32) / 32768.0
    rms = np.sqrt(safe_mean(x**2))
    return 20.0 * np.log10(rms / dB_73) + 73

def find_f0(db: np.ndarray, volume_db: float): 
    spectrum = smooth_spectrum_blackman(db)

    valid_idx = np.where((freqs >= CUTOFF_L) & (freqs <= CUTOFF_H))[0]

    peaks = []
    for idx in valid_idx[1:-2]:
        if spectrum[idx - 1] < spectrum[idx] > spectrum[idx + 1]:
            x = freqs[idx - 2:idx + 3]
            y = spectrum[idx - 2:idx + 3]
            a, b, _ = np.polyfit(x, y, 2)
            peak_freq = -b / (2 * a)

            if len(peaks) > 0 and spectrum[idx] < peaks[-1][1] - 22:
                continue
            if len(peaks) == 1 and (peaks[-1][1] + 15 < spectrum[idx]) and (CUTOFF_L <= peak_freq <= CUTOFF_H) and (a < MAX_A):
                if abs(freqs[idx] - peak_freq) < PEAK_SHIFT_TOLERANCE:
                    peaks[-1] = [peak_freq, spectrum[idx], idx]
                else:
                    peaks[-1] = [freqs[idx], spectrum[idx], idx]
            elif len(peaks) < MAX_PEAK and CUTOFF_L <= peak_freq <= CUTOFF_H and a < MAX_A:
                if abs(freqs[idx] - peak_freq) < PEAK_SHIFT_TOLERANCE:
                    peaks.append([peak_freq, spectrum[idx], idx])
                else:
                    peaks.append([freqs[idx], spectrum[idx], idx])
    
    peaks = np.array(peaks, dtype=float)

    i = 1
    while i < len(peaks) - 1:
        x = peaks[i-1:i+2, 0]
        y = peaks[i-1:i+2, 1]
        a, _, _ = np.polyfit(x, y, 2)
        if a > 3e-4 and peaks[i][1] < -40:
            peaks = np.delete(peaks, i, axis=0)
        else:
            i += 1

    if len(peaks) == 0 or volume_db < 40:
        return 0.0
    elif len(peaks) < MIN_PEAK:
        return peaks[0][0]
    else:
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

        f0s = f0s[near(f0s[:, 0], median)] 

        if len(f0s) == 0:
            if small_prom >= big_prom:
                return safe_mean(small_f0s[:, 0])
            else:
                return safe_mean(big_f0s[:, 0])
        else:
            return safe_mean(f0s[:, 0])

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SR, input=True, frames_per_buffer=FRAME_LEN)

# 建立畫布
figure, plots = plt.subplots(3, 1)
figure.tight_layout()
a = plots[0]
b = plots[1]
c = plots[2]

ax = np.arange(0, TIME_WINDOW_SEC * SR)
ay = np.zeros(TIME_WINDOW_SEC * SR)
aline, = a.plot(ax, ay)
a.set_xlim([0, TIME_WINDOW_SEC * SR])
a.set_ylim([-32768, 32767])

bx = np.arange(0, TIME_WINDOW_SEC * SR // HOP_LEN)
by = np.zeros(TIME_WINDOW_SEC * SR // HOP_LEN)
bline, = b.plot(bx, by)
b.set_xlim([0, TIME_WINDOW_SEC * SR // HOP_LEN])
b.set_ylim([0, 500])

cx = np.arange(0, TIME_WINDOW_SEC * SR // HOP_LEN)
cy = np.zeros(TIME_WINDOW_SEC * SR // HOP_LEN)
cline, = c.plot(cx, cy)
c.set_xlim([0, TIME_WINDOW_SEC * SR // HOP_LEN])
c.set_ylim([-50, 50])

buffer_sample = np.zeros(TIME_WINDOW_SEC * SR, dtype=np.int16)
buffer_f0 = np.zeros(TIME_WINDOW_SEC * SR // HOP_LEN, dtype=np.float32)
buffer_dynamic = np.zeros(TIME_WINDOW_SEC * SR // HOP_LEN, dtype=np.float32)

for i in range(1000):
    START = time.time()
    for i in range(FRAME_LEN // HOP_LEN - 1, -1, -1):
        frame = buffer_sample[-i * HOP_LEN - FRAME_LEN : -i * HOP_LEN if i != 0 else None]
        magnitude = dft(frame, FRAME_LEN)

        volume = get_volume(frame)
        f0 = find_f0(magnitude, volume)

        buffer_f0 = np.roll(buffer_f0, -1)
        buffer_f0[-1] = f0
        buffer_dynamic = np.roll(buffer_dynamic, -1)
        buffer_dynamic[-1] = volume

        bline.set_ydata(buffer_f0)
        cline.set_ydata(buffer_dynamic)

    data = stream.read(FRAME_LEN, exception_on_overflow=False)
    new_samples = np.frombuffer(data, dtype=np.int16)

    buffer_sample = np.roll(buffer_sample, -FRAME_LEN)
    buffer_sample[-FRAME_LEN:] = new_samples
    
    aline.set_ydata(buffer_sample)

    END = time.time()
    plt.pause(max(0.0001, TICK + START - END))

stream.stop_stream()
stream.close()
p.terminate()
