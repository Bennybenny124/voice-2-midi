import pyaudio
import numpy as np
import librosa
import matplotlib.pyplot as plt
import time
import struct
from system_config import *

FRAME_LEN = 2048
HOP = 256
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
TIME = 1
TICK = HOP / RATE
START = 0
END = 0

with open("calibration/73dB.bin", "rb") as f:
    dB_73 = struct.unpack("f", f.read(4))[0]

freqs = np.fft.rfftfreq(FRAME_LEN, d=1.0 / RATE)
cut_freqs = freqs[np.where((freqs >= CUTOFF_L) & (freqs <= CUTOFF_H))[0]]

def find_f0(spectrum, last_f0, volume):
    if volume < 30 or is_flat(spectrum): return 0

    peak = 0
    for i in range(1, len(spectrum) - 1):
        if spectrum[i - 1] < spectrum[i] > spectrum[i + 1]:
            x = freqs[i - 1:i + 2]
            y_fit = spectrum[i - 1:i + 2]
            a, b, _ = np.polyfit(x, y_fit, 2)
            peak_freq = -b / (2 * a)

            if peak_freq > FMAX * 1.1:
                break
            if FMIN < peak_freq and peak == 0:
                peak = peak_freq
            elif FMIN < peak_freq and last_f0 != 0 and (abs(1.0 - peak_freq / last_f0) < abs(1.0 - peak / last_f0)):
                peak = peak_freq
    return peak

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAME_LEN)

figure, plots = plt.subplots(3, 1)
figure.tight_layout()
a = plots[0]
b = plots[1]
c = plots[2]

ax = np.arange(0, TIME * RATE)
ay = np.zeros(TIME * RATE)
aline, = a.plot(ax, ay)
a.set_xlim([0, TIME * RATE])
a.set_ylim([-32768, 32767])

bx = np.arange(0, TIME * RATE)
by = np.zeros(TIME * RATE)
bline, = b.plot(bx, by)
b.set_xlim([0, TIME * RATE])
b.set_ylim([0, 500])

cx = np.arange(0, TIME * RATE)
cy = np.zeros(TIME * RATE)
cline, = c.plot(cx, cy)
c.set_xlim([0, TIME * RATE])
c.set_ylim([-50, 50])

buffer_sample = np.zeros(TIME * RATE, dtype=np.int16)
buffer_f0 = np.zeros(TIME * RATE, dtype=np.float32)
buffer_dynamic = np.zeros(TIME * RATE, dtype=np.float32)

for i in range(1000):
    START = time.time()
    for i in range(FRAME_LEN // HOP - 1, -1, -1):
        frame = buffer_sample[-i * HOP - FRAME_LEN : -i * HOP if i != 0 else None]
        raw_spectrum = dft(frame, FRAME_LEN)
        dB_spectrum = 20.0 * np.log10(raw_spectrum / np.max(raw_spectrum))
        smoothed_spectrum = smooth_spectrum_blackman(dB_spectrum)
        spectrum = smoothed_spectrum[np.where((freqs >= CUTOFF_L) & (freqs <= CUTOFF_H))[0]]
        
        volume = get_volume(frame, dB_73, 73)
        f0 = find_f0(spectrum, buffer_f0[-1],volume)
        
        buffer_f0 = np.roll(buffer_f0, -HOP)
        buffer_f0[-HOP:] = f0
        buffer_dynamic = np.roll(buffer_dynamic, -HOP)
        buffer_dynamic[-HOP:] = volume

        bline.set_ydata(buffer_f0)
        cline.set_ydata(buffer_dynamic)

    data = stream.read(FRAME_LEN, exception_on_overflow=False)
    new_samples = np.frombuffer(data, dtype=np.int16)

    buffer_sample = np.roll(buffer_sample, -FRAME_LEN)
    buffer_sample[-FRAME_LEN:] = new_samples
    
    aline.set_ydata(buffer_sample)

    END = time.time()
    plt.pause(max(0.0001, TICK - START + END))

stream.stop_stream()
stream.close()
p.terminate()
