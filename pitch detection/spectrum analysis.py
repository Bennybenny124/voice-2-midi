import numpy as np
import matplotlib.pyplot as plt
import struct
from config import *

y, S_db, num_frames, times, freqs = load()
silence_spectrum = smooth_spectrum_blackman(np.full((1 + FRAME_LEN // 2,), -80))
compensation = -silence_spectrum - 80

t = 0.5
while t >= 0:
    try: t = float(input("Please enter the time(in seconds). Exit if input is negative: "))
    except ValueError: print("Invalid input!"); continue
    if t < 0: print("Exit."); break

    frame = np.argmin(np.abs(times - t))
    spectrum = np.array(S_db[:, frame])
    spectrum = smooth_spectrum_blackman(spectrum) + compensation

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, spectrum)

    peaks = []
    for i in range(1, len(spectrum) - 1):
        if spectrum[i - 1] < spectrum[i] > spectrum[i + 1]:
            x = freqs[i - 2:i + 3]
            y = spectrum[i - 2:i + 3]
            a, b, c = np.polyfit(x, y, 2)

            x_fit = np.linspace(x[0], x[-1], 64)
            y_fit = a * x_fit**2 + b * x_fit + c
            plt.plot(x_fit, y_fit, 'r--', linewidth=1)

            x_vertex = -b / (2 * a) if a != 0 else x[1]
            y_vertex = a * x_vertex**2 + b * x_vertex + c
            plt.scatter([x_vertex], [y_vertex], s=12, c='r')
            plt.text(x_vertex, y_vertex, f"a={a:.2e}", color='red', fontsize=8, ha='left', va='bottom')
            
            if a < MAX_A:
                peaks.append([x_vertex, y_vertex])

    i = 0
    while i + 2 < len(peaks):
        xs = [peaks[i][0], peaks[i + 1][0], peaks[i + 2][0]]
        ys = [peaks[i][1], peaks[i + 1][1], peaks[i + 2][1]]

        a, b, c = np.polyfit(xs, ys, 2)

        x_fit = np.linspace(xs[0], xs[-1], 128)
        y_fit = a * x_fit**2 + b * x_fit + c

        if a > 0: plt.plot(x_fit, y_fit, linestyle='--', color='orange', linewidth=1)

        if a > 0:
            plt.scatter(xs[1], ys[1], s=12, c='orange')
            plt.text(xs[1], ys[1] + 2, f"a={a:.2e}", color='orange', fontsize=12, ha='center', va='bottom')

        if a > 3e-4:
            del peaks[i + 1]
        else:
            i += 1

    plt.title(f"Frequency Slice at t = {t:.3f} s")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(CUTOFF_L, CUTOFF_H)
    plt.show()
