FILE = "test files/kisama.wav"
SR = 48000
FMIN = 100
FMAX = 600
FRAME_LEN = 2048
HOP_LEN = 512

CUTOFF_L = 100
CUTOFF_H = 3000
MIN_PEAK = 3
MAX_PEAK = 7

F0_RELATIVE_TOLERANCE = 0.08
MAX_A = -1.35e-3
PEAK_SHIFT_TOLERANCE = 8

DEBUG = False
DEBUG_TIME = 0.889

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
import librosa
import numpy as np
import struct

def load():
    y, _ = librosa.load(FILE, sr=SR)
    num_frames = 1 + (len(y) - FRAME_LEN) // HOP_LEN
    times = librosa.frames_to_time(np.arange(num_frames), sr=SR, hop_length=HOP_LEN) + FRAME_LEN / SR
    freqs = np.fft.rfftfreq(FRAME_LEN, d=1.0 / SR)
    S_db = np.zeros((len(freqs), num_frames))

    dB_73 = -1
    if dB_73 != -1:
        with open("calibration/73dB.bin", "rb") as f:
            dB_73 = struct.unpack("f", f.read(4))[0]
            print(dB_73)
    
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LEN, hop_length=HOP_LEN, center=False)[0]
    volume = librosa.amplitude_to_db(rms, ref=dB_73) + 73

    window = np.hanning(FRAME_LEN)
    for i in range(num_frames):
        start = i * HOP_LEN
        frame = y[start:start + FRAME_LEN]
        if len(frame) < FRAME_LEN:
            frame = np.pad(frame, (0, FRAME_LEN - len(frame)))
        xw = frame * window
        spectrum = np.abs(np.fft.rfft(xw))
        S_db[:, i] = 20.0 * np.log10(np.maximum(spectrum, 1e-12))

    return y, S_db, volume, num_frames, times, freqs


def smooth_spectrum_blackman(spectrum, window_size=9):
    if window_size % 2 == 0:
        window_size += 1
    window = np.blackman(window_size)
    window /= window.sum()
    return np.convolve(spectrum, window, mode='same')

def near(f1, f2):
    return abs(f1 - f2) < F0_RELATIVE_TOLERANCE * f2

def safe_mean(x):
    if x is None or len(x) == 0:
        return 0.0
    return float(np.mean(x))

def is_flat(spectrum, threshold = 25):
    return np.max(spectrum) - np.median(spectrum) < threshold

'''
pitch_to_freqs = {
    "A2": 110.00, "Bb2": 116.54, "B2": 123.47, "C3": 130.81, "Db3": 138.59,
    "D3": 146.83, "Eb3": 155.56, "E3": 164.81, "F3": 174.61, "Gb3": 185.00,
    "G3": 196.00, "Ab3": 207.65, "A3": 220.00, "Bb3": 233.08, "B3": 246.94,
    "C4": 261.63, "Db4": 277.18, "D4": 293.66, "Eb4": 311.13, "E4": 329.63,
    "F4": 349.23, "Gb4": 369.99, "G4": 392.00, "Ab4": 415.30, "A4": 440.00,
    "Bb4": 466.16, "B4": 493.88, "C5": 523.25, "Db5": 554.37, "D5": 587.33,
    "Eb5": 622.25, "E5": 659.25, "F5": 698.46, "Gb5": 739.99, "G5": 783.99,
    "Ab5": 830.61, "A5": 880.00
}
'''