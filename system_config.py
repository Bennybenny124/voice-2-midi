# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
SR              = 48000
CHANNELS        = 1
FMIN            = 100
FMAX            = 600
FRAME_LEN       = 2048
HOP_LEN         = 512
TICK = HOP_LEN / SR
INPUT_DEVICE    = None

CUTOFF_L        = 100
CUTOFF_H        = 3000

MIN_PEAK        = 3
MAX_PEAK        = 7
F0_RELATIVE_TOLERANCE = 0.08
MAX_A           = -1.35e-3
PEAK_SHIFT_TOLERANCE = 8

SILENCE_DB      = 40

# Plotting
TIME_WINDOW_SEC = 3
ENABLE_PLOT     = True

# Calibration
CALIBRATION_FILE = "calibration/73dB.bin"

# Window
WINDOW_FN_NAME  = "hann"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
import numpy as np

def smooth_spectrum_blackman(spectrum, window_size=9):
    if window_size % 2 == 0:
        window_size += 1
    window = np.blackman(window_size)
    window /= window.sum()
    return np.convolve(spectrum, window, mode='same')

def dft(frame, frame_len = FRAME_LEN, window_fn=np.hanning):
    window = window_fn(frame_len)
    xw = frame * window
    return np.abs(np.fft.rfft(xw))

def get_volume(frame, reference, reference_dB):
    x = frame.astype(np.float32) / 32768.0
    rms = np.sqrt(safe_mean(x**2))
    return 20.0 * np.log10(rms / reference) + reference_dB

def near(f1, f2):
    return abs(f1 - f2) < F0_RELATIVE_TOLERANCE * f2

def safe_mean(x):
    if x is None or len(x) == 0:
        return 0.0
    return float(np.mean(x))

def is_flat(spectrum, threshold = 16):
    return np.max(spectrum) - np.median(spectrum) < threshold

def get_window(name: str, n: int):
    name = (name or "").lower()
    if name in ("hann", "hanning"):
        return np.hanning(n).astype(np.float32)
    if name == "hamming":
        return np.hamming(n).astype(np.float32)
    if name == "blackman":
        return np.blackman(n).astype(np.float32)
    return np.ones(n, dtype=np.float32)