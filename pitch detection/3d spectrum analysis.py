import librosa
import numpy as np
import matplotlib.pyplot as plt

# ==== 參數 ====
wav_path = "pitch detection/0812.wav"
frame_length = 2048
hop_length = 256
min_freq = 100
max_freq = 600

# ==== 載入音訊 ====
y, sr = librosa.load(wav_path, sr=None)

# ==== STFT ====
S = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

# ==== 頻率與時間 ====
freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)

# ==== 限制到 max_freq ====
freq_mask = freqs <= max_freq
freqs = freqs[freq_mask]
S_db = S_db[freq_mask, :]

# ==== 建立網格 ====
T, F = np.meshgrid(times, freqs)

# ==== 畫 3D 圖 ====
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# rstride, cstride 控制解析度（可調整以加速渲染）
ax.plot_surface(T, F, S_db, cmap='magma')

ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_zlabel("Amplitude (dB)")

ax.set_title("3D Spectrogram (Linear Frequency, up to 10 kHz)")
plt.tight_layout()
plt.show()
