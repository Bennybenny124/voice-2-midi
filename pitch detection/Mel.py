import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# ==== 參數 ====
wav_path = "test files/maaale.wav"
frame_length = 2048
hop_length = 64
n_mels = 128
fmin = 100
fmax = 10000

# ==== 載入音訊 ====
y, sr = librosa.load(wav_path, sr=None)

# ==== 梅爾頻譜 ====
S_mel = librosa.feature.melspectrogram(
    y=y, sr=sr,
    n_fft=frame_length,
    hop_length=hop_length,
    n_mels=n_mels,
    fmin=fmin,
    fmax=fmax,
    power=2.0
)

# 轉成 dB
S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

# ==== 畫 2D 梅爾頻譜 ====
plt.figure(figsize=(10, 6))
librosa.display.specshow(
    S_mel_db,
    sr=sr,
    hop_length=hop_length,
    x_axis='time',
    y_axis='mel',
    fmin=fmin,
    fmax=fmax,
    cmap='magma'
)
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram (2D)")
plt.tight_layout()
plt.show()
