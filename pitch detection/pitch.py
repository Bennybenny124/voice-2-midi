# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import uniform_filter1d
# from mpl_toolkits.mplot3d import Axes3D

# # 從 A3 (220Hz) 到 A5 (880Hz) 的音高頻率（半音階）
# pitch_freqs = {
#     "A3": 220.00,
#     "Bb3": 233.08,
#     "B3": 246.94,
#     "C4": 261.63,
#     "Db4": 277.18,
#     "D4": 293.66,
#     "Eb4": 311.13,
#     "E4": 329.63,
#     "F4": 349.23,
#     "Gb4": 369.99,
#     "G4": 392.00,
#     "Ab4": 415.30,
#     "A4": 440.00,
#     "Bb4": 466.16,
#     "B4": 493.88,
#     "C5": 523.25,
#     "Db5": 554.37,
#     "D5": 587.33,
#     "Eb5": 622.25,
#     "E5": 659.25,
#     "F5": 698.46,
#     "Gb5": 739.99,
#     "G5": 783.99,
#     "Ab5": 830.61,
#     "A5": 880.00,
# }

# # 載入音訊
# audio_path = 'pitch.wav'
# y, sr = librosa.load(audio_path, sr=None)

# # 設定參數
# n_fft = 2048  # FFT窗長（通常是2的次方，越長低頻解析度越好）
# hop_length = int(n_fft*0.1) 

# # STFT 計算
# stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
# stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

# # 時間與頻率軸
# times = librosa.frames_to_time(np.arange(stft_db.shape[1]), sr=sr, hop_length=hop_length)
# frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

# # 建立網格
# T, F = np.meshgrid(times, frequencies)
# Z = stft_db

# # 先找出特定音高對應到的 frequency index
# pitch_indices = {}
# for name, freq in pitch_freqs.items():
#     temp = list()
#     for i in range(1, 6):
#         temp.append(np.argmin(np.abs(frequencies - freq)))
#     pitch_indices[name] = temp

# # 每個時間點根據這些頻率裡找出最大者
# inferred_pitches = []
# for t in range(Z.shape[1]):
#     print("time t = " + str(t) + ":")
#     max_pitch = None
#     max_value = -np.inf
#     for name, indices in pitch_indices.items():
#         print("note name: " + name, end="\t| ")
#         val = 0.0
#         for idx in indices:
#             val += Z[idx, t]
#             print(val, end=", ")
#         if val > max_value:
#             max_value = val
#             max_pitch = pitch_freqs[name]
#         print("")
#     inferred_pitches.append(max_pitch)
#     print("\n")

# inferred_pitches = np.array(inferred_pitches)


# # 繪圖
# fig = plt.figure(figsize=(14, 6))

# # (1) 左邊：3D 頻譜
# ax1 = fig.add_subplot(1, 2, 1, projection='3d')
# ax1.plot_surface(T, F, Z, cmap='viridis', alpha=0.9)
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Frequency (Hz)')
# ax1.set_zlabel('Amplitude (dB)')
# ax1.set_title('STFT Spectrogram')

# ax2 = fig.add_subplot(1, 2, 2)
# ax2.plot(times, inferred_pitches, color='blue', linestyle='--', linewidth=1.5, label='Inferred Pitch (Max FFT freq)')
# ax2.grid(True)
# ax2.legend()

# plt.tight_layout()
# plt.show()

import librosa 
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from mpl_toolkits.mplot3d import Axes3D

# 音高設定（CQT 預設就是以半音切分）
pitch_names = [
    "A3", "Bb3", "B3", "C4", "Db4", "D4", "Eb4", "E4", "F4", "Gb4", "G4", "Ab4",
    "A4", "Bb4", "B4", "C5", "Db5", "D5", "Eb5", "E5", "F5", "Gb5", "G5", "Ab5", "A5"
]
pitch_freqs = {note: librosa.note_to_hz(note) for note in pitch_names}

# 載入音訊
audio_path = 'up.wav'
y, sr = librosa.load(audio_path, sr=None)

# CQT 參數
hop_length = int(sr * 0.005)  # 每10ms做一次分析
n_bins = 24                  
bins_per_octave = 12
fmin = librosa.note_to_hz("A3")

# 計算 CQT
cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.min)

# 時間軸與音高軸
times = librosa.frames_to_time(np.arange(cqt_db.shape[1]), sr=sr, hop_length=hop_length)
frequencies = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)

# 找出對應音高與泛音位置（以 bin 為單位）
pitch_indices = {}
for name, freq in pitch_freqs.items():
    base_idx = np.argmin(np.abs(frequencies - freq))
    harmonics = []
    for i in range(1, 4):  # 第1到第4泛音
        harmonic_freq = freq * i
        idx = np.argmin(np.abs(frequencies - harmonic_freq))
        harmonics.append(idx)
    pitch_indices[name] = harmonics

# 推論音高（使用泛音加權）
weights = [1.0, 0.8, 0.2, 0.1]
inferred_pitches = []
# last = 0.0
# last1 = 0.0
# last2 = 0.0
# last3 = 0.0
for t in range(cqt_db.shape[1]):
    max_pitch = None
    max_value = -np.inf
    for name, indices in pitch_indices.items():
        val = 0.0
        for i, idx in enumerate(indices):
            if idx < cqt_db.shape[0]:
                val += np.log(cqt_db[idx, t] + 1e-20)
        if val > max_value:
            max_value = val
            max_pitch = pitch_freqs[name]

    # if last1 == last2 and last2 == last3 and last3 == max_pitch or t == 0:
    #     inferred_pitches.append(max_pitch)
    # else: inferred_pitches.append(inferred_pitches[t-1])
    inferred_pitches.append(max_pitch)

    # last3 = last2
    # last2 = last1
    # last1 = max_pitch

# 中值濾波平滑
inferred_pitches = np.array(inferred_pitches)

# 繪圖
fig = plt.figure(figsize=(14, 6))

# (1) 左邊：3D CQT 頻譜圖
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

# 建立 3D 網格（時間 × 頻率）
T, F = np.meshgrid(times, frequencies)
Z = cqt_db

# 畫 3D surface
ax1.plot_surface(T, F, Z, cmap='viridis', linewidth=0, antialiased=False)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Frequency (Hz)')
ax1.set_zlabel('Amplitude (dB)')
ax1.set_title('3D CQT Spectrogram')


# (2) 右邊：推論基音曲線
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(times, inferred_pitches, color='gray', linewidth=1, label='Raw Inferred')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_title('Inferred Pitch Over Time')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

cqt_mag = np.abs(cqt)

# 用迴圈持續詢問時間點，並畫出對應頻譜橫切面
t = 0.5
while t >= 0:
    try:
        t = float(input("輸入要觀察的時間點（秒，輸入負數結束）："))
        if t < 0:
            break

        # 找出最接近時間點的 frame index
        frame_idx = np.argmin(np.abs(times - t))
        spectrum = cqt_mag[:, frame_idx]

        # 畫圖
        plt.figure(figsize=(8, 4))
        plt.plot(frequencies, 20 * np.log10(spectrum + 1e-10))  # dB scale
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (dB)")
        plt.title(f"Spectrum at t = {t:.2f} s (frame {frame_idx})")
        plt.grid(True)
        plt.show()

    except ValueError:
        print("請輸入有效的數字。")
    
