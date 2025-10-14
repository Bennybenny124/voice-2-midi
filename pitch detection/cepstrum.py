# cepstrum_f0.py
import numpy as np
import librosa
import matplotlib.pyplot as plt
from config import *  # 取用 SR, FRAME_LEN, HOP_LEN, FMIN, FMAX 與 load()

# 載入資料（與 stft.py 相同介面與時間軸）
y, S_db, num_frames, times, freqs = load()

# 用相同的 frame 設定計算音量門檻（跟 stft.py 一樣邏輯）
rms = librosa.feature.rms(y=y, frame_length=FRAME_LEN, hop_length=HOP_LEN, center=True)[0]
volume = librosa.amplitude_to_db(rms, ref=np.max)

# 把 dB 轉回線性幅度（相對尺度就夠做 cepstrum）
S_mag = librosa.db_to_amplitude(S_db, ref=1.0) + 1e-12  # 防 log(0)

# 目標：在 quefrency q ∈ [1/FMAX, 1/FMIN] 內找最大峰
qmin = int(np.floor(SR / FMAX))  # 以「樣本」為單位的最小週期
qmax = int(np.ceil(SR / FMIN))   # 以「樣本」為單位的最大週期
n_fft = FRAME_LEN

def parabolic_interpolation(x_vals, y_vals):
    """
    對三點 (x-1, x, x+1) 做二次曲線插值，回傳頂點 x_peak
    """
    x0, x1, x2 = x_vals
    y0, y1, y2 = y_vals
    # 以離散點做通用二次插值的頂點位置（避免直接解係數）
    denom = (y0 - 2*y1 + y2)
    if denom == 0:
        return x1
    delta = 0.5 * (y0 - y2) / denom
    return x1 + delta  # 可能是非整數樣本位置

# 用倒頻譜估計 f0
f0_by_cep = np.zeros(num_frames, dtype=np.float32)

for frame in range(num_frames):
    # 取該 frame 的線性幅度頻譜（與 STFT 同步）
    mag = S_mag[:, frame]

    # real cepstrum：c = irfft(log|X(ω)|)
    c = np.fft.irfft(np.log(mag), n=n_fft)

    # 音量太小就直接設 0（跟 stft.py 類似的門檻）
    if volume[frame] < -30:
        f0_by_cep[frame] = 0.0
        continue

    # 在可能的基音週期區間找最大峰（避免 0 附近的包絡）
    q_slice = c[qmin:qmax]
    if q_slice.size < 3:
        f0_by_cep[frame] = 0.0
        continue

    # 取最大點索引（相對於 qmin）
    k_rel = int(np.argmax(q_slice))
    k = k_rel + qmin

    # 邊界保護，做三點拋物線插值改善解析度
    if k <= 0 or k >= len(c) - 1:
        q_peak = float(k)
    else:
        x_vals = np.array([k - 1, k, k + 1], dtype=np.float32)
        y_vals = c[k - 1:k + 2]
        q_peak = parabolic_interpolation(x_vals, y_vals)

    # quefrency(樣本) → 週期(秒) → 頻率(Hz)
    period_sec = q_peak / SR
    if period_sec <= 0:
        f0 = 0.0
    else:
        f0 = 1.0 / period_sec

    # 超出合理範圍就歸零
    if f0 < FMIN or f0 > FMAX:
        f0 = 0.0

    f0_by_cep[frame] = f0

# 視覺化
plt.figure(figsize=(10, 4))
plt.plot(times, f0_by_cep, label='Cepstrum Estimated Pitch (Hz)', alpha=0.85)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Pitch Estimation via Real Cepstrum")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.ylim(FMIN * 0.8, FMAX * 1.1)
plt.show()
