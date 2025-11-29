import librosa
import numpy as np
import struct
import os

def load_reference():
    try:
        with open("calibration/73dB.bin", "rb") as f:
            dB_73 = struct.unpack("f", f.read(4))[0]
    except:
        dB_73 = main()

    return dB_73

def main():
    ref_wav = "calibration/73dB.wav"                 
    out_bin  = "calibration/73dB.bin"

    y_ref, _ = librosa.load(ref_wav, sr=None, mono=True)
    rms = np.sqrt(np.mean(y_ref**2))

    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(struct.pack("f", float(rms)))

    print("rms =", rms)
    print("saved to", out_bin)

    return rms

if __name__ == "__main__":
    main()