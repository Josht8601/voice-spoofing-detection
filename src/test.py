import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_path = "../data/LA/ASVspoof2019_LA_train/flac/LA_T_1004644.flac"

waveform, sr = librosa.load(audio_path, sr=None)

spec = librosa.feature.melspectrogram(y=waveform, sr=sr)
spec_db = librosa.power_to_db(spec, ref=np.max)

plt.figure(figsize=(8, 4))
librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram")
plt.tight_layout()
plt.savefig("spectrogram.png")