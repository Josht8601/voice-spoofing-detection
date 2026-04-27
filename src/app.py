import streamlit as st
import torch
import numpy as np
import librosa

from audiorecorder import audiorecorder

from baseline_cnn_model import SimpleCNN
from preprocess import preprocess_waveform, waveform_to_mel_spectrogram


# -----------------------------
# 🔥 IMPORTANT: Match training width
# -----------------------------
TARGET_WIDTH = 126


# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = SimpleCNN().to(device)

    # 🔥 Initialize LazyLinear with correct shape
    dummy = torch.randn(1, 1, 128, TARGET_WIDTH)
    _ = model(dummy)

    # Load weights
    model.load_state_dict(
        torch.load("../models/baseline_cnn.pt", map_location=device)
    )

    model.eval()
    return model


model = load_model()


# -----------------------------
# Helper: Prepare input tensor
# -----------------------------
def prepare_input(waveform, sr):
    # Resample to 16kHz
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

    # Normalize
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-6)

    # Preprocess
    waveform = preprocess_waveform(waveform)
    spec = waveform_to_mel_spectrogram(waveform)

    # 🔥 Match training shape EXACTLY
    if spec.shape[1] > TARGET_WIDTH:
        spec = spec[:, :TARGET_WIDTH]
    else:
        pad = TARGET_WIDTH - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad)))

    # Convert to tensor
    spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return spec


# -----------------------------
# UI
# -----------------------------
st.title("🎤 Voice Spoofing Detection")
st.write("Upload audio or record your voice.")

label_map = {0: "Real (Bonafide)", 1: "Spoof"}


# -----------------------------
# FILE UPLOAD
# -----------------------------
st.subheader("📂 Upload Audio")

uploaded_file = st.file_uploader("Choose file", type=["wav", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    waveform, sr = librosa.load(uploaded_file, sr=None)
    spec = prepare_input(waveform, sr)

    with torch.no_grad():
        output = model(spec)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    st.subheader("Prediction")
    st.write(label_map[pred])

    st.write(f"Real: {probs[0][0]:.4f}")
    st.write(f"Spoof: {probs[0][1]:.4f}")


# -----------------------------
# MICROPHONE RECORDING
# -----------------------------
st.subheader("🎤 Record Audio")

audio = audiorecorder("Click to record", "Recording...")

if len(audio) > 0:
    st.audio(audio.export().read())

    waveform = np.array(audio.get_array_of_samples()).astype(np.float32)
    sr = audio.frame_rate

    spec = prepare_input(waveform, sr)

    with torch.no_grad():
        output = model(spec)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    st.subheader("Prediction")
    st.write(label_map[pred])

    st.write(f"Real: {probs[0][0]:.4f}")
    st.write(f"Spoof: {probs[0][1]:.4f}")