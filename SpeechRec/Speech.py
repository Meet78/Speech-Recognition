import streamlit as st
import tempfile
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ========== Load model ========== #
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("wav2vec2-finetuned")
    model = Wav2Vec2ForCTC.from_pretrained("wav2vec2-finetuned").to("cpu")
    return processor, model

# ========== Load audio file and resample ========== #
def load_audio(file_path, target_sr=16000):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy()

# ========== Transcribe ========== #
def transcribe(audio_path, processor, model):
    audio = load_audio(audio_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# ========== Dynamic Theme Styling ========== #
st.markdown("""
    <style>
        :root {
            --light-bg: #f9f9f9;
            --light-text: #111111;
            --dark-bg: #0e1117;
            --dark-text: #ffffff;
            --accent-color: #3b82f6;
            --border-radius: 10px;
        }

        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }

        @media (prefers-color-scheme: light) {
            body {
                background-color: var(--light-bg);
                color: var(--light-text);
            }
        }

        @media (prefers-color-scheme: dark) {
            body {
                background-color: var(--dark-bg);
                color: var(--dark-text);
            }
        }

        .block-container {
            padding: 2rem 1.5rem;
        }

        h1, h2, h3, h4 {
            color: var(--accent-color);
        }

        .stButton > button {
            background-color: var(--accent-color);
            color: white;
            border-radius: var(--border-radius);
            padding: 0.5rem 1rem;
            font-size: 1rem;
            border: none;
        }

        .stFileUploader, .stAudio {
            background-color: rgba(255, 255, 255, 0.03);
            border-radius: var(--border-radius);
        }

        .stMarkdown {
            font-size: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# ========== Streamlit App UI ========== #
st.title("🎙️ Wav2Vec2 Live Speech-to-Text")

st.markdown("Upload a short `.wav` audio file to transcribe")

audio_file = st.file_uploader("📤 Upload your WAV file", type=["wav"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_file.read())
        tmp_path = tmpfile.name

    st.audio(tmp_path, format="audio/wav")

    with st.spinner("🔄 Loading model..."):
        processor, model = load_model()

    if st.button("🔍 Transcribe"):
        with st.spinner("🧠 Transcribing..."):
            result = transcribe(tmp_path, processor, model)
        st.success("✅ Transcription complete!")
        st.markdown(f"**Transcript:** `{result}`")
