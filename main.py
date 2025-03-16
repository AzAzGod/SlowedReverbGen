import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from pydub import AudioSegment
import soundfile as sf
import os
import wave
import struct
import tempfile
import time
import base64
from pathlib import Path

# Import your helper functions
from test import modify_frame_rate, apply_convolution_reverb, save_audio, plot_waveform_from_file

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(
    page_title="Slowed Reverb Generator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- CUSTOM CSS -------------------------
def local_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #241b52 0%, #1e2156 100%);
        color: #ffffff;
        position: relative;
        min-height: 100vh;
    }
    .starfield {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }
    .starfield::before, .starfield::after {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(#ffffff 1px, rgba(255,255,255,0) 1px);
        background-position: center center;
        background-repeat: repeat;
        background-size: 50px 50px;
        animation: starfield 50s linear infinite;
        opacity: 0.3;
    }
    @keyframes starfield {
        0% { transform: translate(0,0); }
        100% { transform: translate(50px,50px); }
    }
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #ffffff;
    }
    .title-container {
        background: linear-gradient(90deg, #8e2de2, #4a00e0);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.19), 0 6px 6px rgba(0,0,0,0.23);
    }
    .stButton > button {
        border-radius: 50px;
        padding: 0.5rem 2rem;
        background-color: #4a00e0;
        color: white;
        border: none;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #8e2de2;
        transition: background-color 0.3s ease;
    }
    audio {
        width: 100%;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.1);
    }
    .stSlider > div > div {
        color: #ffffff !important;
    }
    .stSlider > div > div > span > div[data-baseweb="slider"] > div {
        background-color: #ffffff !important; 
    }
    .stSlider > div > div > span > div[data-baseweb="slider"] > div > div {
        background-color: #8e2de2 !important; 
    }
    .wave-animation {
        display: none;
    }
    @keyframes wave {
        0% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0); }
    }
    .wave-animation div {
        background-color: #ffffff;
        width: 5px;
        height: 20px;
        margin: 0 3px;
        border-radius: 5px;
        animation: wave 1s infinite;
    }
    .wave-animation div:nth-child(2) {
        animation-delay: 0.1s;
    }
    .wave-animation div:nth-child(3) {
        animation-delay: 0.2s;
    }
    .wave-animation div:nth-child(4) {
        animation-delay: 0.3s;
    }
    .wave-animation div:nth-child(5) {
        animation-delay: 0.4s;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Starfield overlay
st.markdown('<div class="starfield"></div>', unsafe_allow_html=True)

def get_wave_animation_html():
    """Return the HTML snippet for the wave animation."""
    return """
    <div class="wave-animation" style="display: flex; align-items: center; justify-content: center; margin: 20px 0;">
        <div></div>
        <div></div>
        <div></div>
        <div></div>
        <div></div>
    </div>
    """

# ------------------------- LAYOUT -------------------------
st.markdown('''
<div class="title-container">
    <h1>Slowed Reverb Generator</h1>
    <p>Transform your music with the perfect slowed reverb effect</p>
</div>
''', unsafe_allow_html=True)

# Create two columns so the slider is on the left, uploader on the right
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Control Panel")
    speed_factor = st.slider(
        "Speed Factor",
        min_value=0.5, max_value=1.0, value=0.8, step=0.01,
        help="Lower values make the audio slower."
    )
    st.markdown("""
    <h4>How it works:</h4>
    <ol>
        <li>Upload your WAV audio file</li>
        <li>Adjust the speed factor</li>
        <li>Click "Generate Slowed Reverb"</li>
        <li>Listen and download your transformed audio</li>
    </ol>
    """, unsafe_allow_html=True)

with col_right:
    st.subheader("Upload Audio")
    uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

# Path to your default IR file
ir_path = "ir.wav"

if uploaded_file is not None:
    st.success("Audio file uploaded successfully!")
    
    # Show original audio
    st.markdown("### Original Audio")
    st.audio(uploaded_file)
    
    # Temporary file path
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_input.write(uploaded_file.getvalue())
    temp_input_path = temp_input.name
    temp_input.close()
    
    if st.button("Generate Slowed Reverb File"):
        # Create a container for wave animation
        wave_container = st.empty()
        
        with st.spinner("Processing your audio..."):
            # Show wave animation while processing
            wave_container.markdown(get_wave_animation_html(), unsafe_allow_html=True)
            
            time.sleep(1)  # Slight delay so the animation is visible
            
            # 1. Load audio
            input_file = AudioSegment.from_file(file=temp_input_path, format="wav")
            # 2. Slow down
            temp_slowed = modify_frame_rate(input_file, speed_factor)
            # 3. Apply reverb
            processed_audio, sample_rate = apply_convolution_reverb(temp_slowed, ir_path)
            # 4. Save result
            output_path = "processed_output.wav"
            save_audio(processed_audio, sample_rate, output_path)
            
            # 5. Generate waveform plot
            plot_image_path = "output_waveform.png"
            with wave.open(output_path, 'rb') as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)
                
                # Unpack
                if sample_width == 1:
                    fmt = "{}B".format(n_frames * n_channels)
                elif sample_width == 2:
                    fmt = "{}h".format(n_frames * n_channels)
                else:
                    raise ValueError("Unsupported sample width: {}".format(sample_width))
                
                samples = struct.unpack(fmt, frames)
                samples = np.array(samples)
                if n_channels > 1:
                    samples = samples.reshape(-1, n_channels)
            
            duration = (len(samples) / framerate) if n_channels == 1 else (samples.shape[0] / framerate)
            time_axis = np.linspace(0, duration, len(samples) if n_channels == 1 else samples.shape[0])
            
            plt.figure(figsize=(10, 4), facecolor='#121212')
            if n_channels == 1:
                plt.plot(time_axis, samples, color='#8e2de2', alpha=0.7)
                plt.fill_between(time_axis, samples, alpha=0.2, color='#8e2de2')
            else:
                plt.plot(time_axis, samples[:, 0], color='#8e2de2', alpha=0.7)
                plt.plot(time_axis, samples[:, 1], color='#4a00e0', alpha=0.7)
                plt.fill_between(time_axis, samples[:, 0], alpha=0.2, color='#8e2de2')
                plt.fill_between(time_axis, samples[:, 1], alpha=0.2, color='#4a00e0')
            
            plt.grid(alpha=0.1)
            plt.xlabel("Time (s)", color='white')
            plt.ylabel("Amplitude", color='white')
            plt.title("Waveform Visualization", color='white')
            plt.tick_params(colors='white')
            for spine in plt.gca().spines.values():
                spine.set_color('#333333')
            
            plt.tight_layout()
            plt.savefig(plot_image_path, facecolor='#121212')
            plt.close()
        
        # Remove the wave animation now that processing is complete
        wave_container.empty()
        
        st.success("Processing complete! Your slowed reverb track is ready.")
        
        st.markdown("### Processed Audio")
        st.audio(output_path)
        
        st.markdown("### Waveform Visualization")
        st.image(plot_image_path)
        
        # Download button
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Your Slowed Reverb Track",
                data=file,
                file_name="slowed_reverb_audio.wav",
                mime="audio/wav"
            )
        
        # Cleanup
        os.remove(temp_input_path)
        os.remove(temp_slowed)

else:
    st.info("Please upload a WAV file to get started.")

st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #cccccc;">
    <p>Slowed Reverb Generator v1.0 | Authored by Azim Usmanov</p>
</div>
""", unsafe_allow_html=True)
