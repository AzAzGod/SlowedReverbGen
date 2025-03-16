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

# Import your existing helper functions
from test import modify_frame_rate, apply_convolution_reverb, save_audio, plot_waveform_from_file

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(
    page_title="Slowed Reverb Generator",
    page_icon="",  # remove any emoji or icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- CUSTOM CSS -------------------------
def local_css():
    st.markdown("""
    <style>
    /* Dark purple/blue gradient for the entire app */
    .stApp {
        background: linear-gradient(135deg, #241b52 0%, #1e2156 100%);
        color: #ffffff; /* White text for contrast */
        position: relative;
        min-height: 100vh;
    }
    
    /* Twinkling starfield overlay */
    .starfield {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none; /* So you can still interact with the UI */
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
        opacity: 0.3; /* Slightly dim for subtle look */
    }
    @keyframes starfield {
        0% { transform: translate(0,0); }
        100% { transform: translate(50px,50px); }
    }

    /* General text styling */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #ffffff;
    }

    /* Title container styling */
    .title-container {
        background: linear-gradient(90deg, #8e2de2, #4a00e0);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.19), 0 6px 6px rgba(0,0,0,0.23);
    }

    /* Button styling */
    .button-container {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
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

    /* File uploader styling */
    .uploadedFileData {
        background-color: rgba(74, 0, 224, 0.1);
        border-radius: 10px;
        padding: 1rem;
    }

    /* Audio player styling */
    audio {
        width: 100%;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.1);
    }

    /* Slider styling */
    .stSlider > div > div {
        color: #ffffff !important;
    }
    .stSlider > div > div > span > div[data-baseweb="slider"] > div {
        background-color: #ffffff !important; /* White track */
    }
    .stSlider > div > div > span > div[data-baseweb="slider"] > div > div {
        background-color: #8e2de2 !important; /* Purple thumb */
    }

    /* Wave animation is hidden by default; we show it manually */
    .wave-animation {
        display: none;
    }
    /* Loading animation frames */
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

# Add the starfield overlay
st.markdown('<div class="starfield"></div>', unsafe_allow_html=True)

# Function returning the wave animation HTML (we'll inject it dynamically)
def get_wave_animation_html():
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

# Title
st.markdown('''
<div class="title-container">
    <h1>Slowed Reverb Generator</h1>
    <p>Transform your music with the perfect slowed reverb effect</p>
</div>
''', unsafe_allow_html=True)

# Control Panel
st.subheader("Control Panel")
speed_factor = st.slider(
    "Speed Factor",
    0.5, 1.0, 0.8, 0.01,
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

# Upload Audio
st.subheader("Upload Audio")
uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

# Default IR file path
ir_path = "ir.wav"

if uploaded_file is not None:
    st.success("Audio file uploaded successfully!")
    
    # Show the original audio
    st.markdown("### Original Audio")
    st.audio(uploaded_file)
    
    # Save uploaded file to a temp path
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_input.write(uploaded_file.getvalue())
    temp_input_path = temp_input.name
    temp_input.close()
    
    # Process button
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("Generate Slowed Reverb File"):
        # LINE A: Create a container for wave animation
        wave_container = st.empty()
        
        with st.spinner("Processing your audio..."):
            # LINE B: Show wave animation while processing
            wave_container.markdown(get_wave_animation_html(), unsafe_allow_html=True)
            
            time.sleep(1)  # Just so user sees the animation momentarily
            
            # 1. Load the audio
            input_file = AudioSegment.from_file(file=temp_input_path, format="wav")
            # 2. Slow it down
            temp_slowed = modify_frame_rate(input_file, speed_factor)
            # 3. Apply convolution reverb
            processed_audio, sample_rate = apply_convolution_reverb(temp_slowed, ir_path)
            # 4. Save output
            output_path = "processed_output.wav"
            save_audio(processed_audio, sample_rate, output_path)
            
            # 5. Plot waveform
            plot_image_path = "output_waveform.png"
            with wave.open(output_path, 'rb') as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)
                
                # Unpack frames
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
            time_axis = np.linspace(0, duration, num=len(samples) if n_channels == 1 else samples.shape[0])
            
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
        
        # LINE C: Remove the wave animation after processing
        wave_container.empty()
        
        # Display output
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
        
        # Clean up temp files
        os.remove(temp_input_path)
        os.remove(temp_slowed)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Please upload a WAV file to get started.")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #cccccc;">
    <p>Slowed Reverb Generator v1.0 | Authored by Azim Usmanov</p>
</div>
""", unsafe_allow_html=True)
