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

# Import functions from your existing code
from test import modify_frame_rate, apply_convolution_reverb, save_audio, plot_waveform_from_file

# Title and description
st.title("Slowed Reverb Generator")
st.write("Upload a WAV file and apply slowed reverb effects")

# File uploader for audio only
uploaded_file = st.file_uploader("Upload a WAV file", type="wav")

# Parameters
st.sidebar.header("Effect Parameters")
speed_factor = st.sidebar.slider("Speed Factor", 0.5, 1.0, 0.8, 0.01, help="Lower values make the audio slower")

# Path to the default IR file
ir_path = "ir.wav"  # This should be in the same folder as your app

if uploaded_file is not None:
    st.write("Audio file uploaded successfully!")
    
    # Create a temporary file path for the uploaded file
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_input.write(uploaded_file.getvalue())
    temp_input_path = temp_input.name
    temp_input.close()
    
    # Display audio player for the original file
    st.subheader("Original Audio")
    st.audio(uploaded_file)
    
    # Process button
    if st.button("Apply Slowed Reverb Effect"):
        with st.spinner("Processing audio..."):
            # Load the audio file
            input_file = AudioSegment.from_file(file=temp_input_path, format="wav")
            
            # Modify frame rate (slow down the audio)
            temp_slowed = modify_frame_rate(input_file, speed_factor)
            
            # Apply convolution reverb using the default IR file
            processed_audio, sample_rate = apply_convolution_reverb(temp_slowed, ir_path)
            
            # Save the final output
            output_path = "processed_output.wav"
            save_audio(processed_audio, sample_rate, output_path)
            
            # Plot the waveform of the output file
            plt.figure(figsize=(10, 4))
            plot_image_path = "output_waveform.png"
            plot_waveform_from_file(output_path, plot_image_path)
            
            # Display results
            st.subheader("Slowed + Reverb Audio")
            st.audio(output_path)
            
            st.subheader("Waveform Visualization")
            st.image(plot_image_path)
            
            # Download button for the processed file
            with open(output_path, "rb") as file:
                btn = st.download_button(
                    label="Download Slowed Audio",
                    data=file,
                    file_name="slowed_reverb_audio.wav",
                    mime="audio/wav"
                )
            
            # Clean up temp files
            os.remove(temp_input_path)
            os.remove(temp_slowed)
            # We keep output_path and plot_image_path as they're used for display
            
    st.info("Note: For best results, use high-quality WAV files.")

else:
    st.info("Please upload a WAV file to get started.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Slowed Reverb Generator v1.0")
