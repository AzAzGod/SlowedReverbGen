import numpy as np
from scipy.signal import fftconvolve
from pydub import AudioSegment
import soundfile as sf
import os
import wave
import matplotlib.pyplot as plt
import struct

# Set ffmpeg path
AudioSegment.converter = "C:\\ffmpeg\\ffmpeg.exe"

def get_user_input():
    """Get file paths and processing parameters from user"""
    file_path = input("Enter file path: ")
    new_speed = float(input("Enter new speed (Recommend 0.8): "))
    ir_path = input("Enter impulse response file path (default: ir.wav): ") or "ir.wav"
    output_path = input("Enter output file path (default: modified.wav): ") or "modified.wav"
    
    return file_path, new_speed, ir_path, output_path

def modify_frame_rate(input_file, new_speed):
    """Modify the frame rate of the audio file based on speed factor"""
    # Change playback speed by modifying the frame rate
    new_frame_rate = int(input_file.frame_rate * new_speed)
    modified_file = input_file._spawn(input_file.raw_data, overrides={'frame_rate': new_frame_rate})
    
    # Export the slowed file to a temporary WAV file
    temp_file = "temp.wav"
    modified_file.export(temp_file, format="wav")
    
    return temp_file

def apply_convolution_reverb(audio_path, ir_path):
    """Apply convolution reverb to the audio file using the impulse response"""
    # Load the audio file into a NumPy array
    audio = AudioSegment.from_file(audio_path, format="wav")
    samples = np.array(audio.get_array_of_samples())
    
    # Check if audio is stereo or mono
    audio_is_stereo = audio.channels == 2
    if audio_is_stereo:
        samples = samples.reshape((-1, 2))
    
    # Normalize to float32 in range [-1, 1]
    samples = samples.astype(np.float32) / (2**15)
    
    # Load impulse response file
    ir_audio = AudioSegment.from_file(ir_path, format="wav")
    ir_samples = np.array(ir_audio.get_array_of_samples())
    
    # Check if IR is stereo or mono
    ir_is_stereo = ir_audio.channels == 2
    if ir_is_stereo:
        ir_samples = ir_samples.reshape((-1, 2))
    
    ir_samples = ir_samples.astype(np.float32) / (2**15)
    
    # Apply convolution reverb with adaptation for mono/stereo combinations
    if not audio_is_stereo and not ir_is_stereo:
        # Both mono - simple case
        convolved = fftconvolve(samples, ir_samples, mode='full')
    elif audio_is_stereo and ir_is_stereo:
        # Both stereo - process each channel
        convolved = np.zeros((samples.shape[0] + ir_samples.shape[0] - 1, 2), dtype=np.float32)
        convolved[:, 0] = fftconvolve(samples[:, 0], ir_samples[:, 0], mode='full')
        convolved[:, 1] = fftconvolve(samples[:, 1], ir_samples[:, 1], mode='full')
    elif audio_is_stereo and not ir_is_stereo:
        # Stereo audio, mono IR
        convolved = np.zeros((samples.shape[0] + ir_samples.shape[0] - 1, 2), dtype=np.float32)
        convolved[:, 0] = fftconvolve(samples[:, 0], ir_samples, mode='full')
        convolved[:, 1] = fftconvolve(samples[:, 1], ir_samples, mode='full')
    else:
        # Mono audio, stereo IR - use the average of IR channels
        ir_mono = np.mean(ir_samples, axis=1)
        convolved = fftconvolve(samples, ir_mono, mode='full')
    
    # Normalize the output to prevent clipping
    max_val = np.max(np.abs(convolved))
    if max_val > 0:
        convolved = convolved / max_val * 0.9  # Leave a little headroom
    
    # Convert back to int16
    convolved_int16 = np.int16(convolved * (2**15))
    
    return convolved_int16, audio.frame_rate

def save_audio(audio_data, sample_rate, output_path):
    """Save the audio data to a file"""
    sf.write(output_path, audio_data, sample_rate)
    return output_path

def plot_waveform_from_file(wav_file, image_path):
    # Open the WAV file
    with wave.open(wav_file, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
        
        # Determine the correct format for unpacking
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
    
    # Create a time axis in seconds
    duration = samples.shape[0] / framerate
    time = np.linspace(0, duration, num=samples.shape[0])
    
    # Plot the waveform
    plt.figure(figsize=(12, 6))
    if n_channels == 1:
        plt.plot(time, samples, label="Mono")
    else:
        for ch in range(n_channels):
            plt.plot(time, samples[:, ch], label=f"Channel {ch+1}")
    
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.legend()
    plt.tight_layout()
    
    # Save the plot to the provided image path
    plt.savefig(image_path)
    plt.close()

def main():
    # Get user input
    file_path, new_speed, ir_path, output_path = get_user_input()
    
    # Load the input file
    input_file = AudioSegment.from_file(file=file_path, format="wav")
    
    # Modify frame rate
    temp_file = modify_frame_rate(input_file, new_speed)
    
    # Apply convolution reverb
    processed_audio, sample_rate = apply_convolution_reverb(temp_file, ir_path)
    
    # Save the final output
    save_audio(processed_audio, sample_rate, output_path)
    
    # Plot the waveform of the output file
    # plot_waveform_from_file(output_path, "output_waveform.png")

    # Clean up the temporary file
    os.remove(temp_file)
    print(f"Processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    main()
