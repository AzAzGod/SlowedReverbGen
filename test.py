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

def apply_convolution_reverb(audio_path, ir_path, chunk_size=44100, progress_callback=None):
    """
    Apply convolution reverb in chunks, using overlap-add.
    Optionally call `progress_callback(chunk_idx, total_chunks)` after each chunk
    so Streamlit can display progress.
    """
    audio = AudioSegment.from_file(audio_path, format="wav")
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    sample_rate = audio.frame_rate
    audio_channels = audio.channels

    if audio_channels == 2:
        samples = samples.reshape((-1, 2))
    else:
        samples = samples.reshape((-1, 1))

    samples /= 2**15  # int16 -> float32 in [-1,1]

    # Load IR
    ir_audio = AudioSegment.from_file(ir_path, format="wav")
    ir_samples = np.array(ir_audio.get_array_of_samples(), dtype=np.float32)
    ir_channels = ir_audio.channels

    if ir_channels == 2:
        ir_samples = ir_samples.reshape((-1, 2))
    else:
        ir_samples = ir_samples.reshape((-1, 1))

    ir_samples /= 2**15

    n_samples = samples.shape[0]
    n_ir = ir_samples.shape[0]
    out_length = n_samples + n_ir - 1
    convolved = np.zeros((out_length, audio_channels), dtype=np.float32)

    total_chunks = (n_samples // chunk_size) + 1
    start_idx = 0

    for i in range(total_chunks):
        end_idx = min(start_idx + chunk_size, n_samples)
        audio_chunk = samples[start_idx:end_idx, :]

        # Convolve each channel in the chunk
        from scipy.signal import fftconvolve
        for ch in range(audio_channels):
            chunk_conv = fftconvolve(audio_chunk[:, ch], ir_samples[:, ch], mode='full')
            convolved[start_idx : start_idx + len(chunk_conv), ch] += chunk_conv.astype(np.float32)

        # Call the progress callback, if provided
        if progress_callback is not None:
            progress_callback(i + 1, total_chunks)

        start_idx += chunk_size
        if start_idx >= n_samples:
            break

    # Normalize
    max_val = np.max(np.abs(convolved))
    if max_val > 1e-9:
        convolved /= max_val * 1.0  # or 0.9 for headroom

    convolved_int16 = (convolved * 2**15).astype(np.int16)
    return convolved_int16, sample_rate

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
