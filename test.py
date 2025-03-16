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

import numpy as np
from pydub import AudioSegment
from scipy.signal import fftconvolve

def apply_convolution_reverb_chunked(audio_path, ir_path, chunk_size=44100):
    """
    Apply convolution reverb to the audio file in chunks using the impulse response (IR).
    This uses a simple overlap-add method. 

    :param audio_path: Path to the main audio WAV.
    :param ir_path: Path to the impulse response WAV.
    :param chunk_size: Number of samples to process per chunk. 
                      Larger chunk_size => fewer chunks, but higher memory usage.
    :return: 
        - convolved_int16: The final convolved audio in int16 numpy array
        - output_frame_rate: The sample rate of the output audio
    """
    # 1) LOAD MAIN AUDIO
    audio = AudioSegment.from_file(audio_path, format="wav")
    frame_rate = audio.frame_rate
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # Reshape if stereo
    channels = audio.channels
    if channels == 2:
        samples = samples.reshape((-1, 2))
    else:
        samples = samples.reshape((-1, 1))

    # Convert int16 -> float32 in [-1, 1]
    samples /= 2**15

    # 2) LOAD IMPULSE RESPONSE
    ir_audio = AudioSegment.from_file(ir_path, format="wav")
    ir_samples = np.array(ir_audio.get_array_of_samples(), dtype=np.float32)

    ir_channels = ir_audio.channels
    if ir_channels == 2:
        ir_samples = ir_samples.reshape((-1, 2))
    else:
        ir_samples = ir_samples.reshape((-1, 1))

    # Also convert IR from int16 -> float32 in [-1, 1]
    ir_samples /= 2**15

    if channels != ir_channels:
        # If your IR is mono but audio is stereo (or vice versa), 
        # you have to decide how to handle that. For simplicity,
        # let's just replicate the IR channels if needed.
        # This ensures shapes match up for stereo convolution.
        if ir_channels == 1 and channels == 2:
            ir_samples = np.column_stack([ir_samples, ir_samples])
            ir_channels = 2
        elif ir_channels == 2 and channels == 1:
            # e.g. if IR is stereo but main audio is mono
            # use only the first IR channel or average them, etc.
            # For simplicity, let's average the stereo IR.
            ir_samples = ir_samples.mean(axis=1, keepdims=True)
            ir_channels = 1

    # 3) SET UP OUTPUT ARRAY
    n_samples = samples.shape[0]
    n_ir = ir_samples.shape[0]
    # Convolution length: N + M - 1
    out_length = n_samples + n_ir - 1
    
    # We'll store float32 in [-1, 1] then convert to int16 at the end
    convolved = np.zeros((out_length, channels), dtype=np.float32)

    # 4) OVERLAP-ADD CHUNKED CONVOLUTION
    # Process each chunk for each channel
    start_idx = 0
    while start_idx < n_samples:
        end_idx = min(start_idx + chunk_size, n_samples)
        
        # Extract chunk
        audio_chunk = samples[start_idx:end_idx, :]
        
        # Convolve channel by channel
        for ch in range(channels):
            chunk_conv = fftconvolve(audio_chunk[:, ch], ir_samples[:, ch], mode='full')
            # Overlap-add into our output buffer
            convolved[start_idx : start_idx + len(chunk_conv), ch] += chunk_conv.astype(np.float32)
        
        # Move to next chunk
        start_idx += chunk_size
        
        # Update progress
        if progress_callback:
            progress_callback(chunk_index, total_chunks)

        chunk_index += 1
        start_idx += chunk_size
        
    # 5) NORMALIZE THE RESULT (avoid clipping)
    max_val = np.max(np.abs(convolved))
    if max_val > 1e-9:  # avoid division by zero
        convolved /= max_val

    # 6) CONVERT BACK TO int16
    convolved_int16 = (convolved * (2**15)).astype(np.int16)

    return convolved_int16, frame_rate


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
