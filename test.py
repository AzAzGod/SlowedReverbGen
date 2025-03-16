import numpy as np
from scipy.signal import fftconvolve
from pydub import AudioSegment
import soundfile as sf

AudioSegment.converter = "C:\\ffmpeg\\ffmpeg.exe"

# Importing audio file
file_path = input("Enter file path: ")

# For test, delete and use file_path after done
file='2024 prod. ojivolta, earlonthebeat, and Kanye West.wav'

input_file = AudioSegment.from_file(file=file, format="wav")

# Get the slowdown factor from the user
new_speed = float(input("Enter new speed (Recommend 0.8): "))

# Change playback speed by modifying the frame rate
new_frame_rate = int(input_file.frame_rate * new_speed)
modified_file = input_file._spawn(input_file.raw_data, overrides={'frame_rate': new_frame_rate})

# Optionally, for compatibility, you could reset the frame rate:
# modified_file = modified_file.set_frame_rate(input_file.frame_rate)

# Export the slowed file to a temporary WAV file
temp_file = "temp.wav"
modified_file.export(temp_file, format="wav")

# --- Convolution Reverb Implementation ---

# Load the slowed audio file into a NumPy array
audio = AudioSegment.from_file(temp_file, format="wav")
samples = np.array(audio.get_array_of_samples())
if audio.channels == 2:
    samples = samples.reshape((-1, 2))
# Normalize to float32 in range [-1, 1]
samples = samples.astype(np.float32) / (2**15)

# Load your impulse response file (make sure 'ir.wav' exists)
ir_audio = AudioSegment.from_file("ir.wav", format="wav")
ir_samples = np.array(ir_audio.get_array_of_samples())
if ir_audio.channels == 2:
    ir_samples = ir_samples.reshape((-1, 2))
ir_samples = ir_samples.astype(np.float32) / (2**15)

# Apply convolution reverb using fftconvolve
if audio.channels == 1:
    convolved = fftconvolve(samples, ir_samples, mode='full')
else:
    # Process each channel separately for stereo
    convolved = np.zeros((samples.shape[0] + ir_samples.shape[0] - 1, 2), dtype=np.float32)
    convolved[:, 0] = fftconvolve(samples[:, 0], ir_samples[:, 0], mode='full')
    convolved[:, 1] = fftconvolve(samples[:, 1], ir_samples[:, 1], mode='full')

# Normalize the output to prevent clipping
max_val = np.max(np.abs(convolved))
if max_val > 0:
    convolved = convolved / max_val

# Convert back to int16
convolved_int16 = np.int16(convolved * (2**15))

# Save the final output
final_file = "modified.wav"
sf.write(final_file, convolved_int16, audio.frame_rate)

# Optionally, clean up the temporary file
import os
os.remove(temp_file)