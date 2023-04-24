
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample, find_peaks
from scipy.io import wavfile

# Load and preprocess the data
sr, audio = wavfile.read('Datasets/2023-04-20_19-14-01.wav')
# # Resample to 1000 Hz
# orig_num_samples = len(audio)
# new_num_samples = int(orig_num_samples * 1000 / sr)
# audio_resampled = resample(audio, new_num_samples)

# # Save the resampled audio to a new file
# wavfile.write('Datasets/file.wav', 1000, audio_resampled)
# sr, audio = wavfile.read('Datasets/file.wav')
# audio = resample(audio, sr, 1000)  # Resample to 1000 Hz
# Define the filter parameters
nyquist_rate = sr / 2.0
cutoff_freq = 100 / nyquist_rate
b, a = butter(4, cutoff_freq, 'highpass')

audio = filtfilt(b, a, audio)  # High-pass filter


audio = audio / np.max(np.abs(audio))  # Normalize the signal
print(audio)
# Peak detection
peaks, _ = find_peaks(audio, height=0.2, distance=100)

# Respiration rate calculation
deltas = np.diff(peaks)
rr = sr / np.mean(deltas) * 60
print(rr)
# Visualization
fig, ax = plt.subplots()
ax.plot(audio, label='Audio Signal')
ax.plot(peaks, audio[peaks], "x", label='Peak Detection')
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Amplitude')
ax.legend()

fig, ax = plt.subplots()
ax.plot(rr, label='Respiratory Rate')
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Respiration Rate (breaths per minute)')
ax.legend()
plt.show()
