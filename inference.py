"""
Test the model on an actual song
"""
from model import get_model_v1
import json
import cfg
import os
import numpy as np
import wave
import tensorflow as tf
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import resample
import matplotlib.pyplot as plt

# Helpers
def sigmoid(x):
      return 1 / (1 + np.exp(-x))

# Get dataset metadata
with open(cfg.dataset_metadata_path) as f:
    metadata = json.load(f)

max_sequence_length = metadata['max_sequence_length']
spectrum_size = cfg.SAMPLE_SIZE//2 - 1

# TODO Instead of expect_partial follow this post: 
# https://stackoverflow.com/questions/58289342/tf2-0-translation-model-error-when-restoring-the-saved-model-unresolved-objec
model = get_model_v1(max_sequence_length, spectrum_size)
model.load_weights(cfg.checkpoint_filepath).expect_partial()

# Load up a song
Path(cfg.test_songs_root).mkdir(exist_ok=True, parents=True)
if len(os.listdir(cfg.test_songs_root)) == 0:
    print(f"Add a song to {cfg.test_songs_root} so that we can make a sample inference")
    exit()

for filename in os.listdir(cfg.test_songs_root):
    wav_path = os.path.join(cfg.test_songs_root, filename)
    fs, wav_data = wavfile.read(wav_path)

    # Preprocessing
    wav_data = np.array(wav_data, dtype=np.float32)
    wav_data = np.sum(wav_data, axis=1)         # stereo --> mono
    wav_data = wav_data / np.max(wav_data)      # normalize
    wav_data = resample(wav_data, int(len(wav_data) * cfg.SAMPLE_RATE / fs))
    wav_data = wav_data[:np.size(wav_data) // cfg.SAMPLE_SIZE * cfg.SAMPLE_SIZE]
    wav_data_split = np.reshape(wav_data, [np.size(wav_data) // cfg.SAMPLE_SIZE, cfg.SAMPLE_SIZE])
    fft_raw = np.fft.fft(wav_data_split)
    fft = fft_raw[:,:np.shape(fft_raw)[1] // 2 - 1]
    fft = np.expand_dims(fft, 0)
    
    predictions = model.predict(fft[:,:max_sequence_length])
    #predictions = sigmoid(predictions)

    # Graph predictions
    fig, ax = plt.subplots()
    ax.plot(predictions[0])
    plt.show() 
