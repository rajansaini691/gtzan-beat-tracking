import tensorflow as tf
import tensorflow_io as tfio
import os
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy.signal import resample
from tcn import TCN
import numpy as np
import json
from functools import lru_cache
import time
from pathlib import Path

# Number of samples
SAMPLE_SIZE = 1024
SAMPLE_RATE = 44100

wav_data_root = "./data/Data/genres_original/"
json_annotations_root = "./data/GTZAN-Rhythm_v2_ismir2015_lbd/jams"
cache_root = "./data/cache"
Path(cache_root).mkdir(exist_ok=True)


# See https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files
@lru_cache
def process_wav(filename):
    fs, data = wavfile.read(filename) # load the data

    resampling_factor = SAMPLE_RATE / fs
    required_num_samples = resampling_factor * len(data)
    data = resample(data, int(required_num_samples))

    a = np.array(data, dtype='float') # this is a two channel soundtrack, I get the first track
    b=np.array([(ele/2**16.)*2 for ele in a]) # this is 8-bit track, b is now normalized on [-1,1)
    assert(len(data) == len(b))

    b = b[:int(len(b) / SAMPLE_SIZE)*SAMPLE_SIZE]
    c = b.reshape(int(len(b) / SAMPLE_SIZE), SAMPLE_SIZE)

    
    def get_fft(x):
        d = fft(x) # create a list of complex number
        e = int(len(d)/2)  # you only need half of the fft list
        return abs(d[:(e-1)])

    return np.expand_dims(np.array(list(map(get_fft, c))), axis=0)

# Note: num_fourier_samples = num_samples / BUFFER_SIZE
def process_annotations(filename, num_fourier_samples):
    output_annotations = np.zeros(num_fourier_samples)

    with open(filename) as f:
        data = json.load(f)
        # There should also be a downbeat annotation type
        # TODO Refactor to use filter
        for annotation in data['annotations']:
            if annotation['sandbox']['annotation_type'] == 'beat':
                for datapoint in annotation['data']:
                    seconds_from_start = float(datapoint['time'])
                    samples_from_start = seconds_from_start * SAMPLE_RATE
                    fourier_samples_from_start = int(samples_from_start / SAMPLE_SIZE)
                    if fourier_samples_from_start < len(output_annotations):
                        output_annotations[fourier_samples_from_start] = 1

                return output_annotations

    assert(False), "We shouldn't have reached this point"
    return None


def gen_filenames(wav_data_root, json_annotations_root, cache_root):
    for root, dirs, files in os.walk(wav_data_root):
        for f in files:
            wav_file_path = os.path.join(root, f)
            json_file_path = os.path.join(json_annotations_root, f + ".jams")
            cached_fourier_file_path = os.path.join(cache_root, f + ".fourier.npy")
            cached_beats_file_path = os.path.join(cache_root, f + ".beats.npy")

            assert(os.path.isfile(wav_file_path))
            assert(os.path.isfile(json_file_path))

            try:
                if os.path.isfile(cached_fourier_file_path) and os.path.isfile(cached_beats_file_path):
                    fourier = np.load(cached_fourier_file_path)
                    beats = np.load(cached_beats_file_path)
                else:
                    fourier = process_wav(wav_file_path)
                    beats = process_annotations(json_file_path, fourier.shape[1])

                    np.save(cached_fourier_file_path, fourier)
                    np.save(cached_beats_file_path, beats)

                yield fourier, beats
            except Exception:
                continue

dataset = tf.data.Dataset.from_generator(
        lambda: gen_filenames(wav_data_root, json_annotations_root, cache_root),
        output_signature=(
            tf.TensorSpec(shape=(None, None, SAMPLE_SIZE//2 - 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None), dtype=tf.int8)))

baseline_model = tf.keras.Sequential([
    TCN(input_shape=(None, SAMPLE_SIZE//2 - 1)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

baseline_model.compile(optimizer='adam', loss='binary_crossentropy')
baseline_model.fit(dataset, epochs=10)
