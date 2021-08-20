"""
Parse the json beat annotation records and convert to tfrecord
"""
from pathlib import Path
import os
from scipy.io import wavfile
import numpy as np
import json
import tensorflow as tf
from cfg import *

# FIXME Need to refactor into proper functions and stuff
Path(numpy_annotations_root).mkdir(exist_ok=True, parents=True)

def json_to_output_vector(json_file_path, wav_file_path):
    fs, audio_data = wavfile.read(wav_file_path)

    # Number of samples after resampling to required SAMPLE_RATE
    num_samples = int(SAMPLE_RATE / fs * len(audio_data))

    # Input vector length
    num_fourier_samples = int(num_samples / SAMPLE_SIZE)

    # Ground truth needs same length as input, since we're doing seq2seq
    output_annotations = np.zeros(num_fourier_samples, dtype=np.uint8)

    with open(json_file_path) as f:
        json_data = json.load(f)
        
        # There should also be a downbeat annotation type
        # TODO Refactor to use filter
        for annotation in json_data['annotations']:
            if annotation['sandbox']['annotation_type'] == 'beat':
                for datapoint in annotation['data']:
                    seconds_from_start = float(datapoint['time'])
                    samples_from_start = seconds_from_start * SAMPLE_RATE
                    fourier_samples_from_start = int(samples_from_start / SAMPLE_SIZE)
                    if fourier_samples_from_start < len(output_annotations):
                        output_annotations[fourier_samples_from_start] = 1

                return output_annotations

    raise("We shouldn't have reached this point")

def update_metadata(metadata, beat_annotation_array):
    metadata['max_sequence_length'] = \
        max(metadata['max_sequence_length'], len(beat_annotation_array))

    classes, current_class_counts = np.unique(beat_annotation_array, return_counts=True)
    metadata['class_count'][0] += int(current_class_counts[0])
    metadata['class_count'][1] += int(current_class_counts[1])

metadata = { "max_sequence_length": 0, "class_count": {0: 0, 1: 0} }

for root, dirs, files in os.walk(wav_data_root):
    for f in files:
        wav_file_path = os.path.join(root, f)
        json_file_path = os.path.join(json_annotations_root, f + ".jams")
        annotation_path = os.path.join(numpy_annotations_root, f + ".npy")

        beat_annotation_array = json_to_output_vector(json_file_path, wav_file_path)
        update_metadata(metadata, beat_annotation_array)
        np.save(annotation_path, beat_annotation_array) 

        print(wav_file_path)

# TODO Put schema into cfg.py
with open(dataset_metadata_path, 'w') as f:
    json.dump(metadata, f)
