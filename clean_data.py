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
Path(cache_root).mkdir(exist_ok=True)
Path(numpy_annotations_root).mkdir(exist_ok=True)

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


for root, dirs, files in os.walk(wav_data_root):
    for f in files:
        wav_file_path = os.path.join(root, f)
        json_file_path = os.path.join(json_annotations_root, f + ".jams")
        annotation_path = os.path.join(numpy_annotations_root, f + ".npy")

        beat_annotation_array = json_to_output_vector(json_file_path, wav_file_path)
        beat_annotation_tfrecord_feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=beat_annotation_array))
        feature = { 'beat_locations_44100-hz-1024-samples': beat_annotation_tfrecord_feature }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

        print(wav_file_path)
        with open(annotation_path, 'wb') as f:
            f.write(example_proto.SerializeToString())

        np.save(annotation_path, beat_annotation_array) 
