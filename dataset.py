"""
Use tf Dataset API to process audio so that we can shuffle, prefetch, etc.
"""
import tensorflow as tf
import tensorflow_io as tfio
import os
import numpy as np
import cfg
import json


def get_data_from_filename(wav_path):
    # Process audio
    raw_data = tf.io.read_file(wav_path)
    wav_data, fs = tf.audio.decode_wav(raw_data) 
    wav_data = tfio.audio.resample(wav_data, tf.cast(fs, tf.int64), cfg.SAMPLE_RATE)
    wav_data = tf.squeeze(wav_data)
    wav_data = tf.slice(wav_data, begin=[0], size=[(tf.math.floordiv(tf.size(wav_data), cfg.SAMPLE_SIZE)) * cfg.SAMPLE_SIZE])

    wav_data_split = tf.reshape(wav_data, [tf.size(wav_data) / cfg.SAMPLE_SIZE, cfg.SAMPLE_SIZE])
    fft_raw = tf.map_fn(fn=tf.signal.fft, elems=tf.cast(wav_data_split, tf.complex64))
    fft_float = tf.map_fn(fn=tf.abs, elems=fft_raw, dtype=tf.float32)
    fft = tf.slice(fft_float, begin=[0, 0], size=[-1, tf.shape(fft_float)[1] // 2 - 1])
    # TODO Look into whether to normalize fft here

    # Process annotations
    wav_filename = tf.strings.split(wav_path, '/')[-1]
    annotation_path = cfg.numpy_annotations_root + '/' + wav_filename + ".npy"
    annotations = tf.py_function(lambda x: tf.convert_to_tensor(np.load(x.numpy())), inp=[annotation_path], Tout=tf.uint8)

    return fft, annotations

# Add zero padding to ensure input vector always has constant size
def add_padding(fft, annotations):
    # Get metadata
    with open(cfg.dataset_metadata_path) as f:
        metadata = json.load(f)             # TODO If called more than once, use cache decorator
    max_sequence_length = metadata['max_sequence_length']

    fft = tf.pad(fft, [[0, max_sequence_length - tf.shape(fft)[0]], [0,0]])
    annotations = tf.pad(annotations, [[0, max_sequence_length - tf.shape(annotations)[0]]])

    return fft, annotations
