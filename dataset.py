"""
Use tf Dataset API to process audio so that we can shuffle, prefetch, etc.
"""
import tensorflow as tf
import os
import numpy as np
from cfg import *


def get_data_from_filename(wav_path):
    # Process audio
    raw_data = tf.io.read_file(wav_path)
    wav_data, fs = tf.audio.decode_wav(raw_data) 
    wav_data = tf.squeeze(wav_data)
    wav_data = tf.slice(wav_data, begin=[0], size=[(tf.math.floordiv(tf.size(wav_data), SAMPLE_SIZE)) * SAMPLE_SIZE])

    # FIXME Need to resample to 44100 hz!! Right now we're getting garbage

    wav_data_split = tf.reshape(wav_data, [tf.size(wav_data) / SAMPLE_SIZE, SAMPLE_SIZE])
    fft_raw = tf.map_fn(fn=tf.signal.fft, elems=tf.cast(wav_data_split, tf.complex64))
    fft_float = tf.map_fn(fn=tf.abs, elems=fft_raw, dtype=tf.float32)
    fft = tf.slice(fft_float, begin=[0, 0], size=[-1, tf.shape(fft_float)[1] // 2 - 1])
    fft = tf.expand_dims(fft, 0)

    # Process annotations
    wav_filename = tf.strings.split(wav_path, '/')[-1]
    annotation_path = numpy_annotations_root + '/' + wav_filename + ".npy"
    annotations = tf.py_function(lambda x: np.load(x.numpy()), inp=[annotation_path], Tout=tf.int64)

    return fft, annotations
