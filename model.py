import tensorflow as tf
from tcn import TCN
import cfg
import json
from dataset import get_data_from_filename, add_padding
import numpy as np
import sys

# Get dataset metadata
with open(cfg.dataset_metadata_path) as f:
    metadata = json.load(f)

max_sequence_length = metadata['max_sequence_length']
spectrum_size = cfg.SAMPLE_SIZE//2 - 1

# Get dataset
dataset = tf.data.Dataset.list_files(cfg.wav_data_root + "*/*", shuffle=True)
dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
dataset = dataset.map(get_data_from_filename)
dataset = dataset.map(add_padding)

# Model
baseline_model = tf.keras.Sequential([
    # Calculate spectral features
    tf.keras.layers.Reshape((max_sequence_length, spectrum_size, 1),
        input_shape=(max_sequence_length, spectrum_size)),
    tf.keras.layers.Conv2D(10, kernel_size=(3,3), padding='same',
        input_shape=(max_sequence_length, spectrum_size), activation='relu'),
    tf.keras.layers.Conv2D(10, kernel_size=(3,3), padding='same',
        input_shape=(max_sequence_length, spectrum_size), activation='relu'),
    tf.keras.layers.Conv2D(1, kernel_size=(3,3), padding='same',
        input_shape=(max_sequence_length, spectrum_size), activation='relu'),

    # Exploit temporal relationships
    tf.keras.layers.Reshape((max_sequence_length, spectrum_size)),
    TCN(nb_filters=128, input_shape=(max_sequence_length, spectrum_size),
        dilations=(1, 2, 4, 8, 16, 32, 64, 128), kernel_size=2, return_sequences=True),

    # Get an output sequence vector
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
    tf.keras.layers.Reshape((max_sequence_length,))
])

# Weight classes, since dataset is heavily skewed towards negatives
neg = metadata['class_count']['0']
pos = metadata['class_count']['1']
pos_weight = neg / pos

# Train
baseline_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['categorical_accuracy'],
        loss = lambda truth, pred : 
            tf.nn.weighted_cross_entropy_with_logits(tf.cast(truth, tf.float32), pred, pos_weight)
        )
baseline_model.fit(dataset, epochs=1)
