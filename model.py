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

# Get dataset
dataset = tf.data.Dataset.list_files(cfg.wav_data_root + "*/*", shuffle=True)
dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
dataset = dataset.map(get_data_from_filename)
dataset = dataset.map(add_padding)

# Model
baseline_model = tf.keras.Sequential([
    TCN(nb_filters=128, input_shape=(None, cfg.SAMPLE_SIZE//2 - 1),
        dilations=(1, 2, 4, 8, 16, 32, 64, 128), kernel_size=2),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Weight classes, since dataset is heavily skewed towards negatives
neg = metadata['class_count']['0']
pos = metadata['class_count']['1']
total = neg + pos
class_weight = {0: (1 / neg) * (total / 2.0), 1: (1 / pos) * (total / 2.0)}

# Train
baseline_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall()])
baseline_model.fit(dataset, epochs=1, class_weight=class_weight)
