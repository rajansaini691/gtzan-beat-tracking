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
dataset = dataset.map(get_data_from_filename, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.map(add_padding, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(2)      # Can be higher on GPU
dataset = dataset.cache()
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Model
baseline_model = tf.keras.Sequential([
    # Calculate spectral features
    tf.keras.layers.Reshape((max_sequence_length, spectrum_size, 1),
        input_shape=(max_sequence_length, spectrum_size)),
    tf.keras.layers.Conv2D(16, kernel_size=(3,3),
        input_shape=(max_sequence_length, spectrum_size), activation='relu',
        padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=(1,3)),
    tf.keras.layers.Conv2D(16, kernel_size=(3,3),
        input_shape=(max_sequence_length, spectrum_size), activation='relu',
        padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=(1,3)),
    tf.keras.layers.Conv2D(16, kernel_size=(3,3),
        input_shape=(max_sequence_length, spectrum_size), activation='relu',
        padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=(1,3)),
    tf.keras.layers.Conv2D(16, kernel_size=(3,3),
        input_shape=(max_sequence_length, spectrum_size), activation='relu',
        padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=(1,3)),

    # Exploit temporal relationships
    tf.keras.layers.Reshape((max_sequence_length, 6*16)),     # FIXME Don't hardcode
    TCN(nb_filters=16, dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        kernel_size=2, return_sequences=True),

    # Get an output sequence vector
    tf.keras.layers.Dense(1),
])

# Weight classes, since dataset is heavily skewed towards negatives
neg = metadata['class_count']['0']
pos = metadata['class_count']['1']
pos_weight = neg / pos * 2


def cross_entropy_loss(truth, pred):
    truth = tf.cast(truth, tf.float32)
    return tf.nn.weighted_cross_entropy_with_logits(truth, pred, pos_weight)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=cfg.tensorboard_dir, histogram_freq=1)
    
# Train
baseline_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss = cross_entropy_loss)
baseline_model.fit(dataset, epochs=30, callbacks=[tensorboard_callback])

# Make a sample prediction
np.set_printoptions(threshold=sys.maxsize)
for x, y in dataset:
    y = tf.cast(y, tf.float32)
    y = tf.reshape(y, (max_sequence_length,))
    logits = baseline_model.predict(x)
    logits = tf.reshape(logits, (max_sequence_length,))
    pred = tf.math.sigmoid(logits)

    print(tf.stack([pred, y]))
    raise
