import tensorflow as tf
import cfg
import os
import json
from dataset import get_data_from_filename, add_padding
from model import get_model_v1
import numpy as np
import sys
from pathlib import Path

# Get dataset metadata
with open(cfg.dataset_metadata_path) as f:
    metadata = json.load(f)

max_sequence_length = metadata['max_sequence_length']
spectrum_size = cfg.SAMPLE_SIZE//2 - 1

# Training config
batch_size = 2
test_dataset_size = 100
checkpoint_dir = './out/ckpt'
checkpoint_filepath = os.path.join(checkpoint_dir, 'checkpoint')
Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)

# Get dataset
dataset = tf.data.Dataset.list_files(cfg.wav_data_root + "*/*", shuffle=True)
dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
dataset = dataset.map(get_data_from_filename, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.map(add_padding, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(batch_size)      # Can be higher on GPU
dataset = dataset.cache()
dataset = dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = dataset.take(test_dataset_size)
train_dataset = dataset.skip(test_dataset_size)

# Model
model = get_model_v1(max_sequence_length, spectrum_size)

# Weight classes, since dataset is heavily skewed towards negatives
neg = metadata['class_count']['0']
pos = metadata['class_count']['1']
pos_weight = neg / pos * 2


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=cfg.tensorboard_dir, histogram_freq=1)
    
# Configure checkpoint and load weights if we can find any
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    mode='min',
    save_weights_only=True,
    save_best_only=True)

if any(fname.startswith('checkpoint') for fname in os.listdir(checkpoint_dir)):
    model.load_weights(checkpoint_filepath)

# Train:
def cross_entropy_loss(truth, pred):
    truth = tf.cast(truth, tf.float32)
    return tf.nn.weighted_cross_entropy_with_logits(truth, pred, pos_weight)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss = cross_entropy_loss)
model.fit(train_dataset, epochs=30, validation_data=train_dataset,
        callbacks=[tensorboard_callback, model_checkpoint_callback])

# Make a sample prediction
np.set_printoptions(threshold=sys.maxsize)
for x, y in dataset:
    y = tf.cast(y, tf.float32)
    y = tf.reshape(y, (batch_size, max_sequence_length))
    logits = model.predict(x)
    logits = tf.reshape(logits, (batch_size, max_sequence_length))
    pred = tf.math.sigmoid(logits).numpy()

    print(tf.stack([pred, y], axis=-1))
    raise
