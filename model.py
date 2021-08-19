import tensorflow as tf
from tcn import TCN
from cfg import *
from dataset import get_data_from_filename

dataset = tf.data.Dataset.list_files(wav_data_root + "*/*", shuffle=True)
dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
dataset = dataset.map(get_data_from_filename)

baseline_model = tf.keras.Sequential([
    TCN(nb_filters=128, input_shape=(None, SAMPLE_SIZE//2 - 1)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

baseline_model.compile(optimizer='sgd', loss='binary_crossentropy')
baseline_model.fit(dataset, epochs=10)
