import tensorflow as tf
from tcn import TCN

def get_model_v1(max_sequence_length, spectrum_size):
    return tf.keras.Sequential([
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


