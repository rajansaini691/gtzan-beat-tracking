import os

# TODO Put this stuff into a struct that gets passed around
SAMPLE_SIZE = 1024
SAMPLE_RATE = 44100

# TODO Refactor to use os.path.join
wav_data_root = "./data/Data/genres_original/"
json_annotations_root = "./data/GTZAN-Rhythm_v2_ismir2015_lbd/jams"
numpy_annotations_root = "./out/numpy"
dataset_metadata_path = "./out/metadata"
tensorboard_dir = "./out/tensorboard"
test_songs_root = "./data/test_songs"
checkpoint_dir = './out/ckpt'
checkpoint_filepath = os.path.join(checkpoint_dir, 'checkpoint')
