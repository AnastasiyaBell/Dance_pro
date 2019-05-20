import os
from collections import Counter

import tensorflow as tf


def get_file_names_in_dataset(dataset_path):
    classes = os.listdir(dataset_path)
    file_names_in_dataset = {}
    for cl in classes:
        file_names_in_dataset[cl] = sorted(os.listdir(os.path.join(dataset_path, cl)))
    return file_names_in_dataset


def video_name_from_file_name(file_name):
    return '_'.join(file_name.split('_')[:-1])


def get_num_of_frames_in_videos(list_of_file_names):
    videos_names = map(lambda x: video_name_from_file_name(x), list_of_file_names)
    return Counter(videos_names)


def select_videos_with_N_frames(list_of_file_names, N):
    nfr = get_num_of_frames_in_videos(list_of_file_names)
    video_names, _ = zip(*filter(lambda x: x[1] == N, nfr.items()))
    return video_names


def select_video_names_for_dances(file_names_in_dataset, N):
    """Selects videos with N frames for each dance so all dances
    have equal number of videos. Number of videos for a dance is
    the smallest number of videos having N frames among all dances."""
    selected = {}
    for dance_name, list_of_file_names in file_names_in_dataset.items():
        videos_with_N_frames = select_videos_with_N_frames(list_of_file_names, N)
        selected[dance_name] = videos_with_N_frames
    min_num_of_videos_with_N_frames = min(map(len, selected.values()))
    for k, v in selected.items():
        selected[k] = sorted(v)[:min_num_of_videos_with_N_frames]
    return selected


def select_file_names_for_work(file_names_in_dataset, N):
    video_names = select_video_names_for_dances(file_names_in_dataset, N)
    selected_file_names = {}
    for dance, list_of_file_names in file_names_in_dataset.items():
        selected_file_names[dance] = [fn for fn in list_of_file_names
                                      if video_name_from_file_name(fn) in video_names[dance]]
    return selected_file_names


def prepend_path(file_names_in_dataset, path):
    for dance, loffn in file_names_in_dataset.items():
        n = len(loffn)
        for i in range(n):
            loffn[i] = os.path.join(path, dance, loffn[i])


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    return tf.image.resize_images(image_decoded, (227, 227)), label


def build_dataset(file_names_for_dataset, batch_size, num_dances):
    datasets_by_dance = {}

    for idx, (dance, loffn) in enumerate(sorted(file_names_for_dataset.items())):
        labels = tf.constant([idx] * len(loffn))
        filenames = tf.constant(loffn)
        datasets_by_dance[dance] = tf.data.Dataset.from_tensor_slices(
            (filenames, labels)
        ).shuffle(len(loffn)).map(_parse_function)
    # print()
    dance_zip = tf.data.Dataset.zip(tuple(datasets_by_dance.values()))
    # print(dance_zip)
    return dance_zip.batch(batch_size // num_dances)


def filter_dataset(file_names_in_dataset):
    print("beforer filtering")
    for dance, loffn in file_names_in_dataset.items():
        print(dance,
              'total number of frames: {}'.format(len(loffn)),
              'number of videos: {}'.format(len(get_num_of_frames_in_videos(loffn))),
              end='\n\n', sep='\n')
    print('*********\n\nAfter filtering')
    file_names = select_file_names_for_work(file_names_in_dataset, 300)
    dance, loffn = list(file_names.items())[0]
    print('total number of frames: {}'.format(len(loffn)),
          'number of videos: {}'.format(len(get_num_of_frames_in_videos(loffn))),
          end='\n\n', sep='\n')
    return file_names


def select_train_valid_test_file_names(dataset_path, train_path, valid_path, test_path):
    train_path = os.path.join(dataset_path, train_path)
    train_file_names = get_file_names_in_dataset(train_path)
    print(("*"*20 + '\n') * 2 + "filtering file names for train dataset")
    train_file_names = filter_dataset(train_file_names)
    prepend_path(train_file_names, train_path)

    valid_path = os.path.join(dataset_path, valid_path)
    valid_file_names = get_file_names_in_dataset(valid_path)
    print(("*"*20 + '\n') * 2 + "filtering file names for validation dataset")
    valid_file_names = filter_dataset(valid_file_names)
    prepend_path(valid_file_names, valid_path)

    test_path = os.path.join(dataset_path, test_path)
    test_file_names = get_file_names_in_dataset(test_path)
    print(("*"*20 + '\n') * 2 + "filtering file names for test dataset")
    test_file_names = filter_dataset(test_file_names)
    prepend_path(test_file_names, test_path)

    file_names = {
        'train': train_file_names,
        'valid': valid_file_names,
        'test': test_file_names,
    }

    return file_names
