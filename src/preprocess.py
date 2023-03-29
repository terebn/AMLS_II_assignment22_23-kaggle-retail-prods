from config import Config
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gensim
from nltk.corpus import stopwords
import pandas as pd

def train_val_image_ds(data_path):
    """
    Returns two TensorFlow datasets: one for training and one for validation. The datasets are created using the 
    images in the 'train' subdirectory of the given 'data_path' directory, which should be organized into subdirectories 
    for each class. The datasets are split into training and validation sets according to the 'validation_split' 
    parameter in the Config object. The datasets are batched using the 'BATCH_SIZE' parameter in the Config object and 
    resized to the height and width specified in the Config object.

    Parameters:
    - data_path (str or pathlib.Path): the directory containing the image data

    Returns:
    - train_ds, val_ds (tuple of tensorflow.data.Dataset): the training and validation datasets, respectively
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path / 'train',
        labels='inferred',
        class_names=[f"{x}" for x in range(21)],
        label_mode='categorical',
        validation_split=Config.VALIDATION_PROP,
        subset="training",
        batch_size=Config.BATCH_SIZE,
        seed=123,
        image_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH))
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path / 'train',
        labels='inferred',
        class_names=[f"{x}" for x in range(21)],
        label_mode='categorical',
        validation_split=Config.VALIDATION_PROP,
        subset="validation",
        batch_size=Config.BATCH_SIZE,
        seed=123,
        image_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH))
    
    return train_ds, val_ds


def get_text_from_image_ds(train_ds, val_ds):
    """
    Given two TensorFlow datasets representing image data for training and validation, extracts the corresponding text 
    descriptions from a Parquet file containing the text data. The text descriptions are combined from the 'title' and 
    'description' columns of the Parquet file, and are returned along with their corresponding labels and image IDs.

    Parameters:
    - train_ds (tensorflow.data.Dataset): the training dataset of images
    - val_ds (tensorflow.data.Dataset): the validation dataset of images

    Returns:
    - train_text, val_text (tuple of pandas.DataFrame): the text descriptions, labels, and image IDs for the training 
      and validation datasets, respectively. Each DataFrame has columns for 'label', 'text', and 'ImgId'.
    """
    # get text description
    df = pd.read_parquet(Config.data_path / 'train.pq')
    df['text'] = np.where((df['title'] + ' ' + df['description']).isna(), 'none', (df['title'] + ' ' + df['description']))

    ImgId_train = [p.split('/')[-1].replace('.jpg', '') for p in train_ds.file_paths]
    ImgId_val = [p.split('/')[-1].replace('.jpg', '') for p in val_ds.file_paths]

    train_text = pd.DataFrame(ImgId_train, columns=['ImgId']).merge(df[['label', 'text', 'ImgId']], on='ImgId')
    val_text = pd.DataFrame(ImgId_val, columns=['ImgId']).merge(df[['label', 'text', 'ImgId']], on='ImgId')

    return train_text, val_text


def prepare_ds(ds, shuffle=False, augment=False):
    """
    Preprocesses a given TensorFlow dataset for use in a convolutional neural network (CNN).

    Args:
        ds (tf.data.Dataset): A TensorFlow dataset to be preprocessed. Each element of the dataset
                                should be a tuple of (image, label).
        shuffle (bool): Whether to shuffle the dataset (default False).
        augment (bool): Whether to apply data augmentation to the training set (default False).

    Returns:
        The preprocessed TensorFlow dataset ready for training a CNN. The dataset has been resized
        and rescaled, shuffled (if shuffle=True), and augmented (if augment=True) for training. The
        dataset has also been batched and prefetched for optimal performance.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    resize_and_rescale = tf.keras.Sequential([
     layers.Resizing(Config.IMG_HEIGHT, Config.IMG_WIDTH),
     layers.Rescaling(1./255)
    ])

    # Resize and rescale all datasets.
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
                num_parallel_calls=AUTOTUNE)
    
    # Shuffle only training data
    if shuffle:
        ds = ds.shuffle(1000)

    # Use data augmentation only on the training set.
    data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            ])

    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                    num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)
