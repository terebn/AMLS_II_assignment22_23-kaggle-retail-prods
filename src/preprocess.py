from config import Config
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


def train_val_image_ds(data_path, save_ImgId=False):

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_path / 'train',
        labels='inferred',
        label_mode='categorical',
        validation_split=Config.VALIDATION_PROP,
        subset="training",
        seed=123,
        image_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH))
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_path / 'train',
        labels='inferred',
        label_mode='categorical',
        validation_split=Config.VALIDATION_PROP,
        subset="validation",
        seed=123,
        image_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH))
    
    if save_ImgId:

        ImgId_val = [p.split('/')[-1].replace('.jpg', '') for p in val_ds.file_paths]
        ImgId_train = [p.split('/')[-1].replace('.jpg', '') for p in train_ds.file_paths]

        ImgId_val = pd.DataFrame(ImgId_val, columns=["ImgId"])
        ImgId_val.to_csv(Config.out_path / 'ImgId_val.csv', index=False)

        ImgId_train = pd.DataFrame(ImgId_train, columns=["ImgId"])
        ImgId_train.to_csv(Config.out_path / 'ImgId_train.csv', index=False)
    
    return train_ds, val_ds


def get_image_text_label_id():

    train_ds, val_ds = train_val_image_ds(data_path=Config.data_path, save_ImgId=False)

    for images, labels in train_ds.take(1):
        train_images = images.numpy()
        train_labels = labels.numpy()

    for images, labels in val_ds.take(1):
        val_images = images.numpy()
        val_labels = labels.numpy()

    ImgId_train = [p.split('/')[-1].replace('.jpg', '') for p in train_ds.file_paths]
    ImgId_val = [p.split('/')[-1].replace('.jpg', '') for p in val_ds.file_paths]

    df = pd.read_parquet(Config.data_path / 'train.pq')

    df_train = df[df['ImgId'].isin(ImgId_train)]
    df_val = df[df['ImgId'].isin(ImgId_val)]

    train_text  = df_train['title'] + df_train['description']
    val_text = df_val['title'] + df_val['description']

    train_ids = df_train['ImgId']
    val_ids = df_val['ImgId']

    return ({'labels':train_labels, 'images':train_images, 'text':train_text, 'ids':train_ids},
            {'labels':val_labels, 'images':val_images, 'text':val_text, 'ids':val_ids})


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

    # Batch all datasets.
    ds = ds.batch(Config.BATCH_SIZE)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)
