from config import Config
import pandas as pd
import tensorflow as tf
import gensim
from nltk.corpus import stopwords


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


def get_image_text_label_id(data_path):

    train_ds, val_ds = train_val_image_ds(data_path=data_path, save_ImgId=False)

    class_names = train_ds.class_names

    for images, labels in train_ds.take(1):
        train_images = images.numpy()
        train_labels = labels.numpy()

    for images, labels in val_ds.take(1):
        val_images = images.numpy()
        val_labels = labels.numpy()

    ImgId_train = [p.split('/')[-1].replace('.jpg', '') for p in train_ds.file_paths]
    ImgId_val = [p.split('/')[-1].replace('.jpg', '') for p in val_ds.file_paths]

    df = pd.read_parquet(data_path / 'train.pq')

    df_train = df[df['ImgId'].isin(ImgId_train)]
    df_val = df[df['ImgId'].isin(ImgId_val)]

    train_text  = df_train['title'] + ' ' + df_train['description']
    val_text = df_val['title'] + ' ' + df_val['description']

    train_ids = df_train['ImgId']
    val_ids = df_val['ImgId']

    return ({'labels':train_labels, 'images':train_images, 'text':train_text, 'ids':train_ids},
            {'labels':val_labels, 'images':val_images, 'text':val_text, 'ids':val_ids},
            class_names)


def get_token(description):
    stop_english=set(stopwords.words('english'))
    
    token = list(gensim.utils.tokenize(description))
    token = [i for i in token if(len(i) > 2)]
    token = [s for s in token if s not in stop_english]
    return token