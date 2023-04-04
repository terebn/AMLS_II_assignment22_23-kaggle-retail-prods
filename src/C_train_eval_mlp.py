import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer, 
    TFAutoModelForSequenceClassification,
    )
from tensorflow.keras.optimizers import Adam
from datasets import Dataset, DatasetDict
from scipy.special import softmax

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from config import Config
from modules.preprocess import train_val_image_ds, prepare_ds, get_text_from_image_ds
from modules.evaluate import predict_and_compare


# Set paths
data_path = Config.data_path
out_path = Config.out_path
model_path = Config.model_path


# Get train and val Image data
train_image_ds, val_image_ds = train_val_image_ds(data_path=data_path)
class_names = train_image_ds.class_names

train_image_ds_preprocessed = prepare_ds(train_image_ds, shuffle=False, augment=False)
val_image_ds_preprocessed = prepare_ds(val_image_ds, shuffle=False, augment=False)

# get predictions
file_paths = train_image_ds_preprocessed._input_dataset._input_dataset.file_paths
ImgId_train = [p.split('/')[-1].replace('.jpg', '') for p in file_paths]

file_paths_val = val_image_ds_preprocessed._input_dataset._input_dataset.file_paths
ImgId_val = [p.split('/')[-1].replace('.jpg', '') for p in file_paths_val]

# Get model
cnn_aumg_model = tf.keras.models.load_model(model_path / 'cnn_augm')

image_train_preds = predict_and_compare(cnn_aumg_model,
                                        train_image_ds_preprocessed,
                                        predictions=None,
                                        is_image_data=True)
image_train_preds['ImgId'] = ImgId_train

image_val_preds = predict_and_compare(cnn_aumg_model,
                                        val_image_ds_preprocessed,
                                        predictions=None,
                                        is_image_data=True)
image_val_preds['ImgId'] = ImgId_val

# Get train and val Text data
train_text_df, val_text_df = get_text_from_image_ds(train_image_ds, val_image_ds)
train_ds = Dataset.from_pandas(train_text_df)
val_ds = Dataset.from_pandas(val_text_df)
datasets = DatasetDict({'train': train_ds, 'val': val_ds})

# Get model
distilbert_model = TFAutoModelForSequenceClassification.from_pretrained(model_path / "distilbert_3_2")

# Tokenize
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize_function(x):
    return tokenizer(x["text"], return_tensors="np", padding="max_length", truncation=True)

tokenized_datasets = datasets.map(tokenize_function, batched=True)

tf_dataset_train = distilbert_model.prepare_tf_dataset(tokenized_datasets["train"], batch_size=Config.BERT_BATCH_SIZE, shuffle=True)
tf_dataset_val = distilbert_model.prepare_tf_dataset(tokenized_datasets["val"], batch_size=Config.BERT_BATCH_SIZE, shuffle=False)

# Make predictions
preds = distilbert_model.predict(tf_dataset_train)["logits"]
y_probs = [softmax(p) for p in preds]

text_train_preds = predict_and_compare(distilbert_model,
                                       tf_dataset_train,
                                       predictions=y_probs,
                                       is_image_data=False)

# Validation set
preds_val = distilbert_model.predict(tf_dataset_val)["logits"]
y_probs_val = [softmax(p) for p in preds_val]

text_val_preds = predict_and_compare(distilbert_model,
                                     tf_dataset_val,
                                     predictions=y_probs_val,
                                     is_image_data=False)

# combine text and image predictions for Train set
df_train = pd.concat([train_text_df[['ImgId', 'label']],
                           text_train_preds.add_suffix('_text'),
                           image_train_preds.add_suffix('_image')], axis=1)

# text_train_df = pd.concat([train_text_df[['ImgId', 'label']],
#                            text_train_preds.add_suffix('_text')], axis=1)

# df_train = text_train_df.merge(image_train_preds.add_suffix('_image'), left_on=['ImgId', 'label'], right_on=['ImgId_image', 'label_image'])
df_train.to_parquet(out_path / 'train_preds_text_image.pq')

# combine text and image predictions for Val set
df_val = pd.concat([val_text_df[['ImgId', 'label']],
                         text_val_preds.add_suffix('_text'),
                         image_val_preds.add_suffix('_image')], axis=1)

# text_val_df = pd.concat([val_text_df[['ImgId', 'label']],
#                          text_val_preds.add_suffix('_text')], axis=1)

# df_val = text_val_df.merge(image_val_preds.add_suffix('_image'), left_on=['ImgId', 'label'], right_on=['ImgId_image', 'label_image'])
df_val.to_parquet(out_path / 'val_preds_text_image.pq')

# fit a MLP

def fit_MLPClassifier(df, df_val):

    X_train = df[['label_pred_text', 'label_pred_image']]
    y_train = df['label']

    X_val = df_val[['label_pred_text', 'label_pred_image']]
    y_val = df_val['label']

    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

    return pd.DataFrame({
        'train_accuracy':clf.score(X_train, y_train),
        'val_accuracy':clf.score(X_val, y_val),
    }, index=[0])

mlp_accuracy = fit_MLPClassifier(df_train, df_val)
mlp_accuracy.to_csv(out_path / f'mlp_accuracy.csv', index=False)
