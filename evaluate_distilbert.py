import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer, 
    TFAutoModelForSequenceClassification,
    )
from tensorflow.keras.optimizers import Adam
from datasets import Dataset
from scipy.special import softmax

from config import Config
from src.evaluate import run_evaluation, plot_confusion_matrix


out_path = Config.out_path
model_path = Config.model_path
data_path = Config.data_path

# load fine-tuned model
model = TFAutoModelForSequenceClassification.from_pretrained(model_path / "distilbert_3_2")

# prepare test data
test_df = pd.read_parquet(data_path / 'test.pq')
test_df['text'] = np.where((test_df['title'] + ' ' + test_df['description']).isna(), 'none', (test_df['title'] + ' ' + test_df['description']))

# load the test images to make sure we only use those
test_ds_images = tf.keras.preprocessing.image_dataset_from_directory(
    Config.data_path / 'test',
    labels='inferred',
    label_mode='categorical',
    batch_size=Config.BATCH_SIZE,
    seed=123,
    image_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH))

ImgId_test = [p.split('/')[-1].replace('.jpg', '') for p in test_ds_images.file_paths]
test_text_df = pd.DataFrame(ImgId_test, columns=['ImgId']).merge(test_df[['label', 'text', 'ImgId']], on='ImgId')

# text Test df into Dataset
test_ds = Dataset.from_pandas(test_text_df)

# tokenize
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(x):
    return tokenizer(x["text"], return_tensors="np", padding="max_length", truncation=True)

tokenized_test_ds = test_ds.map(tokenize_function, batched=True)

tf_dataset_test = model.prepare_tf_dataset(tokenized_test_ds, batch_size=Config.BERT_BATCH_SIZE, shuffle=False)

# make predictions
preds = model.predict(tf_dataset_test)["logits"]

y_probs = [softmax(p) for p in preds]

model.compile(
    optimizer=Adam(3e-5),
    metrics=["accuracy"],
)

# get results
res = run_evaluation(model,
                     tf_dataset_test,
                     beta=3,
                     predictions=y_probs,
                     is_image_data=False)

res['predictions_df'].to_csv(out_path / f'test_predictions_{Config.BERT_SUFFIX}.csv', index=False)
res['acc_by_category'].to_csv(out_path / f'test_accuracy_by_category_{Config.BERT_SUFFIX}.csv', index=False)
res['metrics'].to_csv(out_path / f'test_metrics_{Config.BERT_SUFFIX}.csv', index=False)

# plot confusion matrix
fig = plot_confusion_matrix(res['confusion_matrix'], labels=res['predictions_df']['category'].unique())
fig.savefig(out_path / f'test_confusion_matrix_{Config.BERT_SUFFIX}.png')
