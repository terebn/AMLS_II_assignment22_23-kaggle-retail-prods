import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
import tensorflow as tf
from transformers import (
    AutoTokenizer, 
    TFAutoModelForSequenceClassification,
    )
from tensorflow.keras.optimizers import Adam
from modules.preprocess import train_val_image_ds, get_text_from_image_ds
from modules.helpers import (plot_acc_loss, evaluate,
                             plot_sample_image_and_top_categories)
from modules.evaluate import predict_and_compare, accuracy_by_category

from config import Config

MODEL_NAME = 'distilbert-base-uncased'

# Set paths
data_path = Config.data_path
out_path = Config.out_path
model_path = Config.model_path

# Get train and val data 
train_image_ds, val_image_ds = train_val_image_ds(data_path)
n_classes = len(train_image_ds.class_names)
train_text_df, val_text_df = get_text_from_image_ds(train_image_ds, val_image_ds)

# Prepare data 
train_ds = Dataset.from_pandas(train_text_df)
val_ds = Dataset.from_pandas(val_text_df)
datasets = DatasetDict({'train': train_ds, 'val': val_ds})

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(x):
    return tokenizer(x["text"], return_tensors="np", padding="max_length", truncation=True)

tokenized_datasets = datasets.map(tokenize_function, batched=True)

# Load and compile our model
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=n_classes)

# Lower learning rates are often better for fine-tuning transformers
model.compile(
    optimizer=Adam(3e-5),
    metrics=["accuracy"],
)

tf_dataset_train = model.prepare_tf_dataset(tokenized_datasets["train"], batch_size=Config.BERT_BATCH_SIZE, shuffle=True)
tf_dataset_val = model.prepare_tf_dataset(tokenized_datasets["val"], batch_size=Config.BERT_BATCH_SIZE, shuffle=False)

history = model.fit(
    tf_dataset_train,
    validation_data=tf_dataset_val,
    epochs=Config.BERT_EPOCHS)

model.save_pretrained(model_path / f'{Config.BERT_SUFFIX}_{Config.BERT_EPOCHS}')

# Evaluate on Validation and Test set 
fig_acc_loss = plot_acc_loss(history=history, n_epochs=Config.BERT_EPOCHS)
plt.savefig(out_path / f'acc_loss_{Config.BERT_SUFFIX}.png')

print(f"Train set evaluation: {evaluate(model=model, ds=tf_dataset_train)}")
print(f"Validation set evaluation: {evaluate(model=model, ds=tf_dataset_val)}")

# Val set prediction and prob 
val_df = predict_and_compare(model=model, ds=tf_dataset_val)
val_df.to_parquet(out_path / f'val_preds_{Config.BERT_SUFFIX}.pq')

# Accuracy by category
acc_by_category = accuracy_by_category(preds_df=val_df)
acc_by_category.to_csv(out_path / f'accuracy_by_category_{Config.BERT_SUFFIX}.csv', index=False)

# Plot top 3 predictions for a random set of pictures
fig_sample_top3_categories = plot_sample_image_and_top_categories(ds=tf_dataset_val, model=model,
                                                                  class_names=train_ds.class_names)
plt.savefig(out_path / f'sample_top3_categories_{Config.BERT_SUFFIX}.png')
