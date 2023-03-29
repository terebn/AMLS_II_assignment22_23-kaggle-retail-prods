import tensorflow as tf
import matplotlib.pyplot as plt

from config import Config
from src.helpers import (plot_images, plot_acc_loss, evaluate,
                         plot_sample_image_and_top_categories)
from src.CNNClassifier import CNNClassifier
from src.preprocess import train_val_image_ds, prepare_ds
from src.evaluate import accuracy_by_category, predict_and_compare


# Set paths
data_path = Config.data_path
out_path = Config.out_path
model_path = Config.model_path

# Get train and val data
train_ds, val_ds = train_val_image_ds(data_path=data_path)
class_names = train_ds.class_names

# Plot some images
fig_images = plot_images(ds=train_ds, class_names=class_names)
plt.savefig(out_path / f'sample_train_images{Config.SUFFIX}.png')

# CNN

train_ds_preprocessed = prepare_ds(train_ds, shuffle=True, augment=True)
val_ds_preprocessed = prepare_ds(val_ds)

num_classes = len(class_names)
cnn = CNNClassifier(train_ds=train_ds_preprocessed, val_ds=val_ds_preprocessed, num_classes=num_classes)

# Hyperparam tuning
best_model = cnn.hyperparameter_tuning()
print(best_model.summary())

# Fit model with best params
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, start_from_epoch=5)

history = best_model.fit(train_ds_preprocessed,
                         epochs=Config.EPOCHS,
                         validation_data=val_ds_preprocessed,
                         callbacks=[early_stop],
                         batch_size=32)
best_model.save(model_path / f'cnn{Config.SUFFIX}')

# Evaluate on Validation and Test set 
fig_acc_loss = plot_acc_loss(history=history, n_epochs=Config.EPOCHS)
plt.savefig(out_path / f'acc_loss{Config.SUFFIX}.png')

print(f"Train set evaluation: {evaluate(model=best_model, ds=train_ds_preprocessed)}")
print(f"Validation set evaluation: {evaluate(model=best_model, ds=val_ds_preprocessed)}")

# Val set prediction and prob 
val_df = predict_and_compare(model=best_model,
                             ds=val_ds_preprocessed,
                             predictions=None,
                             is_image_data=True)
val_df.to_parquet(out_path / f'val_preds{Config.SUFFIX}.pq')

# Accuracy by category
acc_by_category = accuracy_by_category(preds_df=val_df)
acc_by_category.to_csv(out_path / f'accuracy_by_category{Config.SUFFIX}.csv', index=False)

# Plot top 3 predictions for a random set of pictures
fig_sample_top3_categories = plot_sample_image_and_top_categories(ds=val_ds, model=best_model,
                                                                  class_names=class_names)
plt.savefig(out_path / f'sample_top3_categories{Config.SUFFIX}.png')
