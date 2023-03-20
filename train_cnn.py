import tensorflow as tf
import matplotlib.pyplot as plt

from config import Config
from src.helpers import (plot_images, plot_acc_loss, evaluate, predict_and_compare,
                         accuracy_by_category, plot_sample_image_and_top_categories)
from src.CNNClassifier import CNNClassifier
from src.preprocess import train_val_image_ds


# Set paths
data_path = Config.data_path
out_path = Config.out_path
model_path = Config.model_path


# Get train and val data
train_ds, val_ds = train_val_image_ds(data_path=data_path, save_ImgId=False)
class_names = train_ds.class_names

# Plot some images
fig_images = plot_images(ds=train_ds, class_names=class_names)
plt.savefig(out_path / 'sample_train_images.png')

# CNN
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)
cnn = CNNClassifier(train_ds=train_ds, val_ds=val_ds, num_classes=num_classes)

# Hyperparam tuning
best_model = cnn.hyperparameter_tuning()

print(best_model.summary())

# Fit model with best params
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, start_from_epoch=5)

history = best_model.fit(train_ds,
                         epochs=Config.EPOCHS,
                         validation_data=val_ds,
                         callbacks=[early_stop],
                         batch_size=32)
best_model.save(Config.project_path  / 'models' / 'cnn')

# Evaluate on Validation and Test set 
fig_acc_loss = plot_acc_loss(history=history, n_epochs=Config.EPOCHS)
plt.savefig(out_path / 'acc_loss.png')

print(f"Train set evaluation: {evaluate(model=best_model, ds=train_ds)}")
print(f"Validation set evaluation: {evaluate(model=best_model, ds=val_ds)}")

# Val set prediction and prob 
val_df = predict_and_compare(model=best_model, ds=val_ds)
val_df.to_parquet(out_path / 'val_preds.pq')

# Accuracy by category
acc_by_category = accuracy_by_category(preds_df=val_df)
acc_by_category.to_csv(out_path / 'accuracy_by_category.csv', index=False)

# Plot top 3 predictions for a random set of pictures
fig_sample_top3_categories = plot_sample_image_and_top_categories(ds=val_ds, model=best_model,
                                                                  class_names=class_names)
plt.savefig(out_path / 'sample_top3_categories.png')
