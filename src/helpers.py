import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import Config


def get_category(class_label):

    labels_map = Config.get_labels_map()
    mapping = labels_map.set_index('label')['categories'].to_dict()
    return mapping[class_label]


def plot_images(ds, class_names):

    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))

            # get index for label from one hot encoded
            label_index = tf.argmax(labels[i], axis=0)
            plt.title(get_category(class_label=int(class_names[label_index])))
            plt.axis("off")
    return plt


def plot_acc_loss(history, n_epochs):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(n_epochs)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')

    return plt


def evaluate(model, ds):
    score = model.evaluate(ds, verbose=0)
    print('Loss:', score[0])
    print('Accuracy:', score[1])


def predict_and_compare(model, ds) -> pd.DataFrame():

    # predictions
    predictions = model.predict(ds)
    label_pred = np.argmax(predictions, axis=1)
    category_pred = [get_category(class_label=l) for l in label_pred]
    predicted_prob = np.max(predictions, axis=1)

    # true labels
    label = np.argmax(np.concatenate([y for x, y in ds], axis=0), axis=1)
    category = [get_category(class_label=l) for l in label]

    # df
    pred_df = pd.DataFrame({'label':label, 'category':category,
                            'label_pred':label_pred, 'category_pred':category_pred,
                            'label_prob':predicted_prob})

    return pred_df


def accuracy_by_category(preds_df):

    acc_by_label = preds_df[preds_df['label']==preds_df['label_pred']].groupby('label').size() / preds_df.groupby('label').size()
    acc_by_label = acc_by_label.reset_index().rename(columns={0:'accuracy'})
    acc_by_label['categories'] = acc_by_label['label'].apply(lambda x : get_category(x))
    return acc_by_label


def plot_random_image_and_top_categories(ds, model, class_names):

    # get a random image and its label
    image, label = ds.take(1).unbatch().shuffle(1000).batch(1).as_numpy_iterator().next()

    # create subplots
    fig, axs = plt.subplots(2, 1, figsize=(5,5))

    # plot image
    axs[0].imshow(image.squeeze().astype("uint8"))
    axs[0].axis("off")

    # get index for label from one hot encoded
    label_index = np.argmax(label, axis=1)[0]

    # get category label
    category_label = get_category(class_label=int(class_names[label_index]))
    axs[0].set_title(category_label)

    # get the top 3 predictions
    predictions = model.predict(image)[0]
    class_predicted = np.argsort(predictions)[::-1][:3]
    category_predicted = [get_category(class_label=int(c)) for c in class_predicted]
    predicted_prob = np.sort(predictions)[::-1][:3]

    # plot bar chart
    axs[1].barh(category_predicted, predicted_prob)
    axs[1].invert_yaxis()
    axs[1].set_xlabel("Probability")

    plt.tight_layout()
    return fig


def plot_sample_image_and_top_categories(ds, model, class_names):

    fig = plt.figure(figsize=(20, 20))
    outer = gridspec.GridSpec(4, 3, wspace=0.5, hspace=0.3)

    for i in range(12):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)

        image, label = ds.take(1).unbatch().shuffle(1000).batch(1).as_numpy_iterator().next()
        label_index = np.argmax(label, axis=1)[0]
        category_label = get_category(class_label=int(class_names[label_index]))

        # get the top 3 predictions
        predictions = model.predict(image)[0]
        class_predicted = np.argsort(predictions)[::-1][:3]
        category_predicted = [get_category(class_label=int(c)) for c in class_predicted]
        predicted_prob = np.sort(predictions)[::-1][:3]

        for j in range(2):
            if (j%2) == 0:
                ax = plt.Subplot(fig, inner[j])
                ax.imshow(image.squeeze().astype("uint8"))
                ax.set_title(category_label)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)
            else:
                ax = plt.Subplot(fig, inner[j])
                ax.barh(category_predicted, predicted_prob)
                ax.invert_yaxis()
                ax.set_xlabel("Probability")
                fig.add_subplot(ax)
    return fig
