from sklearn.metrics import confusion_matrix, fbeta_score, f1_score, roc_auc_score
import numpy as np
import seaborn as sns
from src.modules.helpers import get_category
import pandas as pd
import matplotlib.pyplot as plt



def accuracy_by_category(preds_df):

    acc_by_label = preds_df[preds_df['label']==preds_df['label_pred']].groupby('label').size() / preds_df.groupby('label').size()
    acc_by_label = acc_by_label.reset_index().rename(columns={0:'accuracy'})
    acc_by_label['categories'] = acc_by_label['label'].apply(lambda x : get_category(x))
    return acc_by_label


def plot_confusion_matrix(cf_mtx, labels):

    group_percentages = ["{0:.2%}".format(value) for value in cf_mtx.flatten()/np.sum(cf_mtx)]
    box_labels = [f"{p}" for p in group_percentages]
    box_labels = np.asarray(box_labels).reshape(len(labels), len(labels))

    plt.figure(figsize = (24, 20))
    sns.heatmap(cf_mtx, xticklabels=labels, yticklabels=labels,
                cmap="YlGnBu", fmt="", annot=box_labels)
    plt.xlabel('Predicted Classes')
    plt.ylabel('True Classes')
    return plt


def evaluate_model(model, ds):

    score = model.evaluate(ds, verbose=0)
    loss = score[0]
    accuracy = score[1]
    return loss, accuracy


def predict_and_compare(model, ds, predictions=None, is_image_data=False) -> pd.DataFrame():

    # predictions
    if predictions is None:
        predictions = model.predict(ds)

    # predicted label
    label_pred = np.argmax(predictions, axis=1)

    category_pred = [get_category(class_label=l) for l in label_pred]
    predicted_prob = np.max(predictions, axis=1)

    # true labels
    if is_image_data:
        label = np.argmax(np.concatenate([y for x, y in ds], axis=0), axis=1)
    else:
        label = np.concatenate([y for x, y in ds], axis=0)

    category = [get_category(class_label=l) for l in label]

    # df
    pred_df = pd.DataFrame({'label':label,
                            'category':category,
                            'label_pred':label_pred,
                            'category_pred':category_pred,
                            'label_prob':predicted_prob})

    return pred_df


def run_evaluation(model, ds_preprocessed, beta, predictions=None, is_image_data=False):

    loss, accuracy = evaluate_model(model, ds_preprocessed)
    df = predict_and_compare(model=model, ds=ds_preprocessed, predictions=predictions, is_image_data=is_image_data)
    acc_by_category = accuracy_by_category(preds_df=df)

    y_true = df['category']
    y_pred = df['category_pred']
    if predictions is None:
        y_probs = model.predict(ds_preprocessed)
    else:
        y_probs = predictions

    auc = roc_auc_score(y_true, y_probs, average='macro', multi_class='ovr') #ovr since classes are balanced
    macro_averaged_fbeta = fbeta_score(y_true, y_pred, average='macro', beta=beta)
    macro_averaged_f1 = f1_score(y_true, y_pred, average = 'macro')
    cf_mtx = confusion_matrix(y_true, y_pred)

    metrics = pd.DataFrame({
            'loss':loss,
            'accuracy':accuracy,
            'auc':auc,
            'macro_averaged_f1':macro_averaged_f1,
            'macro_averaged_fbeta':macro_averaged_fbeta,}, index=[0])

    return {'predictions_df':df,
            'acc_by_category':acc_by_category,
            'confusion_matrix':cf_mtx,
            'metrics':metrics}
