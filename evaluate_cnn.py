import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import visualkeras

from config import Config
from src.preprocess import prepare_ds
from src.evaluate import run_evaluation, plot_confusion_matrix


out_path = Config.out_path
model_path = Config.model_path

# load test data
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    Config.data_path / 'test',
    labels='inferred',
    label_mode='categorical',
    batch_size=Config.BATCH_SIZE,
    seed=123,
    image_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH))

test_ds_preprocessed = prepare_ds(test_ds)

# load model
cnn_aumg_model = tf.keras.models.load_model(model_path / 'cnn_augm') 

# plot architecture
plot_model(cnn_aumg_model, to_file=out_path / 'cnn_aumg_model.png',
           show_shapes=True, show_layer_names=True)
visualkeras.layered_view(cnn_aumg_model, legend=True,  to_file=out_path / 'cnn_aumg_model_flow.png')

# get results
res = run_evaluation(model=cnn_aumg_model,
                     ds_preprocessed=test_ds_preprocessed,
                     beta=3,
                     predictions=None,
                     is_image_data=True)

res['predictions_df'].to_csv(out_path / f'test_predictions{Config.SUFFIX}.csv', index=False)
res['acc_by_category'].to_csv(out_path / f'test_accuracy_by_category{Config.SUFFIX}.csv', index=False)
res['metrics'].to_csv(out_path / f'test_metrics{Config.SUFFIX}.csv', index=False)

# plot confusion matrix
fig = plot_confusion_matrix(res['confusion_matrix'], labels=res['predictions_df']['category'].unique())
fig.savefig(out_path / f'test_confusion_matrix{Config.SUFFIX}.png')
