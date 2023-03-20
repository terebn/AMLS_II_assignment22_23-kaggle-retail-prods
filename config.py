from pathlib import Path
import pandas as pd

class Config():

    # Project paths
    project_path = Path(__file__).parent
    out_path = project_path / 'out'
    data_path = project_path / 'datasets'
    model_path = project_path / 'models'
    hp_tuning_path = project_path / 'hp-tuning'

    # Data prep
    VALIDATION_PROP = 0.2
    IMG_HEIGHT = 100
    IMG_WIDTH = 100

    # CNN

    # Tuning
    MAX_EPOCHS_HP = 100  # the maximum number of epochs to train one model. It is recommended to set this to a value slightly higher than the expected epochs to convergence for your largest Model FACTOR = 3  # the reduction factor for the number of epochs and number of models for each
    HYPERBAND_ITERATIONS = 1  # the number of times to iterate over the full Hyperband algorithm
    FACTOR = 3
    SUFFIX = '_100'

    # Fitting
    EPOCHS = 30

    @staticmethod
    def get_labels_map():
        return pd.read_csv(Config.data_path / 'labels.csv')
