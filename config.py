from pathlib import Path

class Config():

    # Project paths
    project_path = Path(__file__).parent
    out_path = project_path / 'out'
    data_path = project_path / 'datasets'

    # Data prep
    VALIDATION_PROP = 0.2
    BATCH_SIZE = 32
    IMG_HEIGHT = 100
    IMG_WIDTH = 100
