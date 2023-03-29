from config import Config
from sklearn.model_selection import train_test_split
import shutil
import os
import pandas as pd


# make directories
data_path = Config.data_path
out_path = Config.out_path
model_path = Config.model_path
hp_tuning_path = Config.hp_tuning_path

data_path.mkdir(parents=True, exist_ok=True)
model_path.mkdir(parents=True, exist_ok=True)
hp_tuning_path.mkdir(parents=True, exist_ok=True)
out_path.mkdir(parents=True, exist_ok=True)

# Data from kaggle
competition_raw_data_path = data_path / 'retail-products-classification' 
image_dir = competition_raw_data_path / 'train' / 'train'
csv_file = competition_raw_data_path / 'train.csv'

# List photos
jpeg_names = os.listdir(image_dir)

# training categories and labels
data = pd.read_csv(csv_file)
labels = pd.read_csv(data_path / 'labels.csv')
data = data.merge(labels, on='categories')

# Split in train and test
label_map = data[['ImgId', 'label']]
train, test = train_test_split(label_map, test_size=0.2, stratify=label_map['label'],
                               random_state=888)

# Get images into folders named with their label
n_train = 0
n_test = 0

for img in jpeg_names:

    label = label_map.loc[label_map['ImgId'] == img.split('.')[0], 'label'].item()
    ImgId = img.split('.')[0]

    if  len(train[train['ImgId']==ImgId])>0:
        out_path = data_path / 'train' / str(label)
        n_train += 1
    elif len(test[test['ImgId']==ImgId])>0:
        out_path = data_path / 'test' / str(label)
        n_test += 1
    else:
        print("not in train nor test")

    out_path.mkdir(parents=True, exist_ok=True)
    
    source_path = image_dir / img

    shutil.copy(source_path, out_path)


# Get title and descriptions

corpus = data.copy()

train_ImgId = []

train_dir = data_path / 'train' 

for l in corpus.label.unique():
    dir = (train_dir / str(l))
    f = list(dir.glob('*.jpg'))
    train_ImgId = train_ImgId + [path.stem.replace('.jpg', '') for path in f]

train = corpus[corpus.ImgId.isin(train_ImgId)]
test = corpus[~corpus.ImgId.isin(train_ImgId)]

train.to_parquet(data_path / 'train.pq')
test.to_parquet(data_path / 'test.pq')
