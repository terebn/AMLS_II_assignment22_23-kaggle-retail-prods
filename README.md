# kaggle-retail-prods

This repo has the code for the [Kaggle Retail Products Classification Challenge](https://www.kaggle.com/competitions/retail-products-classification/overview).

## How the code is organised

The `src` directory contains all the code for the project. Inside, there are `.py` scripts for training and evaluating three models. Each script starts with the letter of the corresponding task: 
* `A_` is for the CNN
* `B_` is for DistilBERT 
* `C_` is for the MLP

Inside `src` there's also a `modules` directory. This contains the functions that are used across the three Tasks. They are divided into:

* `preprocess.py`: functions to pre-process the data from `dataset`
* `evaluate.py`: functions to evaluate the outcome of a model
* `helpers.py`: mostly plot ad utils functions 
* `CNNClassifier.py`: a python class for tuning and training the CNN

Finally, the `config.py` file has all the paramters used in the models.

## Set up

### 1. Set up the enviroment

#### Tensorflow GPU enviroment setup

I used Tensorflow with GPU support enabled as the main ML library for the analysis. To set up an enviroment that supports this, follow the [Tensorflow instruction](https://www.tensorflow.org/install/pip#windows-wsl2) for your operating system to enable GPU, in my case Windows WSL2.

You may have to add `/usr/lib/wsl/lib` on window sub system for linux 2 (wsl2). You may also need to add to `LD_LIBRARY_PATH` env variable as described [here](https://github.com/microsoft/WSL/issues/8587).

Following the set up of this enviroment using conda you can use pip to install the necessary python packages. My `requirements.txt` is included in this repo, but given the environemnt is composed of both conda and pip managed packages (as dictated by Tensorflow), you will likely need to re-build the enviroment from scratch rather than using `pip install requirements.txt`.

### 2. Download the data from Kaggle and unzip it in `dataset`

The dataset can be downloaded form [this link](https://www.kaggle.com/competitions/retail-products-classification/data). If you click 'Download all', you will download a zipped folder with the data. Just unzip it in `dataset` (keep the original name, 'retail-product-classification').

### 3. Create the Train, Test and Validate datasets

From the console, run `split_data_in_train_test.py` to create the Train, Validation and Test sets from the unzipped `retail-product-classification` directory:

```sh
python3 split_data_in_train_test.py
```

This also creates the sub-directories for:
* `out`, where all plots and tables are saved
* `models`, where model artefacts are saved
* `hp-tuning`, where the hyper-parameter tuning artefacts are saved

### 4. Run the whole routine

Now that you have everything set up, to run all the tasks in order, you can launch the `main.sh` file:

```sh
./main.sh
```

## Requirements

Notwitstanding what said above for the creation of the enviroment, here's a list of required packages:

* pandas==1.5.3
* scikit-learn==1.2.2
* tensorflow==2.12.0
* matplotlib==3.7.1
* keras==2.12.0
* keras-tuner==1.3.1
* visualkeras==0.0.2
* transformers==4.27.3
* datasets==2.10.1
* scipy==1.10.1
* umpy==1.23.5
* seaborn==0.12.2
