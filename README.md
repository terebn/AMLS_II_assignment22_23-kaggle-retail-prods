# kaggle-retail-prods
https://www.kaggle.com/competitions/retail-products-classification/overview

# 

# conda env 

To create the `re-prods` conda enviroment:

```sh
conda env create --file environment.yml
```

For jupyter notebooks you'll need ipykernel:

```sh
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=re-prods
```

Note: the librabry is not included in the enviroment as it's not necessay for running hte final routine.


# plan:

1. develp a baseline model: classify images without using any descriptipon.
2. add more to the baseline model
3. add text
4. find similar items and build a search algo to find best matches, ranked.


# dev enviroment

Before creating the environment, follow the [Tensorflow instruction](https://www.tensorflow.org/install/pip#windows-wsl2) for your operating system to enable GPU, in my case Windows WLS2.

You may have to add /usr/lib/wsl/lib on window sub system for linux 2 (wsl2) you mayb need to add to LD_LIBRARY_PATH (see https://github.com/microsoft/WSL/issues/8587)

# resources:


https://neptune.ai/blog/image-classification-tips-and-tricks-from-13-kaggle-competitions

https://machinelearningmastery.com/object-recognition-with-deep-learning/#:~:text=Image%20classification%20involves%20predicting%20the,more%20objects%20in%20an%20image. 

to talk about tuning:
https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7


# data preparation

The function resizes and rescales all images in the dataset using tf.keras.Sequential.

The function also shuffles the dataset (if shuffle is set to True) and applies data augmentation (if augment is set to True) only on the training set.

Finally, the function batches the dataset using ds.batch(Config.BATCH_SIZE) and applies buffered prefetching using ds.prefetch(buffer_size=AUTOTUNE). This technique is used to optimize performance by prefetching the next batch of data before the current batch is finished, allowing for more efficient utilization of CPU and GPU resources.

Overall, this implementation of the prepare_ds function covers the key steps necessary to prepare a dataset for training a CNN in TensorFlow, including resizing, rescaling, shuffling, data augmentation, batching, and prefetching.

# hyper param tuning

cnn_augm
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is 192. The optimal learning rate for the optimizer
        is 0.0001. The optimal dropout rate is 0.4.