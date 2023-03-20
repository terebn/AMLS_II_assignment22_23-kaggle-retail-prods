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


# resources:


https://neptune.ai/blog/image-classification-tips-and-tricks-from-13-kaggle-competitions

https://machinelearningmastery.com/object-recognition-with-deep-learning/#:~:text=Image%20classification%20involves%20predicting%20the,more%20objects%20in%20an%20image. 

to talk about tuning:
https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7
