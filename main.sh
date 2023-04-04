#!/bin/sh

# Task A - CNN
python3 src/A_train_cnn.py 
python3 src/A_evalaute_cnn.py

# Task B - DistilBERT
python3 src/B_train_distilbert.py
python3 src/B_eval__distilbert.py

# Task C - MLP
python3 src/C_train_eval_mlp.py
