# Sequential Path Signature Networks for Personalised Longitudinal Language Modeling

This repository contains the code for the paper "Sequential Path Signature Networks for Personalised Longitudinal Language Modelings", accepted at ACL Findings 2023.

# Datasets

Due to the sensitive nature of the data, the paper datasets of Reddit and Talklife are not publically available. File `paths.py` contains paths to the datasets and directories that we exclude, but which you can use to source datasets for your own longitudinal task. 
The following columns are necessary for the input dataset: `timeline_id`, `postid`, `label`, `datetime`, as well as a column that determines the train/dev/test splits (or a function that randomnly splits the dataset by timeline_id). The user also needs to provide a seperate file that contains the embeddings (here Sentence-BERT) of your choice.

# Installation

Clone the git repo:

`$ git clone git@github.com:Maria-Liakata-NLP-Group/seq-sig-net.git`

Create a conda environment:

`$ conda env create --file=environment.yml`

Actviate the conda environment

`$ conda activate py38-MoC`

# Models

Model notebooks are in the `notebooks` folder.

Model classes are in the `models` folder.
