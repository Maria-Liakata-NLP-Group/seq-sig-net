# Sequential Path Signature Networks for Personalised Longitudinal Language Modeling

This repository contains the code for the paper "Sequential Path Signature Networks for Personalised Longitudinal Language Modeling", accepted at ACL Findings 2023.

# Datasets

Due to the sensitive nature of the data, the datasets we use in the paper (TalkLife and Reddit) are not publically available. File `paths.py` contains user defined paths to the datasets and directories (that we exclude), but which you can use to source datasets for your own longitudinal task. 

The following columns are necessary for the input dataset: `timeline_id`, `label`, `datetime`, as well as a column that determines the train/dev/test splits (or a function that randomnly splits the dataset by timeline_id). Having a `postid` is optional, but recommended for clarity. 

The user also needs to provide a seperate file that contains the embeddings (here Sentence-BERT) of your choice. Here we produced Sentence-BERT embeddings using `'all-MiniLM-L6-v2'`. In order to reproduce such embeddings we recommend this [example](https://www.sbert.net/#usage).

# Installation

Clone the git repo:

```$ git clone git@github.com:Maria-Liakata-NLP-Group/seq-sig-net.git```

Create a conda environment:

```$ conda env create --file=environment.yml```

Actviate the conda environment

```$ conda activate py38-MoC```

# Models

Model notebooks are in the `notebooks` folder.

Model classes are in the `models` folder.

Details of the best hyperparameters of each model are in the Appendix section of the [paper](https://github.com/Maria-Liakata-NLP-Group/seq-sig-net/blob/main/paper.pdf).