## Repository Contents

This directory hosts the implementation of the CLEAR script adapted for the "MicroRec dataset". The following notebooks are included:

- `CLEAR00-dataset.ipynb`: Prepares the training, evaluation, and test datasets. To utilize the dataset corresponding to the one described in our paper, access the provided [data folder](https://drive.google.com/drive/folders/1Me739n00O33kHIAKkKdiQnaBwk320rFy?usp=sharing).

- `CLEAR01-biencoder.ipynb` and `CLEAR02-crossencoder.ipynb`: These notebooks are dedicated to model training and weight saving processes. To employ the pretrained models mentioned in our paper, download them from [this link](https://drive.google.com/drive/folders/1Me739n00O33kHIAKkKdiQnaBwk320rFy?usp=sharing).

- `CLEAR03-rerank.ipynb`: Executes the reranking of top *k* results to derive the final recommendation outcomes.

