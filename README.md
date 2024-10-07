# EPR_EMNLP24

This repository provides the source code of "EPR: An Expert Behavior-enhanced Paper Ranking Framework for the Automotive Industry" accepted in EMNLP 2024 Workshop CustomNLP4U, and V-Paper dataset.


## 1. Overview

We present EPR framework that effectively aggregates handcrafted expert-behavior oriented features and LLM based semantics, achieving high ranking performance.
We additionally release V-Paper, a paper collection on 12 topics in automotive domain.

![architecture](https://github.com/user-attachments/assets/52760abb-af46-4679-9b1d-faba148c54c5)


## 2. Main Results

![experiment](https://github.com/user-attachments/assets/6264fffc-ecd9-4b76-9cbf-8067a018f9db)

Our EPR outperforms other baselines on V-Paper dataset.


## 3. V-Paper

The file ``data/data.csv`` contains all information about the V-Paper dataset. It provides details for each paper, including related topics, EID, title, abstract, published year, and keywords. The other files within the ``data/`` folder can be obtained by appropriately preprocessing ``data/data.csv`` as described in our paper. To ensure anonymization of the expert labels, all papers have been mapped to random IDs, and their labels have been masked to 0.


## 4. Requirements

- Python version: 3.11.7
- Pytorch version: 2.1.2


## 5. Usage

You need to specify the backbone ranking model and activate all three features.
```
python3 train.py --model RankNet --content --keyword --expert
python3 train.py --model GSF     --content --keyword --expert
python3 train.py --model CAR     --content --keyword --expert
```
