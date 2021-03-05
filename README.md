# Knowledge Graph meets Collaborative Filtering Recommender Systems

```diff
@@ Currently in the Process of Cleaning and Updating ... @@
```


----

<img src="https://github.com/ArgentLo/X-2ch-Quad-Channel-Collaborative-Graph-Network-over-Knowledge-aware-Edges/blob/main/structure.png" width="720" height="192.6">


In this repository, all the following parts are included to **support reproductivity and reliablity** of the manuscript.

  - The proposed **X-2ch model**.
  - **All datasets** used in the paper.
  - **All baselines** in experiments.

----

### Double-Blind Submission

**X-2ch** is a novel graph-based model for recommendation, which is capable of learning representative user and item embeddings by distributing information over knowledge-aware edges through a quad-channel mechanism.

All models in experiment are summarized as follows:

- Bayesian Pairwise Reranking MF (BPR) : **included in this repository**.
- CKE : https://github.com/hexiangnan/neural_collaborative_filtering
- RippleNet : https://github.com/hwwang55/RippleNet

- KGCN : **included in this repository**.
- KGAT : https://github.com/xiangwang1223/knowledge_graph_attention_network
- CKAN: **included in this repository**.

- **X-2ch** : **included in this repository**.

----

### Environment Requirement

The code has been tested running under Python 3.5. The required packages are as follows:

```
tensorflow == 1.12.0
numpy
tqdm
networkx

# for CKAN model
torch==1.3.0
```

----

### Dataset

All datasets used in the experiments are provided. 

Since implicit feedback data are considered in our work, all **data values are binarized**. 

For all dataset, 80% of a user’s historical items would be randomly sampled as the training set and the rest items are collected as the test set.

Please **download** the preprocessed datasets and **save in `./data/`**.

- Last-FM

  ```
  https://drive.google.com/file/d/1_aTisDEXXdILbDa52GKijaA6aw8gfvjn/view
  ```

- Amazon-Book (277M):

  ```
  https://drive.google.com/file/d/17MBu5GJtJZOobY0RU3dIcqOSqG6W0yxM/view
  ```

----
