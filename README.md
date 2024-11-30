# Neural Topic Model with Distance Awareness


## Run NSTM

Simply ```python nstm_tSNE_tfidf.py --dataset=20News --K=100```. For other settings, please see ```nstm_tSNE_tfidf.py``` as well as our paper.

The files of a run of the model will be saved under the folder of ```save```. 

## Evaluation

- We provided the Matlab functions (Matlab and Java installations required) to compute topic diversity (with all the topics), top-Purity/NMI and km-Purity/MNI for document clustering.
- Run ```evaluation/evaluate.m``` will give the above results.
- To evaluate the topic coherence results, we used [Palmetto](https://github.com/dice-group/Palmetto), which is not provided in this repo. One needs to download and set up separately.

