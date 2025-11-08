# DMbaGCN

This is the official implementation of the following paper:

Dual Mamba for Node-Specific Representation Learning: Tackling Over-Smoothing with Selective State Space Modeling

<div align="center">
  <img src="https://github.com/hexin5515/MbaGCN/blob/main/Image/MbaGCN.jpg" width="1600px"/>
</div>

## Environment Setup

**Required Dependencies** :

* torch>=2.1.2
* torch_geometric>=2.5.2
* python>=3.8
* scikit-learn>=1.4.1
* networkx>=2.7
* rdkit>=2024.3.2

## Quick Start

**CoraFull Dataset**

The main experiments:
```
cd NodeClassification/

python training.py --dataset Corafull --d_model 512 --d_inner 512 --dt_rank 64 --d_state 1 --mamba_dropout 0.2 --alpha 0.9 --graph_weight 0.8 --layer_num 3 --lr 0.005 --weight_decay 0. --net GCN_mamba_Net --runs 10
```
