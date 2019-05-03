# learning-relevent-tensor-networks
We provide an efficient, multiprocess implementation of the algorithm proposed in [Stoudenmires 2018](https://iopscience.iop.org/article/10.1088/2058-9565/aaba1a/meta). 

## How to use it

The algorithm contains two phases:
1. Unsupervised coarse graining: This allows to construct the tree tensor layer U. 

`python -m experiments.driver_ucg --{arg1}={val1} ...`

2. Supervised optimization of the top tensor: Once we have computed U, we just need to optimize the top tensor.

`python -m experiments.driver_train_top --{arg1}={val1} ...`
