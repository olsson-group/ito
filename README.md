# ITO
This repository can reproduce the results shown for AlanineDipeptide in the paper "Implicit Transfer Operator Learning: Multiple Time-Resolution Surrogates for Molecular Dynamics" - https://arxiv.org/abs/2305.18046


## Installation

To install, run:

```
$ git clone https://gitlab.com/matschreiner/ito
$ cd ito
$ make install-{arch}
```

Replace {arch} with cpu, cu118, or cu121 dependent on which architecture you will be running on. This will install the package-dependencies as well as collect and install the appropriate wheels for pytorch-scatter, pytorch-sparse, and pytorch and pytorch-cluster used by pytorch-geometric.


## Usage
Scripts for training models and sampling trajectories are saved in scripts. 

To train a model TLDDPM model run 

```
python scripts/train_tlddpm.py
```

There are a few arguments available for this training script.

```
optional arguments:
  -h, --help            show this help message and exit
  --path PATH           Base directory for storing data and checkpoints.
  --n_features N_FEATURES
                        Number of features for the score model.
  --n_layers N_LAYERS   Number of layers in the score model.
  --epochs EPOCHS       Number of training epochs.
  --diff_steps DIFF_STEPS
                        Number of diffusion steps in the model.
  --batch_size BATCH_SIZE
                        Batch size for training.
  --lr LR               Learning rate for the optimizer.
  --max_lag MAX_LAG     Maximum lag to consider in the ALA2 dataset.
  --fixed_lag           Enable to use a fixed lag value; disable for variable lag.
  --indistinguishable   Enable this flag to treat atoms as indistinguishable.
  --unscaled            Use unscaled data. When disabled, data is scaled to unit variance.
```










