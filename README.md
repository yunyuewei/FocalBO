# Scalable Bayesian Optimization via Focalized Sparse Gaussian Processes

This repository contains the official implementation of [Scalable Bayesian Optimization via Focalized Sparse Gaussian Processes](https://openreview.net/pdf?id=OF0YsxoRai) accepted by NeurIPS 2024.

## Requirements

To run the experiments, it is expected that there is a Python environment with all the necessary dependencies. To install and run the VecchiaBO baseline, clone the original [VecchiaBO repository](https://github.com/feji3769/VecchiaBO.git) and run `pip install .` inside the code folder.

For the DKitty task, follow the environment setup rules from the original Github repository https://github.com/brandontrabucco/design-baselines. Note that it is best to use Python3.8 and do this in a separate environment, as conflicts between various Python packages may occur.

## Experiment Replication

To replicate the experiments outlined in the paper, run the following command:

```
bash scripts/run_{task}.sh {algo} {opt}
```

where the variables `task`, `algo`, and `opt` should be replaced with the desired task, algorithm, and optimization method. (The muscle task will be released soon.)

## BibTeX

```
@inproceedings{neurips2024focalbo,
  title={Scalable Bayesian Optimization via Focalized Sparse Gaussian Processes},
  author={Wei, Yunyue and Zhuang, Vincent and Soedarmadji, Saraswati and Sui, Yanan},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```