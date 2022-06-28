# Rethinking Exponential Averaging of the Fisher: LM-KLD-WRM algorithms
Codes accompanying the paper: ``Rethinking Exponential Averaging of the Fisher'' (in arXiv https://arxiv.org/abs/2204.04718; upcoming in ECML PKDD 2022). These are three instantiations (SO-KLD-WRM, Q-KLD-WRM, and QE-KLD-WRM) of a particular sub-class (LM-KLD-WRM) of the proposed family (KLD-WRM). In principle, these algorithms can be implemented on other ``NG platforms'' too, but we use K-FAC as an ``implementation platform'' (see the paper and supplementary material for more details). The benchmark (K-FAC) is also considered.


## Citation
If you use the  code provided here, or any concept/results in the associated paper, please cite:

*[1] Puiu, C. O.; Rethinking Exponential Averaging of the Fisher, in arXiv:2204.04718 2022.*

## Requirements
* Main files are standard Jupyter notebooks so will require an appropriate package. Works best with Google Colab and Google Drive (otherwise some editing is required)

* Python 3 (http://www.python.org/)

In addition, you will require the following python packages: 

 * pytorch >= 1.11

 * torchvision >= 0.12.0

 * numpy >= 1.20

 * os, time, math of arbitrary versions

## Installation:
Manual: download, install required packages and run jupyter notebooks. Very simple if you just choose to use Google Colab.

## Documentation
* The main files are the jupyter files, one per solver considered in table 1 of the paper.

* The solvers (optimizers) themselves are placed in the Lib_files folder

* optimizers implementation constructed by starting from the codes of https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/kfac.py. Thanks!

## License
This algorithm is released under the GNU GPL v2.0 license.
