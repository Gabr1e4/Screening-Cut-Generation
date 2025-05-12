## Screening Cut Generation for Sparse Ridge Regression
This is the official implementation of the paper: [Screening Cut Generation for Sparse Ridge Regression](https://arxiv.org/abs/2505.01082). The repository provides code to reproduce the results and experiments described in the paper.

## Package Dependencies
The code is implemented in Python. The required dependencies are listed in `requirements.txt`.

## Usage
The example usage of SCG can be found in `numerical-test/ridge-regression/example.ipynb`. To test the implementation on synthetic datasets and real datasets, you can run the script located in `numerical-test/ridge-regression/ridge-test-synthetic.ipynb` and `numerical-test/ridge-regression/ridge-test-real.ipynb`, respectively.

Feel free to modify the script to adapt to your specific use case.

**NOTE**: In `numerical-test/ridge-regression/ridge-test-real.ipynb`, we adopt Windows syntax to read the real dataset. Please adapt accordingly with your OS syntax.



<!-- ## Model Reformulation
The inner optimization problem can be simplied. Assume $\hat{\beta}$ and $\hat{z}$ are given, then the objective becomes

$\sum_{i \in Supp(\hat{z})} (\boldsymbol{x}_i^T \boldsymbol{\hat{\beta}} - y_i - \boldsymbol{\Delta}_i)^2 + \sum_{i \notin Supp(\hat{z})}(\boldsymbol{x}_i^T \boldsymbol{\hat{\beta}} - y_i)^2$. -->

