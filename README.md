## Scalable Bayesian Optimization

Bayesian Optimization is one of the most popular methods for optimizing expensive black-box functions. In this project, we attempt to understand some of the recent techniques for scaling Bayesian Optimization for large number of input data points. We also try some novel ideas and evaluations. Stay tuned for results and cleaned-up code!

#### Directory Structure

Here is the directory structure. You can access the code in the `/code` folder. Please note that only the code files are added to Git due to space optimization. Other files could be made available on request.

```
BayesOpt/
├── code
│   ├── pybnn
│   │   ├── build
│   │   │   ├── bdist.linux-x86_64
│   │   │   └── lib
│   │   │       ├── pybnn
│   │   │       │   ├── sampler
│   │   │       │   └── util
│   │   │       └── test
│   │   ├── dist
│   │   ├── notebooks
│   │   ├── pybnn
│   │   │   ├── sampler
│   │   │   └── util
│   │   ├── pybnn.egg-info
│   │   └── test
│   ├── __pycache__
│   └── util
│       └── __pycache__
├── experiments
│   └── src
├── latex
└── papers
    ├── hyp_LDA
    └── hyp_LogReg

```

### Note

We have used the implementation of [`pybnn`](https://github.com/automl/pybnn) as a base model for Bayesian Linear Regression on the basis of J. Snoek et al [1]. The usual Bayesian Optimization routine with the neural-network based surrogate model has been implemented by us.

---
### Task 0: Testing the implementation on a Mathematical dataset

We test our implementation through a simple mathematical dataset which looks like this:

![alt text](https://github.com/bpiyush/BayesOpt/raw/master/plots/toy-dataset.png "Mathematical Dataset")

### Task 1: Bayesian Linear Regression on toy dataset

Stay tuned!

---

### Task 2: Logistic Regression on MNIST dataset

Stay tuned!
---




### References
1. J. Snoek et al, Scalable Bayesian Optimization using Deep Neural Networks