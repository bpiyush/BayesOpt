## Scalable Bayesian Optimization

Bayesian Optimization is one of the most popular methods for optimizing expensive black-box functions. In this project, we attempt to understand some of the recent techniques for scaling Bayesian Optimization for large number of input data points. We also try some novel ideas and evaluations. Stay tuned for results and cleaned-up code!

#### Directory Structure

Here is the directory structure. You can access the code in the `/code` folder. 

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