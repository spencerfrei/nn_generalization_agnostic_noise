This is a repository for the code used for the experiments in the paper ["Provable Generalization of SGD-trained Neural Networks of Any Width
in the Presence of Adversarial Label Noise"](https://arxiv.org/abs/2101.01152) by Spencer Frei, Yuan Cao, and Quanquan Gu (accepted at ICML 2021).  

If you use this code or find it useful, please consider citing:
```
@inproceedings{frei2021generalization.nn.label.noise,
  title={Provable Generalization of SGD-trained Neural Networks of Any Width
in the Presence of Adversarial Label Noise},
  author={Frei, Spencer and Cao, Yuan and Gu, Quanquan},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2021}
}
```

Included in the code are two Jupyter notebooks that form the basis for all of our experiments.
# `experiment.{ipynb,py}`
This contains the main code used to run our experiments.  It allows for modularity to consider different hyperparameter configurations (different activation functions, network width, learning rate, sample size, using bias terms, etc.).  You can consider other architectures than one-hidden-layer networks by simply swapping out the one_hlayer_learner() with an equivalent Keras model class and changing the line where one_hlayer_learner() is defined.

We also provide this code as a stand-alone Python script `experiment.py` 

# `decision_boundary.{ipynb,py}`
After using `experiment.{ipynb,py}`, this notebook/script plots the decision boundary.  Note that you need to use at least `num_runs_per_error = 4` in `experiment.py`, as well as ensure that the set of `errors` considered in `experiment.py` includes 0.1, 0.25, and 0.40.  (This is accomplished with the default option of `num_optlin = 3`, or `num_optlin = 25`.)

# Package requirements and runtime etc.
All experiments were run using z1d (CPU) instances on AWS, using the latest Ubuntu Deep Learning AMI with TensorFlow 2.1+. The packages used include:
* tensorflow 2.1+
* numpy
* scipy
* datetime
* pandas
* plotnine
* matplotlib

The baseline experiment for a given hyperparameter configuration with width m=1,000 and T=20,000 online SGD updates evaluated over a grid of 25 values of optlin and 10 randomizations (over the initialization of the weights and the sequence of data observed by SGD) takes approximately 8 hours on the z1d instance.  
