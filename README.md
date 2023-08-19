# Faster Training of Neural ODEs Using Gau&#223;--Legendre Quadrature

**Please note: this is research code that is not actively maintained, if you find any issues please let us know**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/a-norcliffe/torch_gq_adjoint/blob/master/LICENSE.txt)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Paper](https://img.shields.io/badge/paper-TMLR%202023-red)](https://openreview.net/forum?id=f0FSDAy1bU&referrer=%5BTMLR%5D(%2Fgroup%3Fid%3DTMLR))
[![Video](https://img.shields.io/badge/video-YouTube-green)](https://www.youtube.com/watch?v=pKbLwsqy8aM)

Public code for the TMLR 2023 paper [**Faster Training of Neural ODEs Using Gau&#223;--Legendre Quadrature**](https://openreview.net/forum?id=f0FSDAy1bU&referrer=%5BTMLR%5D(%2Fgroup%3Fid%3DTMLR))
(
[Alexander Norcliffe](https://twitter.com/alexnorcliffe98), 
[Marc Deisenroth](https://www.deisenroth.cc/)
)

**We recommend reading through this documentation fully before using the code.**


## Abstract
Neural ODEs demonstrate strong performance in generative and time-series modelling. However, training them via the adjoint method is slow compared to discrete models due to the requirement of numerically solving ODEs. To speed neural ODEs up, a common approach is to regularise the solutions. However, this approach may affect the expressivity of the model; when the trajectory itself matters, this is particularly important. In this paper, we propose an alternative way to speed up the training of neural ODEs. The key idea is to speed up the adjoint method by using Gau&#223;--Legendre quadrature to solve integrals faster than ODE-based methods while remaining memory efficient. We also extend the idea to training SDEs using the Wong--Zakai theorem, by training a corresponding ODE and transferring the parameters. Our approach leads to faster training of neural ODEs, especially for large models. It also presents a new way to train SDE-based models.


# Requirements
To run all the code in the library you will the majority of the standard packages
for machine learning in python. Including but not limited to:

- numpy
- torch
- matplotlib
- scipy
- sklearn

Importantly the library also requires the use of these non standard libraries:

- torchdiffeq
- sdeint
- TorchDiffEqPack
- torchsde

We recommend using a virtual environment to install the packages.



# Reproducing Experiments

## Datasets
To create the data in the paper run the following commands:
```bash
$ python -m datasets.make_nested_spheres_data
$ python -m datasets.make_sine_data
$ python -m datasets.make_ou_data
```

## Running Experiments
Experiments are run from the command line from the home directory. Each experiment
from the paper has its own file in the ```experiments``` directory. For example to run the
nested spheres experiment run the following command:
```bash
$ python -m experiments.nested_spheres --adjoint_option adjoint_gq --experiment_no 1 --width 2000 --lr 0.0001
```
In the above, the ```adjoint_option``` parameter is used to select the method of backprop to use,
```experiment_no``` is used to create the seed for consistency across methods, increase this
to run the experiment multiple times. ```width``` is used to set the width of the model,
and ```lr``` is used to set the learning rate. The model hyperparameters need to match those listed in
the appendix of the paper.

Each experiment has its own specific arguments to run from the command line. Therefore
we recommend looking at the code for each experiment to see the arguments that are required.
Another example is the sine experiment, requiring this command:
```bash
$ python -m experiments.sines --adjoint_option adjoint_ode --experiment_no 1 --width 1000 --lr 0.00003 --regular_times regular
```

The difference is that the adjoint option has changed from ```adjoint_gq``` to ```adjoint_ode```,
so now we use the standard adjoint method rather than the GQ method. The ```regular_times```
parameter is used to select whether the time-series is regularly spaced or irregularly spaced.

After running the experiments, the results are saved in the ```results``` directory, the plotting code (NOTE below about plotting code)
can be used to create the figures in the paper.

## Plotting Code
For the users benefit we have included the plotting code we used to generate the plots in the paper.
However, this was not written with the intention of being used by others, so it
is not well documented and may be difficult to use. We have not tested it since writing the
paper.


## Scaling Experiments
Running each experiment individually is not efficient. We recommend writing bash scripts
to run the experiments if running all of them with many repeats.


# General Use of the Library

The models in this repository are designed to be used in the experiments and have been
built as such. They are not designed to be used as a general Neural ODE. We recommend
using just the torch_gq_adjoint part of the library, since it has been designed to
be used in the same way the torchdiffeq library is used.

## Number of Function Evaluations
All nn modules used as the function in the Neural ODE must have a `nfe` attribute.
This is used to count the number of function evaluations, to calculate the number of
terms in the quadrature calculation. To do this in the __init__ of the module include the
line:
```bash
self.nfe = 0
```
And then in the forward method include:
```bash
self.nfe += 1
```

## gtol
The ```gtol``` parameter is used to determine the number of terms in the quadrature calculation.
It takes the place of $C$ from the paper. **NOTE:** This is different to ```rtol``` and ```atol```
parameters, these are used to determine the tolerance of the ODE solver, the lower these
numbers, the more steps are used in the solve. The ```gtol``` parameter is used to determine
the number of terms in the quadrature calculation, the higher this number, the more terms are used.
**Please note the relationship is not inverse like ```atol``` and ```rtol```**.

## Example
Below we include an example of using the adjoint and the GQ methods to show the difference:

```bash
from torch_gq_adjoint import odeint_adjoint, odeint_adjoint_gq

output_adj = odeint_adjoint(model, x, t, rtol=1e-4, atol=1e-6)  # adjoint method
output_gq = odeint_adjoint_gq(model, x, t, gtol=0.1, rtol=1e-4, atol=1e-6)  # gq method
```

The difference is that the GQ method requries a ```gtol``` parameter, the default is $0.1$.
And the GQ method also **requires** the model to have a ```nfe``` attribute, as mentioned above.

We also include a more comprehensive example in the ```g1d_example.ipynb``` notebook.

## When to Use
As described in the paper, we recommend using the GQ method when there is a small state size,
for example, a small batchsize or small vector size. We also recommend using the GQ method
when the model has many parameters.

As described below, we also found that the speedups are more extreme on CPU than GPU, so when
in using limited resources we recommend using the GQ method as well.


# Tests for Gradient Accuracy

To test the accuracy of the gradients produced by the GQ method we repeat
the tests from the torchdiffeq library. To run the tests run the following command:
```bash
$ python -m gradient_tests.gradient_tests
```

The following fixed methods are included in the tests:
'euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams'

The following adaptive methods are included in the tests:
'dopri5', 'bosh3', 'adaptive_heun', 'dopri8'

**Please note**: The paper only tests the gradient speed up of the dopri5 solver,
the other solvers are included for completeness. We believe these speedups would also
be seen in the other solvers, but we have not tested this.


# Hardware

We found that the speed up of the GQ method becomes more apparent on less advanced hardware.
On a CPU the GQ method was significantly faster than the standard adjoint method, and
often faster than directly backpropagating. On the GPU that we used we found for large
models the GQ method scaled better than the adjoint method. So we recommend using the
appropriate method for the hardware you are using, as mentioned, the GQ method is 
more effective compared to the adjoint method on less advanced hardware.



# Citation
If you find this code or our paper useful on your own research, please cite our paper:
```
@article{norcliffe2023gq,
  title={{F}aster {T}raining of {N}eural {ODE}s {U}sing {G}au{\ss}{\textendash}{L}egendre {Q}uadrature},
  author={Norcliffe, Alexander and Deisenroth, Marc},
  journal={{T}ransactions on {M}achine {L}earning {R}esearch},
  year={2022}
}
```


# Acknowledgements
We thank the anonymous reviewers for their comments and suggestions for this paper.
At the time of this work, Alexander Norcliffe is supported by a GlaxoSmithKline grant.
A very large amount of this work was heavily dependent on the ```torchdiffeq``` library,
we thank the authors for their work on this library.