MAPTrainer
==========

*MAPTrainer* is a Python framework based on the PyTorch library[^1]
dedicated to machine learning research in the context of MAP. As
PyTorch, it uses GPUs and CPUs. MAPTrainer is a native Python package by
design, called `maptrainer`. Its functionalities are built as Python
classes, which enable the integration of its code with Python packages
and modules. Relying on PyTorch, it enables GPU-accelerated training,
through CUDA, for machine learning, and especially neural network
applications. The objective of this framework is to train any machine
learning model, built with PyTorch, for any MAP scenario. Parameters
defining the MAP scenario are provided to the programme.

### `maptrainer.model`

`maptrainer.model` allows implementing any machine learning model in the
same way as PyTorch. Any new machine learning model which is intended to
be experimented shall be added in this package. Then, the model of this
class will be loaded by `maptrainer.model.modelhandler`, which will
check, if specified by the user, whether a previous trained version of
this model for the current MAP scenario exists. Otherwise, a new model
is created for the current scenario.

### `maptrainer.data`

This package adds support for loading MAP data necessary to train a
model. The data of a MAP scenario are loaded by means of a *data
loader*, which is merely an object of
`maptrainer.data.MAPDataLoader.MAPDataLoader` type. `MAPDataLoader` is
an abstract class defining a template for any data loader. A new data
loader shall extend this class and implements its methods. Over the
training stage, the data loader will then return data divided in
validation and training data, and for each type of data, into input and
output data. In execution, the data will then be returned fold after
fold if a $k$-fold cross-validation is required. As part of this work,
two concrete data loaders are developed:

-   `BPDataLoader`: loads *binary sequences*,

-   `IPDataLoader`: loads *on-vertex idleness sequences*.

### `maptrainer.train`

`maptrainer.train` is used to automatically train a model. A complete
training is run with the `maptrainer.train.run_epochs` method, which
runs a training for the loaded model, the criterion to optimise, the
data loader, the learning rate, and the number of epochs, among others.

### `maptrainer.valid`

`maptrainer.valid` is used to validate the model with respect to the
criterion to optimise, the data loader, the learning rate, and the
number of epochs, among others.

[^1]: pytorch.org.
