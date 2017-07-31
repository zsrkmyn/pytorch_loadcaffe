PyTorch loadcaffe
=================
An unofficial library for PyTorch to load caffe model.

**NOTE**: The project is currently work-in-process, so there may be many bugs.
Use it at your own risk. Issues and Pull Requests are welcomed.

Requirements
------------
- PyTorch
- python-protobuf

Build Requirements
------------------
- protobuf

Installation
------------
.. code:: bash

    git clone https://github.com/zsrkmyn/pytorch_loadcaffe
    cd pytorch_loadcaffe
    sudo python setup.py install

Usage
-----
::

    >>> import loadcaffe
    >>> net = loadcaffe.load('some.caffemodel')

Troubleshooting
----------------

The load process is too slow
~~~~~~~~~~~~~~~~~~~~~~~~~~~
A potential reason is that your python-protobuf isn't built with
``--cpp_implementation`` options, which may use pure python code to parse
protobuf and cause really bad performance.
See `here <https://bugs.archlinux.org/task/54959>`_ for more information.

``NotImplementedError`` occurs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The project currently supports the following caffe layers:

- Convolution (2D only)
- Pooling (2D only)
- ReLU
- Sigmoid
- Tanh
- InnerProduct
- Dropout
- SoftmaxWithLoss
- Softmax

Pull requests are welcomed :-)

