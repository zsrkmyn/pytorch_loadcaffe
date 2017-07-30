#
# Copyright (c) 2017 Stephen Zhang <zsrkmyn at gmail dot com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to  use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import torch.nn as nn
from . import caffe_pb2

def CONVOLUTION(layer):
  param = layer.convolution_param
  groups = param.group
  b = layer.blobs[0]
  if b.HasField('shape'):
    in_channels = b.shape.dim[1]
  else:
    in_channels = layer.blobs[0].channels
  out_channels = param.num_output
  k_w = param.kernel_w
  k_h = param.kernel_h
  s_w = param.stride_w
  s_h = param.stride_h
  pad_w = param.pad_w
  pad_h = param.pad_h

  if k_w == 0 or k_h == 0:
    k_w = k_h = param.kernel_size
  if s_w == 0 or s_h == 0:
    s_w = s_h = param.stride
  if pad_w == 0 and pad_h == 0:
    pad_h = pad_w = param.pad

  return nn.Conv2d(
    in_channels, out_channels,
    (k_h, k_w), (s_h, s_w), 1, groups)

def POOLING(layer):
  param = layer.pooling_param
  k_w = param.kernel_w
  k_h = param.kernel_h
  s_w = param.stride_w
  s_h = param.stride_h
  pad_w = param.pad_w
  pad_h = param.pad_h

  if k_w == 0 or k_h == 0:
    k_w = k_h = param.kernel_size
  if s_w == 0 or s_h == 0:
    s_w = s_h = param.stride
  if pad_w == 0 and pad_h == 0:
    pad_h = pad_w = param.pad

  pool_type = caffe_pb2.PoolingParameter.PoolMethod.Value
  if param.pool == pool_type('MAX'):
    pool = nn.MaxPool2d
  elif param.pool == pool_type('AVG'):
    pool = nn.AvgPool2d
  else: # param.pool == pool_type('STOCHASTIC'):
    raise NotImplementedError(
      caffe_pb2.PoolingParameter.PoolMethod.Name(param.pool)
      + ' not implemented')

  return pool((k_h, k_w), (s_h, s_w), (pad_h,pad_w))

def RELU(layer):
  return nn.ReLU()

def TANH(layer):
  return nn.Tanh()

def SIGMOID(layer):
  return nn.Sigmoid()

def INNER_PRODUCT(layer):
  param = layer.inner_product_param
  b = layer.blobs[0]
  if b.HasField('shape'):
    n_input = b.shape.dim[0]
  else:
    n_input = b.width
  n_output = param.num_output
  return nn.Linear(n_input, n_output)

def DROPOUT(layer):
  return nn.Dropout(layer.dropout_param.dropout_ratio)

def SOFTMAX_LOSS(layer):
  return nn.Softmax()

def SOFTMAX(layer):
  return nn.Softmax()

# taken from caffe.proto
v1_layer_loaders = {
  4:  CONVOLUTION,
  6:  DROPOUT,
  14: INNER_PRODUCT,
  17: POOLING,
  18: RELU,
  19: SIGMOID,
  20: SOFTMAX,
  21: SOFTMAX,
  23: TANH,
  #0:  None, # NONE
  #35: None, # ABSVAL
  #1:  None, # ACCURACY
  #30: None, # ARGMAX
  #2:  None, # BNLL
  #3:  None, # CONCAT
  #37: None, # ONTRASTIVE_LOSS
  #5:  None, # DATA
  #39: None, # DECONVOLUTION
  #32: None, # DUMMY_DATA
  #7:  None, # EUCLIDEAN_LOSS
  #25: None, # ELTWISE
  #38: None, # EXP
  #8:  None, # FLATTEN
  #9:  None, # HDF5_DATA
  #10: None, # HDF5_OUTPUT
  #28: None, # HINGE_LOSS
  #11: None, # IM2COL
  #12: None, # IMAGE_DATA
  #13: None, # INFOGAIN_LOSS
  #15: None, # LRN
  #29: None, # MEMORY_DATA
  #16: None, # MULTINOMIAL_LOGISTIC_LOSS
  #34: None, # MVN
  #26: None, # POWER
  #27: None, # SIGMOID_CROSS_ENTROPY_LOSS
  #36: None, # SILENCE
  #22: None, # SPLIT
  #33: None, # SLICE
  #24: None, # WINDOW_DATA
  #31: None, # THRESHOLD
}

v2_layer_loaders = {
  "Convolution": CONVOLUTION,
  "Pooling": POOLING,
  "ReLU": RELU,
  "Sigmoid": SIGMOID,
  "Tanh": TANH,
  "InnerProduct": INNER_PRODUCT,
  "Dropout": DROPOUT,
  "SoftmaxWithLoss": SOFTMAX,
  "Softmax": SOFTMAX,
}

