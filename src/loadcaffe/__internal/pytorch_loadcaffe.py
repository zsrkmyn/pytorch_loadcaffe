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

import google.protobuf as protobuf
import torch
import torch.nn as nn
from . import layers
from . import caffe_pb2

def __convert(netparam):
  if netparam.layer:
    # Use LayerParameter, see caffe.proto L82
    layer_loaders = layers.v2_layer_loaders
    type_str = lambda x: x
  else:
    # Use V1LayerParameter, see caffe.proto L85
    layer_loaders = layers.v1_layer_loaders
    type_str = lambda x: caffe_pb2.V1LayerParameter.LayerType.Name(x)

  net = nn.Sequential()

  for layer in netparam.layers:
    loader = layer_loaders.get(layer.type)
    print(layer.name + ': ' + type_str(layer.type))
    if loader is None:
      raise NotImplementedError(type_str(layer.type) + ' not implemented')

    module = loader(layer)

    if layer.blobs:
      # slow, blobs_data -> list -> FloatTensor
      # but I cannot find a more efficient way using Python
      module.weight.data = torch.FloatTensor(layer.blobs[0].data[:])
      module.bias.data = torch.FloatTensor(layer.blobs[1].data[:])

    net.add_module(layer.name, module)

  return net

def __open_read(f):
  if isinstance(f, str):
    with open(f, 'rb') as fin:
      f = fin.read()
  else:
    f = f.read()
  return f


def load(proto):
  """Load trained caffe model.

  Parameters:
    proto: a str describing the model file path or an opened file stream

  Returns:
    a torch.nn.Sequential containing the loaded model
  """
  proto = __open_read(proto)
  netparam = caffe_pb2.NetParameter()
  netparam.ParseFromString(proto)
  net = __convert(netparam)
  return __convert(netparam)
