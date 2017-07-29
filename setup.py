from setuptools import setup, find_packages, Command
import distutils.command.build

import subprocess
import os

class build_pre(Command):
  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    src_path = os.path.join(dir_path, 'src')
    out_path = os.path.join(src_path, 'loadcaffe/__internal')
    proto_path = os.path.join(src_path, 'caffe.proto')
    cmd = ['protoc',
           '--python_out', out_path,
           '-I', src_path,
           proto_path]
    subprocess.call(cmd)

class build(distutils.command.build.build):
  sub_commands = [
    ('build_pre', lambda self: True),
  ] + distutils.command.build.build.sub_commands

setup(
  name='loadcaffe',
  description='A library to convert caffe model to PyTorch model',
  version='0.1',
  author='Stephen Zhang',
  author_email='zsrkmyn@gmail.com',
  license='MIT',
  url='https://github.com/zsrkmyn/pytorch_loadcaffe',
  packages=find_packages('src'),
  install_requires = [
    'protobuf',
    'torch',
  ],
  cmdclass = {
    'build': build,
    'build_pre': build_pre,
  },
  package_dir = {
    '': 'src'
  },
)

