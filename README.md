# Longformer for Sentiment Analysis

Longformer for Sentiment Analysis applies the Longformer model to the IMDB dataset. 

## Introduction

This work uses the GraphPY framework to integrate a CUDA kernel with Python code. The Longformer uses a novel attention mechanism that scales linearly with sequence length, enabling the processing of long documents with thousands of tokens. For more information, see [Longformer: The Long-Document Transformer](https://github.com/allenai/longformer). 

## Requirements

- torch
- transformers
- tqdm
- pytorch-warmup
- tvm

For installing the required packages, run the `pip install -r requirements.txt` command.

Installing tvm from source with CUDA support:
```bash
sudo apt -y install llvm-10
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
mkdir build
cp cmake/config.cmake build
echo "set(USE_CUDA ON)" | tee -a build/config.cmake
echo "set(USE_LLVM ON)" | tee -a build/config.cmake
cd build
cmake ..
make -j4
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

## Usage

To fine-tune and test Longformer using the GraphPY, use the `python main.py` command. The --attention_mode flag changes the attention mode.

To predict the sentiment of a review, run the `predict.py` script using the command: `python predict.py --sentences [enter the review]`

More options are available in the help list:

`python train.py --help`
`python predict.py --help`

The cuda attention kernel is in `longformer_util/deep-codegen/kernel.cu`.
 
For compiling the kernel, enter the following command in the `longformer_util/deep-codegen` directory:
```bash
mkdir build && cd build
cmake ..
make -j
cp graphpy.cpython-38-x86_64-linux-gnu.so ../
```

For more information, see [deep code-gen information](longformer_util/deep-codegen/README.md).
