import torch
import time
from longformer_util.diagonaled_mm_tvm import diagonaled_mm
from torch.profiler import profile, record_function, ProfilerActivity

import importlib
lformerMM = importlib.import_module("longformer_util.deep-codegen.pytorch_apis").lformerMM

import argparse

default_input1 = [2, 4096, 12, -1]
default_input2 = [2, 4096, 12, 64]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', default='dcg', help='valid options: [tvm, dcg]')
    parser.add_argument('--mode', type=int, default=1, help='valid options: [1, 3]')
    parser.add_argument('--input1_dimensions', type=int, nargs='+', default=None, help='enter four digits')
    parser.add_argument('--input2_dimensions', type=int, nargs='+', default=None, help='enter four digits')
    parser.add_argument('--dilation', type=int, nargs='+', default=None, help='enter heads dimensions')
    parser.add_argument('--window', type=int, default=256)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--autoregressive', default=False, action='store_true')
    parser.add_argument('--coalesced', default=False, action='store_true')
    parser.add_argument('--chk', default=False, action='store_true')

    args = parser.parse_args()
    kernel = args.kernel
    mode = args.mode
    input1_dimensions = args.input1_dimensions
    input2_dimensions = args.input2_dimensions
    dilation = args.dilation
    window = args.window
    padding = args.padding
    autoregressive = args.autoregressive
    window_upper = 0 if autoregressive else window
    coalesced = 1 if args.coalesced else 0

    if mode != 1 and mode != 3:
        raise ValueError('Forward step has mode 1 and 3')

    is_diagonal = False if mode == 3 else True

    if input1_dimensions == None:
        input1_dimensions = default_input1

    if input2_dimensions == None:
        input2_dimensions = default_input2

    if len(input1_dimensions) != 4 or len(input2_dimensions) != 4:
        raise ValueError('Inputs should be 4D')

    if input1_dimensions[:3] != input2_dimensions[:3]:
        raise ValueError('First three dimensions should be the same')

    if mode == 1:
        input1_dimensions[3] = window + window_upper + 1
    else:
        input1_dimensions[3] = input2_dimensions[3]

    if dilation == None:
        dilation = [1]*input1_dimensions[2]

    if len(dilation) == 1:
        dilation = dilation*input1_dimensions[2]

    if len(dilation) != input1_dimensions[2]:
        raise ValueError('Dilation should be equal to heads')

    input1 = torch.rand(input1_dimensions, requires_grad=True, device=device)
    input2 = torch.rand(input2_dimensions, requires_grad=True, device=device)
    dilation = torch.tensor(dilation, dtype=torch.int, requires_grad=False, device=device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        if kernel == 'dcg':
            output1 = lformerMM(input1, input2, window, dilation, is_diagonal, padding, autoregressive, coalesced)
        else:
            output1 = diagonaled_mm(input1, input2, window, dilation, is_diagonal, padding, autoregressive)
        random_target = torch.rand_like(output1, device=device)
        loss = (output1 - random_target).pow(2).mean()
        loss.backward()
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

    if (args.chk == True) and (kernel == 'dcg'):
        output2 = diagonaled_mm(input1, input2, window, dilation, is_diagonal, padding, autoregressive)
        loss = (output1 - output2).pow(2).mean()
        if loss < (10 ** -6): print('dcg and tvm outputs are matched')
        else: print('dcg and tvm outputs are not matched')


