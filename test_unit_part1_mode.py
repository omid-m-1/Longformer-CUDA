import torch
from longformer_util.diagonaled_mm_tvm_mode import diagonaled_mm
from torch.profiler import profile, ProfilerActivity
import importlib
import pickle
import argparse
import os

lformerMM = importlib.import_module("longformer_util.deep-codegen.pytorch_apis_mode").lformerMM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--kernel', default='dcg', help='valid options: [tvm, dcg, dense]')
parser.add_argument('--type', default='seq', help='valid options: [seq, batch, window, dilation]')
args = parser.parse_args()

# parameter ranges
sequence_lengths = range(512, 16385, 512) if args.type == 'seq' else [4096]
batch_sizes = range(1, 17) if args.type == 'batch' else [2]
head_numbers = range(12, 25, 12)
window_lengths = range(64, 1025, 64) if args.type == 'window' else [256]
dilations = range(1, 5)
kernels = [args.kernel]

if args.kernel not in ['tvm', 'dcg', 'dense']:
    raise ValueError(f'{args.kernel} kernel is not defined')

if args.type not in ['seq', 'batch', 'window', 'dilation']:
    raise ValueError(f'{args.type} type is not defined')

running_times = []
for batch_size in batch_sizes:
    for head_number in head_numbers:
        for window_length in window_lengths:
            for dilation in dilations:
                for sequence_length in sequence_lengths:
                    for kernel in kernels:
                        for mode in [1, 2, 3]:
                            input1_dimensions = [batch_size, sequence_length, head_number, -1]
                            input2_dimensions = [batch_size, sequence_length, head_number, 64]
                            padding = 0
                            autoregressive = False
                            window_upper = 0 if autoregressive else window_length
                            is_diagonal = False if mode == 3 else True
                            if mode != 3:
                                if kernel != 'dense':
                                    input1_dimensions[3] = window_length + window_upper + 1
                                else:
                                    input1_dimensions[3] = sequence_length
                            else:
                                input1_dimensions[3] = input2_dimensions[3]
                            try:
                                input1 = torch.rand(input1_dimensions, requires_grad=True, device=device).contiguous()
                                input2 = torch.rand(input2_dimensions, requires_grad=True, device=device).contiguous()
                                dilation_tensor = torch.tensor([dilation]*head_number, dtype=torch.int, requires_grad=False, device=device)
                                if kernel == 'dcg':
                                    output = lformerMM(input1, input2, window_length, dilation_tensor, is_diagonal, autoregressive, mode)
                                    output = lformerMM(input1, input2, window_length, dilation_tensor, is_diagonal, autoregressive, mode)
                                elif kernel == 'tvm':
                                    output = diagonaled_mm(input1, input2, window_length, dilation_tensor, is_diagonal, padding, autoregressive, mode)
                                    output = diagonaled_mm(input1, input2, window_length, dilation_tensor, is_diagonal, padding, autoregressive, mode)
                                else:
                                    if mode != 3:
                                        output = torch.einsum('bxhy,byhd->bxhd', (input1, input2))
                                        output = torch.einsum('bxhy,byhd->bxhd', (input1, input2))
                                    else:
                                        output = torch.einsum('bxhd,byhd->bxhy', (input1, input2))
                                        output = torch.einsum('bxhd,byhd->bxhy', (input1, input2))
                                with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                                    if kernel == 'dcg':
                                        output = lformerMM(input1, input2, window_length, dilation_tensor, is_diagonal, autoregressive, mode)
                                    elif kernel == 'tvm':
                                        output = diagonaled_mm(input1, input2, window_length, dilation_tensor, is_diagonal, padding, autoregressive, mode)
                                    else:
                                        if (mode != 3) :
                                            output = torch.einsum('bxhy,byhd->bxhd', (input1, input2))
                                        else:
                                            output = torch.einsum('bxhd,byhd->bxhy', (input1, input2))
                                cuda_time = sum([event.cuda_time_total for event in prof.key_averages()])
                                cpu_time = sum([event.cpu_time_total for event in prof.key_averages()])
                                running_times.append([kernel, mode, batch_size, head_number, window_length, dilation, sequence_length, f'{cuda_time}us', f'{cpu_time}us'])
                            except RuntimeError as e:
                                if 'out of memory' in str(e):
                                    running_times.append([kernel, mode, batch_size, head_number, window_length, dilation, sequence_length, None, None])
                                else:
                                    raise e
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

os.makedirs('./test', exist_ok=True)
with open(f'./test/list_{args.type}_{args.kernel}.pkl', 'wb') as f:
    pickle.dump(running_times, f)
