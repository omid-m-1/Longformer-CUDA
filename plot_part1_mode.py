import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size': 14})

with open('./test/list_seq_dcg.pkl', 'rb') as f:
    list_seq_dcg = pickle.load(f)
with open('./test/list_seq_dense.pkl', 'rb') as f:
    list_seq_dense = pickle.load(f)
with open('./test/list_seq_tvm.pkl', 'rb') as f:
    list_seq_tvm = pickle.load(f)
list_seq = []
list_seq += list_seq_dcg + list_seq_dense + list_seq_tvm

with open('./test/list_batch_dcg.pkl', 'rb') as f:
    list_batch_dcg = pickle.load(f)
with open('./test/list_batch_dense.pkl', 'rb') as f:
    list_batch_dense = pickle.load(f)
with open('./test/list_batch_tvm.pkl', 'rb') as f:
    list_batch_tvm = pickle.load(f)
list_batch = []
list_batch += list_batch_dcg + list_batch_dense + list_batch_tvm

with open('./test/list_window_dcg.pkl', 'rb') as f:
    list_window_dcg = pickle.load(f)
with open('./test/list_window_dense.pkl', 'rb') as f:
    list_window_dense = pickle.load(f)
with open('./test/list_window_tvm.pkl', 'rb') as f:
    list_window_tvm = pickle.load(f)
list_window = []
list_window += list_window_dcg + list_window_dense + list_window_tvm

with open('./test/list_dilation_dcg.pkl', 'rb') as f:
    list_dilation_dcg = pickle.load(f)
with open('./test/list_dilation_dense.pkl', 'rb') as f:
    list_dilation_dense = pickle.load(f)
with open('./test/list_dilation_tvm.pkl', 'rb') as f:
    list_dilation_tvm = pickle.load(f)
list_dilation = []
list_dilation += list_dilation_dcg + list_dilation_dense + list_dilation_tvm

def average(data, filter_condition):
    filtered_data = [record for record in data if filter_condition(record)]
    if not filtered_data:
        return None
    cuda_times = [float(record[7].replace('us', '')) for record in filtered_data if record[7] is not None]
    if len(cuda_times) != len(filtered_data):
        return None
    return np.mean(cuda_times)

def plot_for_mode(ax, data, seq_len_func, batch_size_func, window_len_func, dilation_func, mode):
    for kernel in kernels:
        times_seq_len = [seq_len_func(kernel, mode, seq_len) for seq_len in sequence_lengths]
        times_batch_size = [batch_size_func(kernel, mode, batch_size) for batch_size in batch_sizes]
        times_window_len = [window_len_func(kernel, mode, window_len) for window_len in window_lengths]
        times_dilation = [dilation_func(kernel, mode, dilation) for dilation in dilations]

kernels = ['tvm', 'dense', 'dcg']
heads = [12, 24]
dilations = range(1, 5)
colors = {1: 'blue', 2: 'green', 3: 'red'}
styles = {'tvm': '--', 'dense': '-', 'dcg': '-.'}

fig, axs = plt.subplots(2, 2, figsize=(14, 14))
# Plot 1: Sequence Length
sequence_lengths = range(512, 16385, 512)
for kernel in kernels:
    if kernel != 'dense':
        modes = [[1], [2], [3]]
    else:
        modes = [[1, 2], [3]]
    for mode in modes:
        times = [average(list_seq, lambda record:
            record[0] == kernel and
            record[1] in mode and
            record[2] == 2 and
            record[3] in heads and
            record[4] == 256 and
            record[5] in dilations and
            record[6] == seq_len)
            for seq_len in sequence_lengths]
        if any(time is not None for time in times):
            axs[0, 0].plot(sequence_lengths, times, label=f'{kernel} Mode {mode[0]}', color=colors[mode[0]], linestyle=styles[kernel], linewidth = 3)
axs[0, 0].set_xlabel('Sequence Length', labelpad=20)
axs[0, 0].set_ylabel('CUDA Time (microseconds)', labelpad=20)
axs[0, 0].set_title('CUDA Running Time vs Sequence Length', pad=20)
# Plot 2: Batch Size
batch_sizes = range(1, 17)
for kernel in kernels:
    if kernel != 'dense':
        modes = [[1], [2], [3]]
    else:
        modes = [[1, 2], [3]]
    for mode in modes:
        times = [average(list_batch, lambda record:
            record[0] == kernel and
            record[1] in mode and
            record[2] == batch_size and
            record[3] in heads and
            record[4] == 256 and
            record[5] in range(1, 5) and
            record[6] == 4096)
            for batch_size in batch_sizes]
        if any(time is not None for time in times):
            axs[0, 1].plot(batch_sizes, times, label=f'{kernel} Mode {mode[0]}', color=colors[mode[0]], linestyle=styles[kernel], linewidth = 3)
axs[0, 1].set_xlabel('Batch Size', labelpad=20)
axs[0, 1].set_ylabel('CUDA Time (microseconds)', labelpad=20)
axs[0, 1].set_title('CUDA Running Time vs Batch Size', pad=20)
# Plot 3: Window Length
window_lengths = range(64, 1025, 64)
for kernel in kernels:
    if kernel != 'dense':
        modes = [[1], [2], [3]]
    else:
        modes = [[1, 2], [3]]
    for mode in modes:
        times = [average(list_window, lambda record:
            record[0] == kernel and
            record[1] in mode and
            record[2] == 2 and
            record[3] in heads and
            record[4] == window_len and
            record[5] in dilations and
            record[6] == 4096)
            for window_len in window_lengths]
        if any(time is not None for time in times):
            axs[1, 0].plot(window_lengths, times, label=f'{kernel} Mode {mode[0]}', color=colors[mode[0]], linestyle=styles[kernel], linewidth = 3)
axs[1, 0].set_xlabel('Window Length', labelpad=20)
axs[1, 0].set_ylabel('CUDA Time (microseconds)', labelpad=20)
axs[1, 0].set_title('CUDA Running Time vs Window Length', pad=20)
# Plot 3: Dilation
#dilations = range(1, 5)
for kernel in kernels:
    if kernel != 'dense':
        modes = [[1], [2], [3]]
    else:
        modes = [[1, 2], [3]]
    for mode in modes:
        times = [average(list_dilation, lambda record:
            record[0] == kernel and
            record[1] in mode and
            record[2] == 2 and
            record[3] in heads and
            record[4] == 256 and
            record[5] == dilation and
            record[6] == 4096)
            for dilation in dilations]
        if any(time is not None for time in times):
            axs[1, 1].plot(dilations, times, label=f'{kernel} Mode {mode[0]}', color=colors[mode[0]], linestyle=styles[kernel], linewidth = 3)
axs[1, 1].set_xticks([1, 2, 3, 4])
axs[1, 1].set_xlabel('Dilation', labelpad=15)
axs[1, 1].set_ylabel('CUDA Time (microseconds)', labelpad=20)
axs[1, 1].set_title('CUDA Running Time vs Dilation Rate', pad=20)

kernel_lines = [Line2D([0], [0], color='black', linestyle=styles[kernel]) for kernel in kernels]
mode_lines = [Line2D([0], [0], color=colors[mode], linestyle='None', marker='o') for mode in range(1, 4)]
kernel_legend = fig.legend(handles=kernel_lines, labels=kernels, loc='lower right', bbox_to_anchor=(0.85, 0.125), title='Kernel Type', ncol=1)
mode_legend = fig.legend(handles=mode_lines, labels=[f'Mode {mode}' for mode in range(1, 4)], loc='lower right', bbox_to_anchor=(.975, 0.125), title='Mode Number', ncol=1)

plt.tight_layout()
plt.subplots_adjust(hspace=0.6, wspace=0.35, bottom=0.3)
plt.savefig('Figure2.pdf', bbox_inches='tight')
