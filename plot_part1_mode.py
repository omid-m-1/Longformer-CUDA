import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import os

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
    return np.mean(cuda_times)/1000

kernels = ['tvm', 'dense', 'dcg']
modes = [1, 2, 3]
heads = [12, 24]
sequence_lengths = range(512, 16385, 512)
batch_sizes = range(1, 17)
window_lengths = range(64, 1025, 64)
dilations = range(1, 5)

colors = {'tvm': 'green', 'dense': 'blue', 'dcg': 'red'}
styles = {'tvm': '--', 'dense': '-', 'dcg': '-.'}
x_label = ['Sequence Length', 'Batch Size', 'Window Length', 'Dilation']
x_ticks = [np.arange(0, 16001, 4000), np.arange(1,9)*2, np.arange(0, 1001, 200), np.arange(1,5)]
y_ticks = [np.arange(0, 51, 10), np.arange(0, 101, 20), np.arange(0, 41, 10), np.arange(4, 15, 2)]

os.makedirs('./fig', exist_ok=True)
for mode in modes:
    times_seq = []
    times_batch = []
    times_window = []
    times_dilation = []
    for kernel in kernels:
        if kernel != 'dense':
            mode_list = [[1], [2], [3]]
        else:
            mode_list = [[1, 2], [1, 2], [3]]
        times_seq.append([average(list_seq, lambda record: record[0] == kernel and record[1] in mode_list[mode-1] and record[2] == 2 and record[3] in heads and
            record[4] == 256 and record[5] in dilations and record[6] == seq_len) for seq_len in sequence_lengths])
        times_batch.append([average(list_batch, lambda record: record[0] == kernel and record[1] in mode_list[mode-1] and record[2] == batch_size and record[3] in heads and
            record[4] == 256 and record[5] in dilations and record[6] == 4096) for batch_size in batch_sizes])
        times_window.append([average(list_window, lambda record: record[0] == kernel and record[1] in mode_list[mode-1] and record[2] == 2 and record[3] in heads and
            record[4] == window_len and record[5] in dilations and record[6] == 4096) for window_len in window_lengths])
        times_dilation.append([average(list_dilation, lambda record: record[0] == kernel and record[1] in mode_list[mode-1] and record[2] == 2 and record[3] in heads and
            record[4] == 256 and record[5] == dilation and record[6] == 4096) for dilation in dilations])
    times = [times_seq, times_batch, times_window, times_dilation]
    x_value = [sequence_lengths, batch_sizes, window_lengths, dilations]
    for fig_id in range(4):
        fig, axs = plt.subplots(figsize=(6, 4))
        for kernel_id in range(3):
            kernel = kernels[kernel_id]
            if any(time is not None for time in times[fig_id][kernel_id]):
                axs.plot(x_value[fig_id], times[fig_id][kernel_id], label=f'{kernel}', color=colors[kernel], linestyle=styles[kernel], linewidth=3)
        axs.set_xticks(x_ticks[fig_id])
        axs.set_yticks(y_ticks[fig_id])
        axs.set_xlabel(x_label[fig_id], labelpad=15)
        axs.set_ylabel('CUDA Time (milliseconds)', labelpad=15)
        axs.set_title(f'CUDA Running Time vs {x_label[fig_id]}: Mode {mode}', pad=20)
        axs.legend()
        plt.tight_layout()
        plt.savefig(f'./fig/{x_label[fig_id]}_Mode {mode}.pdf', bbox_inches='tight')
        plt.close(fig)
