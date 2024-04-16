import pickle
import matplotlib.pyplot as plt
import numpy as np

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

kernels = ['tvm', 'dense', 'dcg']
modes = [1, 2, 3]
heads = [12, 24]
dilations = range(1,5)
colors = {'tvm': 'blue', 'dense': 'green', 'dcg': 'red'}
styles = {'tvm': '-', 'dense': '--', 'dcg': '-.'}

fig, axs = plt.subplots(2, 2, figsize=(14, 12))
# Plot 1: Sequence Length
sequence_lengths = range(512, 16385, 512)
for kernel in kernels:
    times = [average(list_seq, lambda record:
        record[0] == kernel and
        record[1] in modes and
        record[2] == 2 and
        record[3] in heads and
        record[4] == 256 and
        record[5] in dilations and
        record[6] == seq_len)
        for seq_len in sequence_lengths]
    if any(time is not None for time in times):
        axs[0, 0].plot(sequence_lengths, times, label=kernel, color=colors[kernel], linestyle=styles[kernel], linewidth = 3)
axs[0, 0].set_xlabel('Sequence Length')
axs[0, 0].set_ylabel('CUDA Time (microseconds)')
axs[0, 0].set_title('CUDA Running Time vs Sequence Length')
axs[0, 0].legend()
# Plot 2: Batch Size
batch_sizes = range(1, 17)
for kernel in kernels:
    times = [average(list_batch, lambda record:
        record[0] == kernel and
        record[1] in modes and
        record[2] == batch_size and
        record[3] in heads and
        record[4] == 256 and
        record[5] in range(1, 5) and
        record[6] == 4096)
        for batch_size in batch_sizes]
    if any(time is not None for time in times):
        axs[0, 1].plot(batch_sizes, times, label=kernel, color=colors[kernel], linestyle=styles[kernel], linewidth = 3)
axs[0, 1].set_xlabel('Batch Size')
axs[0, 1].set_ylabel('CUDA Time (microseconds)')
axs[0, 1].set_title('CUDA Running Time vs Batch Size')
axs[0, 1].legend()
# Plot 3: Window Length
window_lengths = range(64, 1025, 64)
for kernel in kernels:
    times = [average(list_window, lambda record:
        record[0] == kernel and
        record[1] in modes and
        record[2] == 2 and
        record[3] in heads and
        record[4] == window_len and
        record[5] in dilations and
        record[6] == 4096)
        for window_len in window_lengths]
    if any(time is not None for time in times):
        axs[1, 0].plot(window_lengths, times, label=kernel, color=colors[kernel], linestyle=styles[kernel], linewidth = 3)
axs[1, 0].set_xlabel('Window Length')
axs[1, 0].set_ylabel('CUDA Time (microseconds)')
axs[1, 0].set_title('CUDA Running Time vs Window Length')
axs[1, 0].legend()
# Plot 3: Dilation
for kernel in kernels:
    times = [average(list_dilation, lambda record:
        record[0] == kernel and
        record[1] in modes and
        record[2] == 2 and
        record[3] in heads and
        record[4] == 256 and
        record[5] == dilation and
        record[6] == 4096)
        for dilation in dilations]
    if any(time is not None for time in times):
        axs[1, 1].plot(dilations, times, label=kernel, color=colors[kernel], linestyle=styles[kernel], linewidth = 3)
axs[1, 1].set_xticks([1, 2, 3, 4])
axs[1, 1].set_xlabel('Dilation')
axs[1, 1].set_ylabel('CUDA Time (microseconds)')
axs[1, 1].set_title('CUDA Running Time vs Dilation Rate')
axs[1, 1].legend()

subplot_labels = ['a', 'b', 'c', 'd']
for i, ax in enumerate(axs.flat):
    ax.text(-0.15, 1.1, subplot_labels[i], transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)

for ax in axs.flat:
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

plt.savefig('Figure2.pdf', bbox_inches='tight')
