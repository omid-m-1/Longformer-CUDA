import pickle
import xlsxwriter
import pandas as pd
import numpy as np

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

def calculate_speedup(time_base, time_compared):
    return time_base / time_compared

def calculate_slowdown(previous_time, current_time):
    return current_time / previous_time

kernels = ['tvm', 'dense', 'dcg']
modes = [1, 2, 3]
heads = [12, 24]
sequence_lengths = range(512, 16385, 512)
batch_sizes = range(1, 17)
window_lengths = range(64, 1025, 64)
dilations = range(1, 5)

# seq list
seq_idx_4096 = int((4096-512)/512)
out_of_memory_seq_len = []
speedup_dcg_over_tvm_s4096 = []
speedup_dcg_over_tvm_s16384 = []
average_speedup_dcg_over_tvm_seq = []
average_speedup_dcg_over_einsum_seq = []
slowdown_dcg_double_seq_len = []
# batch list
slowdown_dcg_double_batch = []
slowdown_tvm_double_batch = []
average_speedup_dcg_over_tvm_batch = []
average_speedup_dcg_over_einsum_batch = []
speedup_dcg_over_tvm_b2 = []
speedup_dcg_over_tvm_b16 = []
# window list
average_speedup_dcg_over_tvm_window = []
average_speedup_dcg_over_einsum_window = []
slowdown_dcg_double_window = []
slowdown_tvm_double_window = []
# dilation list
average_speedup_dcg_over_tvm_dilation = []
average_speedup_dcg_over_einsum_dilation = []

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
    out_of_memory_idx_seq = times_seq[1].index(None)
    out_of_memory_idx_batch = times_batch[1].index(None)

    # seq speedup
    out_of_memory_seq_len.append(sequence_lengths[out_of_memory_idx_seq-1])
    speedup_dcg_over_tvm_s4096.append(calculate_speedup(times_seq[0][seq_idx_4096], times_seq[2][seq_idx_4096]))
    speedup_dcg_over_tvm_s16384.append(calculate_speedup(times_seq[0][-1], times_seq[2][-1]))
    average_speedup_dcg_over_tvm_seq.append(np.mean([calculate_speedup(tvm, dcg) for tvm, dcg in zip(times_seq[0], times_seq[2])]))
    average_speedup_dcg_over_einsum_seq.append(np.mean([calculate_speedup(einsum, dcg) for einsum, dcg in zip(times_seq[1][:out_of_memory_idx_seq], times_seq[2][:out_of_memory_idx_seq])]))
    slowdown_dcg_double_seq_len.append(np.mean([calculate_slowdown(times_seq[2][seq_len], times_seq[2][2*seq_len+1]) for seq_len in range(int((len(sequence_lengths)/2)))]))

    # batch speedup
    slowdown_dcg_double_batch.append(np.mean([calculate_slowdown(times_batch[2][batch_size-1], times_batch[2][2*batch_size-1]) for batch_size in batch_sizes[:int(len(batch_sizes)/2)]]))
    slowdown_tvm_double_batch.append(np.mean([calculate_slowdown(times_batch[0][batch_size-1], times_batch[0][2*batch_size-1]) for batch_size in batch_sizes[:int(len(batch_sizes)/2)]]))
    average_speedup_dcg_over_tvm_batch.append(np.mean([calculate_speedup(tvm, dcg) for tvm, dcg in zip(times_batch[0], times_batch[2])]))
    average_speedup_dcg_over_einsum_batch.append(np.mean([calculate_speedup(einsum, dcg) for einsum, dcg in zip(times_batch[1][:out_of_memory_idx_batch], times_batch[2][:out_of_memory_idx_batch])]))
    speedup_dcg_over_tvm_b2.append(calculate_speedup(times_batch[0][1], times_batch[2][1]))
    speedup_dcg_over_tvm_b16.append(calculate_speedup(times_batch[0][-1], times_batch[2][-1]))

    # window speedup
    average_speedup_dcg_over_tvm_window.append(np.mean([calculate_speedup(tvm, dcg) for tvm, dcg in zip(times_window[0], times_window[2])]))
    average_speedup_dcg_over_einsum_window.append(np.mean([calculate_speedup(einsum, dcg) for einsum, dcg in zip(times_window[1], times_window[2])]))
    slowdown_dcg_double_window.append(np.mean([calculate_slowdown(times_window[2][window_len], times_window[2][2*window_len+1]) for window_len in range(int((len(window_lengths)/2)))]))
    slowdown_tvm_double_window.append(np.mean([calculate_slowdown(times_window[0][window_len], times_window[0][2*window_len+1]) for window_len in range(int((len(window_lengths)/2)))]))

    # dilation speedup
    average_speedup_dcg_over_tvm_dilation.append(np.mean([calculate_speedup(tvm, dcg) for tvm, dcg in zip(times_dilation[0], times_dilation[2])]))
    average_speedup_dcg_over_einsum_dilation.append(np.mean([calculate_speedup(einsum, dcg) for einsum, dcg in zip(times_dilation[1], times_dilation[2])]))

print('')
print('sequence info')
print(f'Einsum runs out of memory when sequence length increases beyond {np.min(out_of_memory_seq_len)}.')
print(f'Speedup of DCG over TVM for sequence length of 4096: {np.mean(speedup_dcg_over_tvm_s4096):.1f}')
print(f'Speedup of DCG over TVM for sequence length of 16384: {np.mean(speedup_dcg_over_tvm_s16384):.1f}')
print(f'Average speedup of DCG over TVM for sequence length: {np.mean(average_speedup_dcg_over_tvm_seq):.1f}')
print(f'Average speedup of DCG over einsum for sequence length: {np.mean(average_speedup_dcg_over_einsum_seq):.1f}')
print(f"Slowdown of DCG when sequence length doubles: {np.mean(slowdown_dcg_double_seq_len):.1f}")
print()
print('batch info')
print(f'Slowdown of DCG when batch size doubles: {np.mean(slowdown_dcg_double_batch):.1f}')
print(f'Slowdown of TVM when batch size doubles: {np.mean(slowdown_tvm_double_batch):.1f}')
print(f'Average speedup of DCG over TVM for batch size: {np.mean(average_speedup_dcg_over_tvm_batch):.1f}')
print(f'Average speedup of DCG over einsum for batch size: {np.mean(average_speedup_dcg_over_einsum_batch):.1f}')
print(f'Speedup of DCG over TVM for batch size of 2: {np.mean(speedup_dcg_over_tvm_b2):.1f}')
print(f'Speedup of DCG over TVM for batch size of 16: {np.mean(speedup_dcg_over_tvm_b16):.1f}')
print()
print('window info')
print(f'Average speedup of DCG over TVM for window length: {np.mean(average_speedup_dcg_over_tvm_window):.1f}')
print(f'Average speedup of DCG over einsum for window length: {np.mean(average_speedup_dcg_over_einsum_window):.1f}')
print(f'Slowdown of DCG when window length doubles: {np.mean(slowdown_dcg_double_window):.1f}')
print(f'Slowdown of TVM when window length doubles: {np.mean(slowdown_tvm_double_window):.1f}')
print()
print('dilation info')
print(f'Average speedup of DCG over einsum for dilation rate: {np.mean(average_speedup_dcg_over_einsum_dilation):.1f}')
print(f'Average speedup of DCG over TVM for dilation rate: {np.mean(average_speedup_dcg_over_tvm_dilation):.1f}')
print()

def format_sheet(writer, sheet_name, startrow, dataframe, header_color, info):
    dataframe.to_excel(writer, sheet_name=sheet_name, startrow=startrow + 1, index=False)
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'vcenter',
        'align': 'center',
        'fg_color': header_color,
        'border': 1})
    worksheet.write(startrow, 0, info, header_format)
    last_col_letter = chr(ord('A') + len(dataframe.columns) - 1)
    worksheet.merge_range(f'A{startrow+1}:{last_col_letter}{startrow+1}', info, header_format)
    cell_format = workbook.add_format({
        'valign': 'vcenter',
        'align': 'center'
        })
    cell_format.set_text_wrap()
    for col_num, value in enumerate(dataframe.columns.values):
        column_len = max(dataframe[value].astype(str).map(len).max(), len(value)) + 10
        worksheet.set_column(col_num, col_num, column_len)
        worksheet.set_row(startrow + 1, 30)
        worksheet.write(startrow + 1, col_num, value, header_format)
    for row_num in range(startrow + 2, startrow + 2 + len(dataframe)):
        worksheet.set_row(row_num, 30, cell_format)

df_seq_info = pd.DataFrame({
    'Einsum Out of Memory Seq': np.min(out_of_memory_seq_len),
    'Speedup DCG over TVM S4096': np.mean(speedup_dcg_over_tvm_s4096),
    'Speedup DCG over TVM S16384': np.mean(speedup_dcg_over_tvm_s16384),
    'Avg Speedup DCG over TVM': np.mean(average_speedup_dcg_over_tvm_seq),
    'Avg Speedup DCG over Einsum': np.mean(average_speedup_dcg_over_einsum_seq),
    'Slowdown DCG Double Seq': np.mean(slowdown_dcg_double_seq_len)}, index=[0])
df_batch_info = pd.DataFrame({
    'Slowdown DCG Double Batch': np.mean(slowdown_dcg_double_batch),
    'Slowdown TVM Double Batch': np.mean(slowdown_tvm_double_batch),
    'Avg Speedup DCG over TVM': np.mean(average_speedup_dcg_over_tvm_batch),
    'Avg Speedup DCG over Einsum': np.mean(average_speedup_dcg_over_einsum_batch),
    'Speedup DCG over TVM B2': np.mean(speedup_dcg_over_tvm_b2),
    'Speedup DCG over TVM B16': np.mean(speedup_dcg_over_tvm_b16)}, index=[0])
df_window_info = pd.DataFrame({
    'Avg Speedup DCG over TVM': np.mean(average_speedup_dcg_over_tvm_window),
    'Avg Speedup DCG over Einsum': np.mean(average_speedup_dcg_over_einsum_window),
    'Slowdown DCG Double Window': np.mean(slowdown_dcg_double_window),
    'Slowdown TVM Double Window': np.mean(slowdown_tvm_double_window)}, index=[0])
df_dilation_info = pd.DataFrame({
    'Avg Speedup DCG over Einsum': np.mean(average_speedup_dcg_over_einsum_dilation),
    'Avg Speedup DCG over TVM': np.mean(average_speedup_dcg_over_tvm_dilation)}, index=[0])

with pd.ExcelWriter('speed_part1.xlsx', engine='xlsxwriter') as writer:
    format_sheet(writer, 'Speed-Part1', 0, df_seq_info.round(1), '#ADD8E6', 'Sequence Info')
    format_sheet(writer, 'Speed-Part1', 4, df_batch_info.round(1), '#FFFBC8', 'Batch Info')
    format_sheet(writer, 'Speed-Part1', 8, df_window_info.round(1), '#90EE90', 'Window Info')
    format_sheet(writer, 'Speed-Part1', 12, df_dilation_info.round(1), '#FFA756', 'Dilation Info')

print('Data has been saved to speed_part1.xlsx')
