import os
from transformers import LongformerTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
import tarfile
import urllib.request

def show_progress(block_num, block_size, total_size):
    percentage = 100.0 * (block_num * block_size) / total_size
    percentage = min(percentage, 100)
    downloaded = (block_num * block_size)/(1024*1024)
    size_mb = total_size/(1024*1024)
    print(f'\rDownloaded {percentage:.1f}% of {size_mb:.1f} MB', end = '')

def download_and_extract(url, dest_dir, mode):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    filename = url.split('/')[-1]
    extracted_dir = os.path.join(dest_dir, filename.split('.')[0].split('_')[0])
    if os.path.exists(extracted_dir):
        print(f'{mode} is in {dest_dir}\n')
        return extracted_dir
    filename = os.path.join(dest_dir, filename)
    print(f'\nDownloading {mode} from {url} to {dest_dir}\n')
    urllib.request.urlretrieve(url, filename, reporthook=show_progress)
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(dest_dir)
    os.remove(filename)
    return extracted_dir

def load_data(data_dir):
    texts, labels = [], []
    for label in ['pos', 'neg']:
        label_dir = os.path.join(data_dir, label)
        for filename in os.listdir(label_dir):
            with open(os.path.join(label_dir, filename), encoding='utf-8') as f:
                texts.append(f.read())
            labels.append(0 if label == 'neg' else 1)
    return texts, labels

def load_split_data(data_dir):
    train_dir = f'{data_dir}/train'
    test_dir = f'{data_dir}/test'

    train_texts, train_labels = load_data(train_dir)
    valid_texts, valid_labels = train_texts[2000:2200], train_labels[2000:2200]
    train_texts, train_labels = train_texts[:200], train_labels[:200]
    test_texts, test_labels = load_data(test_dir)
    test_texts, test_labels = test_texts[:200], test_labels[:200]

    return train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels

def create_dataloaders(data_path, model_path, batch_size, pretrain_mode):

    tokenizer_path = os.path.join(model_path, 'tokenizer/')
    if not os.path.exists(tokenizer_path):
        tokenizer = LongformerTokenizer.from_pretrained(f'allenai/longformer-{pretrain_mode}-4096', model_max_length=4096)
        os.makedirs(tokenizer_path)
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = LongformerTokenizer.from_pretrained(tokenizer_path, model_max_length=4096)

    train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels = load_split_data(data_path)
    train_encoding = tokenizer(train_texts, truncation=True, padding='max_length', max_length=4096)
    valid_encoding = tokenizer(valid_texts, truncation=True, padding='max_length', max_length=4096)
    test_encoding = tokenizer(test_texts, truncation=True, padding='max_length', max_length=4096)

    train_dataset = TensorDataset(torch.tensor(train_encoding['input_ids']), torch.tensor(train_encoding['attention_mask']), torch.tensor(train_labels))
    valid_dataset = TensorDataset(torch.tensor(valid_encoding['input_ids']), torch.tensor(valid_encoding['attention_mask']), torch.tensor(valid_labels))
    test_dataset = TensorDataset(torch.tensor(test_encoding['input_ids']), torch.tensor(test_encoding['attention_mask']), torch.tensor(test_labels))

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader
