import torch
from util.train import train_model
from util.evaluate import evaluate_model
from util.data_processor import create_dataloaders, download_and_extract
from util.longformer_for_sequence_classification import LongformerForSequenceClassification
from util.data_processor import download_and_extract
from longformer_util.longformer import LongformerConfig, Longformer

import argparse
import os

data_url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
model_url = 'https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/'
data_default = '/mnt/huge_26TB/IMDB' # os.path.join(os.getcwd(), 'data')
predtrain_default = '/mnt/huge_26TB/IMDB' # os.path.join(os.getcwd(), 'pretrained')
model_default = '/mnt/huge_26TB/IMDB/saved_model' # os.path.join(os.getcwd(), 'saved_model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=data_default, help='the path for saving IMDB dataset')
    parser.add_argument('--pretrain_mode', default='base', help='valid options: [base, large]')
    parser.add_argument('--pretrain_path', default=predtrain_default, help='the path for saving the pretrained model')
    parser.add_argument('--model_path', default=model_default, help='the path for saving the model')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--attention_mode', default='dcg', help='valid options: [tvm, sliding_chunks, n2, sliding_chunks_no_overlap, dcg]')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--learning_rate', type=int, default=3e-5)
    parser.add_argument('--warmup_steps', type=int, default=0.1)

    args = parser.parse_args()
    data_path = args.data_path
    pretrain_mode = args.pretrain_mode
    pretrain_path = args.pretrain_path
    model_path = args.model_path
    batch_size = args.batch_size
    attention_mode = args.attention_mode
    epochs = args.epochs
    learning_rate = args.learning_rate
    warmup_steps = args.warmup_steps

    model_url = os.path.join(model_url, f'longformer-{pretrain_mode}-4096.tar.gz')
    data_path = download_and_extract(data_url, data_path, 'data')
    pretrain_path = download_and_extract(model_url, pretrain_path, 'pretrained model')
    train_loader, valid_loader, test_loader = create_dataloaders(data_path, model_path, batch_size, pretrain_mode)

    config = LongformerConfig.from_pretrained(pretrain_path)
    config.attention_mode = attention_mode
    longformer = Longformer(config)
    state_dict = torch.load(os.path.join(pretrain_path, 'pytorch_model.bin'), map_location=device)
    state_dict = {k.replace('roberta.',''): v for k,v in state_dict.items()}
    longformer.load_state_dict(state_dict, strict=False)

    hidden_size = config.hidden_size
    num_labels = 2
    model = LongformerForSequenceClassification(longformer, hidden_size, num_labels)
    model.to(device)

    model_path = os.path.join(model_path, attention_mode)
    if not os.path.exists(model_path): os.makedirs(model_path)
    train_model(model, train_loader, valid_loader, device, epochs, learning_rate, warmup_steps, model_path)
    evaluate_model(model, test_loader, device)
