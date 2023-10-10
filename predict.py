import torch
from transformers import LongformerTokenizer
from util.longformer_for_sequence_classification import LongformerForSequenceClassification
from util.predict_sentences import predict_sentences
from longformer_util.longformer import LongformerConfig, Longformer

import argparse
import os

predtrain_default =  '/mnt/huge_26TB/IMDB' # os.path.join(os.getcwd(), 'pretrained')
model_default = '/mnt/huge_26TB/IMDB/saved_model' # os.path.join(os.getcwd(), 'saved_model')
sentences_default = [
    "I strongly recommend this movie to anyone who enjoys human drama, theater--especially Shakespeare, or who has ever worked backstage in any capacity.",
    "The atmosphere is occasionally unsettling and the make-up effects are undoubtedly the most superior element of the entire film.",
    "It shows the most important memories of life, all of which can be topped by the single most elusive feeling: unexpected bliss.",
    "However, everything leading up to the end is very good, so the film still gets a deserving 8/10. Good, but not great."
]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_mode', default='base', help='valid options: [base, large]')
    parser.add_argument('--pretrain_path', default=predtrain_default, help='the path of the pretrained model')
    parser.add_argument('--attention_mode', default='dcg', help='valid options: [tvm, sliding_chunks, n2, sliding_chunks_no_overlap, dcg]')
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--model_path', default=model_default, help='the path of the model')
    parser.add_argument('--sentences', nargs='+', default=sentences_default)

    args = parser.parse_args()
    pretrain_mode = args.pretrain_mode
    pretrain_path = args.pretrain_path
    attention_mode = args.attention_mode
    epoch = args.epoch
    model_path = args.model_path
    sentences = args.sentences

    pretrain_path = os.path.join(pretrain_path, f'longformer-{pretrain_mode}-4096/')
    model_name = os.path.join(model_path, attention_mode, f'longformer_IMDB_epoch_{epoch}.pth')
    tokenizer_path = os.path.join(model_path, 'tokenizer/')

    if not os.path.exists(tokenizer_path):
        tokenizer = LongformerTokenizer.from_pretrained(f'allenai/longformer-{pretrain_mode}-4096', model_max_length=4096)
        os.makedirs(tokenizer_path)
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = LongformerTokenizer.from_pretrained(tokenizer_path, model_max_length=4096)

    config = LongformerConfig.from_pretrained(pretrain_path)
    config.attention_mode = attention_mode
    longformer = Longformer(config)
    hidden_size = config.hidden_size
    num_labels = 2
    model = LongformerForSequenceClassification(longformer, hidden_size, num_labels)
    state_dict = torch.load(model_name, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    predictions, probabilities = predict_sentences(sentences, model, tokenizer, device)

    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f'Sentence {i+1}: {sentences[i]}')
        print(f'Prediction: {pred}, Probability: {prob}')
        print()
