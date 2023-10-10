import torch

def prepare_sentences(sentences, tokenizer):
    encodings = tokenizer(sentences, truncation=True, padding='max_length', max_length=4096)
    input_ids = torch.tensor(encodings['input_ids'])
    attention_mask = torch.tensor(encodings['attention_mask'])
    return input_ids, attention_mask

def predict_sentences(sentences, model, tokenizer, device):
    model.eval()
    input_ids, attention_mask = prepare_sentences(sentences, tokenizer)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)

    predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
    return predictions, probabilities.cpu().numpy()
