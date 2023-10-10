import torch
from tqdm import tqdm

def evaluate_model(model, test_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    test_iterator = tqdm(test_loader, desc='Testing', total=len(test_loader))
    with torch.no_grad():
        for inputs, attention_mask, labels in test_iterator:
            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
            logits = model(inputs, attention_mask)

            loss = criterion(logits, labels)

            test_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            test_iterator.set_postfix({'Test Loss': loss.item()})

    test_loss /= len(test_loader)
    accuracy = correct / total

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}')
