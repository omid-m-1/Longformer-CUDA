import torch
import pytorch_warmup as warmup
from torch import nn, optim
from tqdm import tqdm

import os
def train_model(model, train_loader, valid_loader, device, epochs, learning_rate, warmup_steps, model_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader)*epochs
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=int(warmup_steps*total_steps))

    for epoch in range(epochs):
        #train
        model.train()
        train_loss = 0

        train_iterator = tqdm(train_loader, desc='Training', total=len(train_loader))
        for inputs, attention_mask, labels in train_iterator:
            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            with warmup_scheduler.dampening():
                pass

            train_loss += loss.item()
            train_iterator.set_postfix({'Train Loss': loss.item()})

        train_loss /= len(train_loader)

        #validation
        model.eval()
        valid_loss = 0
        correct = 0
        total = 0

        valid_iterator = tqdm(valid_loader, desc='Validation', total=len(valid_loader))
        with torch.no_grad():
            for inputs, attention_mask, labels in valid_iterator:
                inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(inputs, attention_mask)

                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                valid_iterator.set_postfix({'Valid Loss': loss.item()})

        valid_loss /= len(valid_loader)
        accuracy = correct / total

        print(f'Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.2f}')

        save_path = os.path.join(model_path, f'longformer_IMDB_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), save_path)
