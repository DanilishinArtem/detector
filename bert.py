import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import numpy as np
import matplotlib.pyplot as plt
from analizerPCA import AlalizerPCA

# Load the dataset
dataset = load_dataset("glue", "cola")
train_dataset = dataset['train']

# Load the pre-trained tokenizer and model
num_epochs = 5
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
analizer = AlalizerPCA(model)
def learningModel(model, train_dataloader, num_epochs, batch_size, analizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("npu" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        train_loss = 0
        counter = 0
        for batch in train_dataloader:
            counter += 1
            optimizer.zero_grad()
            # Tokenize input
            inputs = tokenizer(batch['sentence'], return_tensors='pt', padding=True, truncation=True, max_length=128)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            token_type_ids = inputs['token_type_ids'].to(device)

            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            analizer.disp_hist()
            optimizer.step()
            train_loss += loss.item()
        print("Epoch {}: Loss = {}".format(epoch, train_loss))
        # analizer.disp_hist()

learningModel(model, train_dataloader, num_epochs, batch_size, analizer)