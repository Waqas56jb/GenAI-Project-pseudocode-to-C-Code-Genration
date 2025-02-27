import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pickle

# Load dataset
file_path = "spoc-train.tsv"  # Update path if needed
df = pd.read_csv(file_path, sep='\t')

# Drop rows with missing text
df = df.dropna(subset=['text'])

# Take only 50 samples for quick training
df = df.sample(n=5000, random_state=42)

# Define Dataset class
class PseudoCodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data.iloc[idx]['text']
        target_text = self.data.iloc[idx]['code']

        inputs = self.tokenizer(input_text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        targets = self.tokenizer(target_text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }

# Initialize tokenizer and dataset
tokenizer = T5Tokenizer.from_pretrained("t5-small")
dataset = PseudoCodeDataset(df, tokenizer)

# DataLoader
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# Training Loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Save the trained model
with open("transformer_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training complete and saved as 'transformer_model.pkl'.")
