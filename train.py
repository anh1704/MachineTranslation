import torch
from transformers import MarianMTModel, MarianTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset

# Define a custom dataset class
class TranslationDataset(Dataset):
    def __init__(self, source_file, target_file, tokenizer, max_length=128):
        self.source_texts = open(source_file, 'r', encoding='utf-8').readlines()
        self.target_texts = open(target_file, 'r', encoding='utf-8').readlines()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx].strip()
        target_text = self.target_texts[idx].strip()

        source = self.tokenizer(source_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors="pt")
        target = self.tokenizer(target_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors="pt")

        source_input_ids = source.input_ids.squeeze()
        source_attention_mask = source.attention_mask.squeeze()
        target_input_ids = target.input_ids.squeeze()

        return {
            'input_ids': source_input_ids,
            'attention_mask': source_attention_mask,
            'labels': target_input_ids
        }

# Load the model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-vi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Load the dataset
train_dataset = TranslationDataset('train.en.txt', 'train.vi.txt', tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):  # Number of epochs
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")