from config import *
import pandas as pd
from setting import * 
import pandas as pd

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import torch

# Dataset and batch size configuration
batch_size = 6000  # Adjust based on available memory

# Initialize a Byte Pair Encoding (BPE) tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Trainer for building vocab
trainer = trainers.BpeTrainer(vocab_size=30_000, special_tokens=["<pad>", "<unk>", "<s>", "</s>"])

# Initialize a set to store unique characters
unique_chars = set()

# Load CSV file and prepare text for training the tokenizer
texts = []
for chunk in pd.read_csv(dataset, chunksize=batch_size):
    chunk["Text"] = chunk["Text"].fillna("").astype(str)
    texts.extend(chunk["Text"].tolist())
    unique_chars.update("".join(chunk["Text"]))
    print(f"Processed batch with {len(chunk)} rows.")

print("Unique characters:", unique_chars)
print("Number of unique characters:", len(unique_chars))

# Train tokenizer on the text data
tokenizer.train_from_iterator(texts, trainer)
vocab_size = tokenizer.get_vocab_size()
print("Subword-level vocab size:", vocab_size)

# Tokenize the dataset in smaller chunks to avoid memory issues
encoded_data = []
for text in texts:
    encoded_data.extend(tokenizer.encode(text).ids)

# Check if data is available
if len(encoded_data) > 0:
    # Split into training and validation sets (90/10 split)
    n = int(0.9 * len(encoded_data))
    train_data = encoded_data[:n]
    val_data = encoded_data[n:]
    print("Training data size:", len(train_data))
    print("Validation data size:", len(val_data))
else:
    print("Data is empty.")





# # data loading for training which generates a small batch of data of inputs x and targets y
# def get_batch(split, batch_size):
#     # whether we grab from our training or validation dataset
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - config.max_position_embeddings, (batch_size,))
#     x = torch.stack([data[i:i+config.max_position_embeddings] for i in ix])
#     y = torch.stack([data[i+1:i+config.max_position_embeddings+1] for i in ix])
#     x, y = x.to(config.device), y.to(config.device)
#     return x, y

def get_batch(split, batch_size, max_seq_len, device):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - max_seq_len, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+max_seq_len]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+max_seq_len+1]) for i in ix])
    return x.to(device), y.to(device)
