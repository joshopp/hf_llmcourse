from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


# load Microsoft Research Paraphrase Corpus (MRPC) training dataset
dataset = load_dataset("nyu-mll/glue", "mrpc", split="train")


# load Bert model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# tokenize batches of examples
def encode(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding=True)

dataset = dataset.map(encode, batched=True)
print(dataset[0])

# rename label column to labels for compatibility with the model
dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)

# set dataset format to PyTorch tensors according to ml framework
dataset = dataset.select_columns(["input_ids", "token_type_ids", "attention_mask", "labels"])
dataset = dataset.with_format(type="torch")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
