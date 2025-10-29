import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# Initialize the same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()

# HF also contains a datasets library (train, validation, test)
# MRCP: 1/10 of GLUE benchmark, MRCP = pairs of sentences that are paraphrases or not
raw_datasets = load_dataset("glue", "mrpc")
print(f"raw_datasets: {raw_datasets}")

# Access via indexing (like lists/dicts)
raw_train_dataset = raw_datasets["train"]
print(f"first element of train dataset: {raw_train_dataset[0]}")
print(f"15th element of train dataset: {raw_train_dataset[14]}")
raw_valid_dataset = raw_datasets["validation"]
print(f"87th element of validation dataset: {raw_valid_dataset[86]}")


# find corresponding labels to the integer by accessing the dataset's features
print(f"\nfeatures of train dataset: {raw_train_dataset.features}")

# Preprocessing and Tokenization:
# the tokenizer can handle pairs of sentences directly:
inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(f"\ntokenized inputs: {inputs}")
# Decode both sentences back to text
input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
print(f"\nDecoded tokens of both sentences: {input_tokens}")

# Tokenizing sentence pairs adds the token_type_ids that automatically distinguishes the two sentences:
s15_1 = tokenizer(raw_train_dataset[14]["sentence1"])
s15_2 = tokenizer(raw_train_dataset[14]["sentence2"])
s15_1_2 = tokenizer(raw_train_dataset[14]["sentence1"], raw_train_dataset[14]["sentence2"])
print(f"sentence 1 tokenized: {s15_1}")
print(f"sentence 2 tokenized: {s15_2}")
print(f"sentence 1 & 2 tokenized: {s15_1_2}")



# tokenize batches of examples with a worker function
def tokenize_fct(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=128)

# function to each element of dataset, slower, returns dict of scalars
tokenized_datasets = raw_datasets.map(tokenize_fct)

# function to whole dataset (dict of lists), faster, returns dict of lists
tokenized_datasets = raw_datasets.map(tokenize_fct, batched=True)
print(f"\ncolumn names: {tokenized_datasets.column_names}")


# Preprocessing: set dataset format to PyTorch tensors according to ml framework
tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence1", "sentence2"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets = tokenized_datasets.with_format("torch") # set format to PyTorch (or TensorFlow/NumPy) tensors
print("\ntokenized training dataset ", tokenized_datasets["train"])

# Create a small train dataset by selecting the range
small_train_dataset = tokenized_datasets["train"].select(range(100))