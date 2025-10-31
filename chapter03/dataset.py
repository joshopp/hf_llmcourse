from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader


# Initialize the same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]

#--------------------- DATASETS --------------------#
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

#--------------------- PREPROCESSING --------------------#
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
tokenized_datasets_orig = raw_datasets.map(tokenize_fct, batched=True)
print(f"\ncolumn names: {tokenized_datasets.column_names}")


# Preprocessing: set dataset format to PyTorch tensors according to ml framework
tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence1", "sentence2"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets = tokenized_datasets.with_format("torch") # set format to PyTorch (or TensorFlow/NumPy) tensors
print("\ntokenized training dataset ", tokenized_datasets["train"])

# Create a small train dataset by selecting the range
small_train_dataset = tokenized_datasets["train"].select(range(100))

#--------------------- DYNAMIC PADDING --------------------#
# Pad during batching -> less padding then using max length
# collate function: puts together samples into a batch, used by DataLoader (standard PyTorch converter)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# tokenize and preprocess batches without padding
def tokenize_coll(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
coll_datasets = raw_datasets.map(tokenize_coll, batched=True)
coll_datasets = coll_datasets.remove_columns(["idx", "sentence1", "sentence2"])
coll_datasets = coll_datasets.rename_column("label", "labels")
coll_datasets = coll_datasets.with_format("torch")

#Use Pytorch DataLoader with the data collator
train_dataloader = DataLoader(coll_datasets["train"], batch_size=8, shuffle=True, collate_fn=data_collator) # enable shuffling for different batches each epoch

for step, batch in enumerate(train_dataloader):
    print(f"shape of input_ids in batch {step}: {batch['input_ids'].shape}")
    if step > 5:
        break


samples = coll_datasets["train"][:8]
samples = {k: v for k, v in samples.items()}
print(f"\n lengths of sentences {[len(x) for x in samples['input_ids']]}")

batch = data_collator(samples)
print(f"lengths after batching/padding with collator: {[v.shape for k, v in batch.items() if k == 'input_ids']}")


#--------------------- EXAMPLE --------------------#
#Preparing GLUE SST-2 Dataset for Fine-Tuning:
sst2_dataset = load_dataset("glue", "sst2")
print(f"\nSST-2 dataset: {sst2_dataset}")
print(f"\nfeatures of train dataset: {sst2_dataset['train'].features}")
def tokenize_sst(examples):
    return tokenizer(examples["sentence"], padding= True, truncation=True)

sst2_tokenized = sst2_dataset.map(tokenize_sst, batched=True)
sst2_tokenized = sst2_tokenized.remove_columns(["idx", "sentence"])
sst2_tokenized = sst2_tokenized.rename_column("label", "labels")
sst2_tokenized = sst2_tokenized.with_format("torch")
print(f"\nSST-2 tokenized train dataset: {sst2_tokenized['train']}")



#-------------------- TRAINER API --------------------#
# API to fine-tune models on datasets directly -> removing columns, setting format, data collator etc. is done automatically
