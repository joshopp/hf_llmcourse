from datasets import load_dataset
import evaluate
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, get_scheduler
from tqdm.auto import tqdm


# standard init and preprocessing with data collator:
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print(f'Finished dataset columns: {tokenized_datasets["train"].column_names}')


#--------------------- DATALOADER, OPTIMIZER; LR SCHEDULER --------------------#
# Define PyTorch Dataloader
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator)

# inspect batch
for batch in train_dataloader:
    break
batch_shape = {k: v.shape for k, v in batch.items()}
print(f"batch items with shape: {batch_shape}")

# inspect output
outputs = model(**batch)
print(f"model output: {outputs.loss, outputs.logits.shape}")


# Pytorch Optimizer, AdamW = Adam with weight decay (e.g. AdamW(model.parameters(), lr=5e-5, weight_decay=0.01))
optimizer = AdamW(model.parameters(), lr=5e-5)
# 8-bit Adam: Use bitsandbytes for memory-efficient optimization
# Different learning rates: Lower learning rates (1e-5 to 3e-5) often work better for large models

# learning rate scheduler
num_epochs = 3 # default of HF Trainer class
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(f"number of training steps: {num_training_steps}")

# Push to GPU if available:
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(f" device used for training: {device}")

#train model with custom progress bar
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs): # 3 iterations
    for batch in train_dataloader: # train batchwise
        batch = {k: v.to(device) for k, v in batch.items()} # push to GPU
        outputs = model(**batch) # calculate output
        loss = outputs.loss
        loss.backward() # computes gradients

        optimizer.step() # update model params by appying gradients
        lr_scheduler.step() # update learning rate (linear decay in this case)
        optimizer.zero_grad() # reset gradients for new batch, prevents accumulation
        progress_bar.update(1)

# Modern Training Optimizations to  your training loop  more efficient:
    # Gradient Clipping: Add torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) before optimizer.step()
    # Mixed Precision: Use torch.cuda.amp.autocast() and GradScaler for faster training
    # Gradient Accumulation: Accumulate gradients over multiple batches to simulate larger batch sizes
    # Checkpointing: Save model checkpoints periodically to resume training if interrupted


#--------------------- EVALUATION --------------------#
metric = evaluate.load("glue", "mrpc")
model.eval() # evaluation mode: disables dropout and implements batch normalization
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad(): # disbles gradient tracking (faster+less memory)
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"]) # accumulate metric over all batches

metric.compute()


#--------------------- ACCELERATION --------------------#
# accelerate by using the Accelerator class and change code like below:
# USAGE: "accelerate config" in terminal to change params, then "accelerate launch xyz.py"

from accelerate import Accelerator
accelerator = Accelerator() # NEW: initiate accelerator

# NEW: prepare model, optimizer and dataloader
train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
progress_bar = tqdm(range(num_training_steps))

#train model
model.train()
for epoch in range(num_epochs):
    for batch in train_dl: 
        outputs = model(**batch) # changing device to GPU not needed
        loss = outputs.loss
        accelerator.backward(loss) # NEW: accelerator steps

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# Usage in Notebook:
# from accelerate import notebook_launcher
# notebook_launcher(training_function)



#--------------------- EXAMPLE --------------------#
# finetuning SST-2 dataset without acceleration
sst2_dataset = load_dataset("glue", "sst2")
print(f"\nfeatures of train dataset: {sst2_dataset['train'].features}")

checkpoint = "bert-base-cased"
sst2_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
sst2_tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_sst(examples):
    return sst2_tokenizer(examples["sentence"], padding= True, truncation=True)

sst2_tokenized = sst2_dataset.map(tokenize_sst, batched=True)
sst2_data_collator = DataCollatorWithPadding(tokenizer=sst2_tokenizer)
sst2_tokenized = sst2_tokenized.remove_columns(["idx", "sentence"])
sst2_tokenized = sst2_tokenized.rename_column("label", "labels")
sst2_tokenized = sst2_tokenized.with_format("torch")
print(f"\nSST-2 tokenized train dataset: {sst2_tokenized['train']}")

sst2_train_dl = DataLoader(
    sst2_tokenized["train"], shuffle=True, batch_size=8, collate_fn=sst2_data_collator)
sst2_eval_dl = DataLoader(
    sst2_tokenized["validation"], batch_size=8, collate_fn=sst2_data_collator)

sst2_optimizer = AdamW(sst2_model.parameters(), lr=5e-5)

num_epochs = 3 
num_training_steps = num_epochs * len(sst2_train_dl)
sst2_lr_scheduler = get_scheduler(
    "linear",
    optimizer=sst2_optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(f"number of training steps: {num_training_steps}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
sst2_model.to(device)
print(f" device used for training: {device}")

# model training
progress_bar = tqdm(range(num_training_steps))
sst2_model.train()
for epoch in range(num_epochs):
    for batch in sst2_train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = sst2_model(**batch) 
        loss = outputs.loss
        loss.backward() 

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


# model evaluation
sst2_metric = evaluate.load("glue", "mrpc")
sst2_model.eval()
for batch in sst2_eval_dl:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = sst2_model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    sst2_metric.add_batch(predictions=predictions, references=batch["labels"])

sst2_metric.compute()



# Best practices for production usage:
    # Model Evaluation: Always evaluate your model on multiple metrics, not just accuracy. Use the HF Evaluate library for comprehensive evaluation.
    # Hyperparameter Tuning: Consider using libraries like Optuna or Ray Tune for systematic hyperparameter optimization.
    # Model Monitoring: Track training metrics, learning curves, and validation performance throughout training.
    # Model Sharing: Once trained, share your model on the Hugging Face Hub to make it available to the community.
    # Efficiency: For large models, consider techniques like gradient checkpointing, parameter-efficient fine-tuning (LoRA, AdaLoRA), or quantization methods.