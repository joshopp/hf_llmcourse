from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import evaluate

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define hyperparams fo Trainer -> local folder to save model checkpoints is required
training_args = TrainingArguments("../trainer/test_ch3", eval_strategy="epoch")
training_args_advanced = TrainingArguments(
    "../trainer/test_ch3",
    eval_strategy="epoch",
    fp16=True,  # Enable mixed precision (faster training, reduces memory usage)
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Gradient Accumulation, change effective batch size to 4 * 4 = 16
    learning_rate=2e-5,
    lr_scheduler_type="cosine",  #  alternative learning rate scheduler (defaukt: linear decay)
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer, # this sets the default data_collator to DataCollatorWithPadding (skip row before)
)

# prediction of Trainer to build eval metrics
eval_preds = trainer.predict(tokenized_datasets["validation"])
print(f"Predictions Shape: {eval_preds.predictions.shape}, {eval_preds.label_ids.shape}") # outputs = logits
preds = np.argmax(eval_preds.predictions, axis=-1) # choose higher prob logit

# compute evaluation metrics
metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=eval_preds.label_ids)

# eval compressed to one function:
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# add metrics to Trainer:
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# finetunes model on training dataset
trainer.train()


# ❯ python3 trainer_api.py
# Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# /home/joshy/Projects/venvs/venv42/lib/python3.10/site-packages/torch/utils/data/dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:12<00:00,  3.96it/s]
# Predictions Shape: (408, 2), (408,)
# Downloading builder script: 5.75kB [00:00, 1.66MB/s]
#   0%|                                                                                                                                                                  | 0/1377 [00:00<?, ?it/s]/home/joshy/Projects/venvs/venv42/lib/python3.10/site-packages/torch/utils/data/dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
# Downloading builder script: 5.75kB [00:00, 7.51MB/s]███▋                                                                                                     | 459/1377 [09:13<15:33,  1.02s/it]
# {'eval_loss': 0.3759743869304657, 'eval_accuracy': 0.8357843137254902, 'eval_f1': 0.8873949579831932, 'eval_runtime': 16.124, 'eval_samples_per_second': 25.304, 'eval_steps_per_second': 3.163, 'epoch': 1.0}                                                                                                                                                                                  
#  33%|██████████████████████████████████████████████████▋                                                                                                     | 459/1377 [09:29<15:33,  1.02s/it/home/joshy/Projects/venvs/venv42/lib/python3.10/site-packages/torch/utils/data/dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
# {'loss': 0.511, 'grad_norm': 11.539841651916504, 'learning_rate': 3.1880900508351494e-05, 'epoch': 1.09}                                                                                        
#  67%|█████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                  | 918/1377 [18:19<07:26,  1.03it/s]/home/joshy/Projects/venvs/venv42/lib/python3.10/site-packages/torch/utils/data/dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
# {'eval_loss': 0.5188243985176086, 'eval_accuracy': 0.8529411764705882, 'eval_f1': 0.8961937716262975, 'eval_runtime': 14.3803, 'eval_samples_per_second': 28.372, 'eval_steps_per_second': 3.547, 'epoch': 2.0}                                                                                                                                                                                 
#  67%|█████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                  | 918/1377 [18:33<07:26,  1.03it/s/home/joshy/Projects/venvs/venv42/lib/python3.10/site-packages/torch/utils/data/dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
# {'loss': 0.2557, 'grad_norm': 0.0578889325261116, 'learning_rate': 1.3725490196078432e-05, 'epoch': 2.18}                                                                                       
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1377/1377 [27:20<00:00,  1.02s/it]/home/joshy/Projects/venvs/venv42/lib/python3.10/site-packages/torch/utils/data/dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
# {'eval_loss': 0.6949222683906555, 'eval_accuracy': 0.8774509803921569, 'eval_f1': 0.9146757679180887, 'eval_runtime': 14.3868, 'eval_samples_per_second': 28.359, 'eval_steps_per_second': 3.545, 'epoch': 3.0}                                                                                                                                                                                 
# {'train_runtime': 1656.1839, 'train_samples_per_second': 6.644, 'train_steps_per_second': 0.831, 'train_loss': 0.30752470394624865, 'epoch': 3.0} 