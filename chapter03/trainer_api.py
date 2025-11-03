from datasets import load_dataset
import evaluate
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import wandb


# standard initialization
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


#--------------------- TRAINER API --------------------#
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


#--------------------- EVALUATION --------------------#
# prediction of Trainer to build eval metrics
eval_preds = trainer.predict(tokenized_datasets["validation"])
print(f"Predictions Shape: {eval_preds.predictions.shape}, {eval_preds.label_ids.shape}") # outputs = logits
preds = np.argmax(eval_preds.predictions, axis=-1) # choose highest prob logit

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


#--------------------- LOSS FUNCTION  --------------------#
# High initial loss (poor initial predictions), loss decreases over time (training progresses), convergences eventually (finished learning patterns) 

# tracking loss with weights and biases:
wandb.init(project="transformer-fine-tuning", name="bert-mrpc-analysis")

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    logging_steps=10,  # log metrics every 10 steps
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    report_to="wandb",  # send logs to Weights & Biases
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# automatically log metrics while training
trainer.train()

# Monitor during training:
    # Loss convergence: Is the loss still decreasing or has it plateaued?
    # Overfitting signs: Is validation loss starting to increase while training loss decreases?
    # Learning rate: Are the curves too erratic (LR too high) or too flat (LR too low)?
    # Stability: Are there sudden spikes or drops that indicate problems?

# Monitoring after training:
    # Final performance: Did the model reach acceptable performance levels?
    # Efficiency: Could the same performance be achieved with fewer epochs?
    # Generalization: How close are training and validation performance?
    # Trends: Would additional training likely improve performance?


#--------------------- ISSUES: OVERFITTING  --------------------#
# Symptoms:
    # Training loss continues to decrease while validation loss increases or plateaus
    # Large gap between training and validation accuracy
    # Training accuracy much higher than validation accuracy
# Solutions:
    # Regularization: Add dropout, weight decay, or other regularization techniques
    # Early stopping: Stop training when validation performance stops improving
    # Data augmentation: Increase training data diversity
    # Reduce model complexity: Use a smaller model or fewer parameters

# Example of detecting overfitting: Add early stopping callback to trainer
from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    num_train_epochs=10,  # Set high, but we'll stop early
)

# Add early stopping to prevent overfitting
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

#--------------------- ISSUES: UNDERFITTING  --------------------#
# Symptoms:
    # Both training and validation loss remain high
    # Model performance plateaus early in training
    # Training accuracy is lower than expected
# Solutions:
    # Increase model capacity: Use a larger model or more parameters
    # Train longer: Increase the number of epochs
    # Adjust learning rate: Try different learning rates (usually too low)
    # Check data quality: Ensure your data is properly preprocessed

#--------------------- ISSUES: ERRATIC LR  --------------------#
# Symptoms:
    # Frequent fluctuations in loss or accuracy
    # Curves show high variance or instability
    # Performance oscillates without clear trend
    # Both training and validation curves show erratic beha
# Solutions:
    # Lower learning rate: Reduce step size for more stable training
    # Increase batch size: Larger batches provide more stable gradients
    # Gradient clipping: Prevent exploding gradients
    # Better data preprocessing: Ensure consistent data quality