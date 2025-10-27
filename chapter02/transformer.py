from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

# checkpoint of model to be used
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# download the tokenizer the model was pretrained with
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

# automatically tokenize the input, padding=True for inputs with different lengths, truncation=True to truncate inputs >max tokens, return_tensors="pt" to return PyTorch tensors
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs) # dict with identifiers of tokens and attention mask

# download the model
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)  # (batch_size, sequence_length, hidden_size)

# download the model with a sequence classification head
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)  # (batch_size, num_labels)
print(outputs.logits)

# convert logits by applying softmax to get probabilities
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

print(model.config.id2label)  # mapping from label IDs to label names