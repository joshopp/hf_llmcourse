from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification, BertModel, BertConfig, BertTokenizer
import torch

# models are defined by their checkpoint
checkpoint = "bert-base-cased"

#------------- MODELS --------------#
# models and configs can be automatically loaded from the model hub
model = AutoModel.from_pretrained(checkpoint)
config = AutoConfig.from_pretrained(checkpoint)

# the checkpoint configuration (blueprint of architecture) can also be fetched directly
bert_config = BertConfig.from_pretrained(checkpoint)
print(f"bert_config type: {type(bert_config)}")
print(f"bert_config: {bert_config}")

# the architecture can be modified by changing config parameters (eg num_hidden_layers)
bert_config_changed = BertConfig.from_pretrained(checkpoint, num_hidden_layers=6)

# model architectures can also be fetched directly and instantiated with custom config
bert_model = BertModel(bert_config)
bert_model_changed = BertModel(bert_config_changed)

# models can be loaded with different heads (e.g. sequence classification head)
class_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
class_model = AutoModelForSequenceClassification.from_pretrained(class_checkpoint)


# save model, config.json and pytorch model.safetensors stored locally
# bert_model.save_pretrained("../models/bert-base-cased-model")

# reload model from local directory
# model = AutoModel.from_pretrained("../models/bert-base-cased-model")


#------------- TOKENIZERS --------------#
#Tokenizers can be loaded automatically:
class_tokenizer = AutoTokenizer.from_pretrained(class_checkpoint)

# or directly from a pretrained model name
tokenizer = BertTokenizer.from_pretrained(checkpoint)

# Tokenizers can be saved and loaded from local files, same as with models
# tokenizer.save_pretrained("../models/bert-tokenizer")


# There are different kinds of tokenizers:
sequence = "Using a Transformer network is simple"
# Word based tokenizers: split on spaces, ID per word, unknown token [UNK]
# -> large vocabulary size, different IDs for similar words
tokens_word = sequence.split()
print(f"Word Tokens: {tokens_word}")

# Character based tokenizers: split on characters, ID per character, no unknown characters
# -> small vocabulary size, minimal information per ID, longer sequences
tokens_char = list(sequence)
print(f"Character Tokens: {tokens_char}")

# Subword based tokenizers: split on subwords (start/root + completion), ID per subword, unknown token [UNK]
# -> middle ground between word and character tokenizers
# -> medium vocabulary size, balance between information per ID and sequence length

# Encoding: converting text to tokens
tokens = tokenizer.tokenize(sequence)
print(f"Subword Tokens: {tokens}") # ## is used to indicate a completion

# Encoding: converting tokens to input IDs
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Input IDs: {input_ids}")

# This can also be done in one step:
ids = tokenizer(sequence)["input_ids"]
print(f"Input IDs (one step): {ids}")

# Decoding: converting input IDs back to tokens
decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(f"Decoded Tokens: {decoded_tokens}")

# Decoding: in one step
decoded_text = tokenizer.decode(ids)
print(f"Decoded Text: {decoded_text}")


#------------- BATCH INPUTS --------------#
# When using tensors, dimensions matter:
input_tensor = torch.tensor(input_ids) # with this line, model(input_tensor) would fail because the model expects a batch dimension
# To add a batch dimension, we can wrap the input IDs in a list:
input_tensor_batch = torch.tensor([input_ids])
print("\nAs tensor:", input_tensor)
print("With batch input:", input_tensor_batch)

output = class_model(input_tensor_batch)
print("Logits:", output.logits)

# Example with batchsize = 2:
batched_ids = [ids, ids]
batched_tensor = torch.tensor(batched_ids)
print("\nBatched Tensor:", batched_tensor)
batch_output = class_model(batched_tensor)
print("Batched Logits:", batch_output.logits)

# If the input sequences have different lengths, add padding (tokenizer.pad_token_id) to maintain rectangular shape:
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, class_tokenizer.pad_token_id],
]

print(f"Logits (Sequence 1): {class_model(torch.tensor(sequence1_ids)).logits}")
print(f"Logits (Sequence 2): {class_model(torch.tensor(sequence2_ids)).logits}")
print(f"Logits (Batched): {class_model(torch.tensor(batched_ids)).logits}") # the batched version is different from the individual ones due to the padding token

# to solve this, we can use attention masks  (tensors with same shape as input_ids, 1 for real tokens and 0 for padding tokens):
attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = class_model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(f"Logits (Batched with Attention Mask): {outputs.logits}") # this preserves the individual outputs

# truncate sequences if length > max length:
max_seq_length = 25
sequence = sequence[:max_seq_length]



# Putting it all together using the tokenizer's built-in methods:
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = class_tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = class_model(**tokens)
print(f"\nLogits shape: {output.logits.shape}")  # (batch_size, num_labels)
print(f"Full Output logits:", output.logits)


# convert logits by applying softmax to get probabilities
predictions = torch.nn.functional.softmax(output.logits, dim=-1)
print(f"Predictions after softmax: {predictions}")

print(class_model.config.id2label)  # mapping from label IDs to label names