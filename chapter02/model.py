from transformers import AutoModel, AutoConfig, AutoTokenizer, BertModel, BertConfig
import torch 

# automatically load model and config from model hub
model = AutoModel.from_pretrained("bert-base-cased")
config = AutoConfig.from_pretrained("bert-base-cased")

# fetch BERT checkpoint configuration directly (blueprint of architecture)
bert_config = BertConfig.from_pretrained("bert-base-cased")
print(f"bert_config type: {type(bert_config)}")
print(f"bert_config: {bert_config}")

# archtitecture can be modified by changing config parameters (eg num_hidden_layers)
bert_config_changed = BertConfig.from_pretrained("bert-base-cased", num_hidden_layers=6)

# fetch BERT model architecture directly (instead of via AutoModel), can be instantiated with custom config
bert_model = BertModel.from_pretrained(bert_config)
bert_model_changed = BertModel.from_pretrained(bert_config_changed)


# save model, config.json and pytorch model.safetensors stored locally
bert_model.save_pretrained("../models/bert-base-cased-model")

# reload model from local directory
model = AutoModel.from_pretrained("../models/bert-base-cased-model")



tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# encode and decode a single sentence
encoded_input = tokenizer("Welcome, this is a single sentence as input!")
print(f"encoded input: {encoded_input}")
decoded_input = tokenizer.decode(encoded_input["input_ids"])
print(f"Decoded input: {decoded_input}") # adds special tokens

# encode and decode a batch of sentences
encoded_input_list = tokenizer("Welcome, this is the first sentence.", "And this is the second one.")
print(f"encoded list input: {encoded_input_list}")
encoded_input_tensor = tokenizer(["Welcome, this is the first sentence.", "And this is the second one."], padding=True, return_tensors="pt") # add padding for different length inputs
print(f"encoded tensor input: {encoded_input_tensor}")

decoded_input_list = tokenizer.decode(encoded_input_list["input_ids"])
print(f"decoded list input: {decoded_input_list}") # adds special tokens


sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

encoded_sequences = tokenizer(sequences, padding=True, truncation=True)["input_ids"] # already rectangular, easy to use with pytorch

# Example usage:
# model_inputs = torch.tensor(encoded_sequences)
# output = model(model_inputs)
