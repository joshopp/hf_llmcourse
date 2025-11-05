from datasets import load_dataset, Dataset, load_from_disk
import html
from transformers import AutoTokenizer

#------------------ CUSTOM DATASETS ------------------#
# load custom datasets via load_dataset(), data_files contains location or dataset_url
    # load_dataset("csv", data_files="my_file.csv") # csv, add sep = ";" as keywords to pandas.read_csv()
    # load_dataset("text", data_files="my_file.txt") # text files
    # load_dataset("json", data_files="my_file.jsonl") # JSON, add field = "..." to access field in nested JSON files (.json)
    # load_dataset("pandas", data_files="my_dataframe.pkl") # pandas

# squad_de_dataset = load_dataset("json", data_files="SQuAD_de_train.json", field = "data")
# example = squad_de_dataset["train"][0]
# print(f"Deutsches SQuAD Dataset: {squad_de_dataset}")
# # print(f"Inhalt: {example}")

# # to work further with datasets, include test and train datasets in one object.
# # load_dataset() can decompress (no gzp needed) or work directly via url:
# url = "https://github.com/crux82/squad-it/raw/master/"
# data_files = {
#     "train": url + "SQuAD_it-train.json.gz",
#     "test": url + "SQuAD_it-test.json.gz",
# }
# squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

# # Example with CSV files: 
# data_files_dota = {
#     "train": "dota2Train.csv",
#     "test": "dota2Test.csv",
# }
# dota_dataset = load_dataset("csv", data_files=data_files_dota)
# (f"Dota Dataset: {dota_dataset}")


#------------------ SLICING AND DICING ------------------#
# load dataset:
data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t") # .tsv = .csv with tabs (\t in Python)

# select either a number of items with range(x), or distinct items by specifying indices (e.g. =[0,10,20,50])
# shuffle to prevent the model from learning an artificial order:
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000)) # seed for reproducibility, remove in real cases
print(f"example entry of drug dataset: {drug_sample[:1]}\n")

# assert first column = patient IDs by matching number of entries
for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0")) # no AssertionError -> correct
    num_meds = len(drug_dataset[split].unique("drugName"))
    print(f"{num_meds} different drugs in {split} dataset")

# rename column
drug_dataset = drug_dataset.rename_column("Unnamed: 0", "patient_id")
print(f"\ndataset with renamed first column: {drug_dataset}")

# function to normalize condition labels (all lowercase)
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

# explicit filter function: filter rows with conditions = None to prevent error
def filter_nones(x):
    return x["condition"] is not None

drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None) # use lambda notation instead of function
drug_dataset = drug_dataset.map(lowercase_condition) # works this time
print(f"examples to verify usage of lowercases: {drug_dataset['train']['condition'][:3]}")


# #------------------ CREATING NEW COLUMNS ------------------#
# function to count number of words in each review:
def compute_review_length(example):
    return {"review_length": len(example["review"].split())} # returns dict

# using the map function automatically add this dict as new column ("review length")
drug_dataset = drug_dataset.map(compute_review_length)
print(f"\nexample item including new column: {drug_dataset['train'][0]}")
# alternative: drug_dataset = drug_dataset.add_column(lenghts), needs precomputed list of lengths

# sort dicts (by length):
print("\n shortest reviews: ", drug_dataset["train"].sort("review_length")[:3])
print("\nlongest review: ", drug_dataset["train"].sort("review_length", reverse=True)[:1])

# filter out short reviews <30 words:
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print("\n functional reviews after filtering:", drug_dataset.num_rows)


#------------------ MAP FUNCTION ------------------#
# use mapping function to unescape HTML characters:
drug_dataset = drug_dataset.map(lambda x : {"review": html.unescape(x["review"])}) # takes ca 15s

# using a batched format (always way faster!):
new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True) # takes <1s

# specify the number of (parallel) processes the map function uses by adding num_proc = 8 e.g.
# generally: don't use for fast tokenizers if batched = True -> already own multiprocessing

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# change number of elements (=rows) in dataset by mapping with batched = True:
def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128, #truncate after 128 words
        return_overflowing_tokens=True, # but return all chunks of text -> more outputs
    )

# example with first element (=row on dataset)
result = tokenize_and_split(drug_dataset["train"][0])
print("\n length of first entry after tokenization with overflow: ", [len(inp) for inp in result["input_ids"]])
# input_ids = [128, 49] instead of [177] -> one additional example

# use this on the whole dataset throws error: overflow creates more tokenized samples than original rows
# -> tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True) # -> fails
# -> other columns (e.g. drugName) keep old length -> shape mismatch

# solution: remove old columns (if you only need tokenized data)
tokenized_dataset = drug_dataset.map(
    tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names)
print("length of tokenized dataset: ", len(tokenized_dataset["train"]))
print("length of old dataset: ", len(drug_dataset["train"])) # much shorter

# second solution: expand size of old columns
def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extract mapping between new and old indices
    sample_map = result.pop("overflow_to_sample_mapping") # automatically generated when using return_overflowing_tokens
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result

tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True) # no errors now!
print("\n dataset with tokenized rows: ", tokenized_dataset)


#------------------ DATA FRAMES WITH PANDAS ------------------#
# datasets can easily switch output format to pandas, numpy, pytorch, etc. (nternally changes __getitem__() method)
drug_dataset.set_format("pandas")
print("\n output in Panda format: \n", drug_dataset["train"][:3])

# create panda DataFrame by slicing through dataset -> makes all panda functions available
train_df = drug_dataset["train"][:]

# example: fancy chaining to compute class distribution
frequencies = (train_df["condition"]
               .value_counts()
               .to_frame()
               .reset_index()
               .rename(columns={"index": "condition", "count": "frequency"}))
print("\n most common conditions: \n", frequencies.head(10))
# create new Arrow dataset from this:
freq_dataset = Dataset.from_pandas(frequencies)

# example: computing average rating
ratings = (train_df
           .groupby("drugName")
           .agg(ratings=("rating", "mean"), count = ("rating", "count")))
print("\n average drug ratings: \n", ratings.head())
rating_dataset = Dataset.from_pandas(ratings)

# convert original dataset back to Arrow format:
drug_dataset.reset_format()


#------------------ CREATING VALIDATION SET ------------------#
# if no distinct train/test/validation split is available, split base dataset:
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42) # 80% size train, 20% size test (=validation here)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]
print("\n Dataset Dict including validation set: ", drug_dataset_clean)


#------------------ SAVING AND RELOADING DATASETS ------------------#
# datasets are stored locally in a cache as arrow tables
# to save manually, use the following functions:
drug_dataset_clean.save_to_disk("../datasets/drug-reviews") # in arrow format
# reload folders of .arrow files like this:
drug_dataset_reloaded = load_from_disk("../datasets/drug-reviews")

# For JSON, CSV or Parquet, the Datasets in the DatasetDict have be saved individually:
for split, dataset in drug_dataset_clean.items():
    dataset.to_csv(f"../datasets/drug-reviews_csv/drug-reviews-{split}.csv", index=None) # or to_json, to_parquet
# load via load_dataset() like shown above
