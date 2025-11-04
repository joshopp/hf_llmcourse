from datasets import load_dataset

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


#------------------ SLICE AND DICE ------------------#
# load dataset
data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t") # .tsv = .csv with tabs (\t in Python)

# grab small sample:
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
print(f"example of drug dataset: {drug_sample[:3]}\n")

# assert first column = patient IDs by matching number of entries
for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0")) # no AssertionError -> correct
    num_meds = len(drug_dataset[split].unique("drugName"))
    print(f"{num_meds} different drugs in {split} dataset")

# rename column
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id")
print(f"\ndataset with renamed first column: {drug_dataset}")

# normalize condition labels (all lowercase)
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

# explicit filter function: filter rows with conditions = None to prevent error
def filter_nones(x):
    return x["condition"] is not None

drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None) # use lambda notation instead
drug_dataset = drug_dataset.map(lowercase_condition)
# Check that lowercasing worked
print(f"example for lowercase verification: {drug_dataset['train']['condition'][:3]}")
