from datasets import load_dataset

#------------------ CUSTOM DATASETS ------------------#
# load custom datasets via load_dataset(), data_files contains location or dataset_url
    # load_dataset("csv", data_files="my_file.csv") # csv, add sep = ";" as keywords to pandas.read_csv()
    # load_dataset("text", data_files="my_file.txt") # text files
    # load_dataset("json", data_files="my_file.jsonl") # JSON, add field = "..." to access field in nested JSON files (.json)
    # load_dataset("pandas", data_files="my_dataframe.pkl") # pandas

squad_de_dataset = load_dataset("json", data_files="SQuAD_de_train.json", field = "data")
example = squad_de_dataset["train"][0]
print(f"Deutsches SQuAD Dataset: {squad_de_dataset}")
# print(f"Inhalt: {example}")

# to work further with datasets, include test and train datasets in one object.
# load_dataset() can decompress (no gzp needed) or work directly via url:
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

# # Example with CSV files: 
# data_files_dota = {
#     "train": "dota2Train.csv",
#     "test": "dota2Test.csv",
# }
# dota_dataset = load_dataset("csv", data_files=data_files_dota)
# # (f"Dota Dataset: {dota_dataset}")