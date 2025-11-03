#--------------------- AUTHORIZATION ---------------------#
# Login via terminal: "hf auth login", "hf auth whoami" to identify
# When using a notebook:
from huggingface_hub import notebook_login
notebook_login()


#--------------------- PUSHING MODELS TO MODEL HUB: TRAINER API ---------------------#
# pushing to repo=name, specify hub_model_id else:
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    "bert-finetuned-mrpc", save_strategy="epoch", push_to_hub=True)
trainer = Trainer()
trainer.push_to_hub()


#--------------------- PUSHING MODELS TO MODEL HUB: DIRECT ---------------------#
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "camembert-base"
model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# ...

# push models to Hub inro repo "dummy-model"
model.push_to_hub("dummy-model")
tokenizer.push_to_hub("dummy-model")
tokenizer.push_to_hub("dummy-model", organization="huggingface") # when in org


#--------------------- PUSHING MODELS TO MODEL HUB: PYTHON LIBRARY ---------------------#
from huggingface_hub import (
    # User management
    login,
    logout,
    whoami,

    # Repository creation and management
    create_repo,
    delete_repo,
    update_repo_visibility,

    # And some methods to retrieve/change information about the content
    list_models,
    list_datasets,
    list_metrics,
    list_repo_files,
    upload_file,
    delete_file,
)

upload_file(
    "<path_to_file>/config.json",
    path_in_repo="config.json",
    repo_id="<namespace>/dummy-model",
    )

#--------------------- PUSHING MODELS TO MODEL HUB: REPO CLASS ---------------------#
from huggingface_hub import Repository
repo = Repository("<path_to_dummy_folder>", clone_from="<namespace>/dummy-model")

# use git on repo:
repo.git_pull()
model.save_pretrained("<path_to_dummy_folder>")
tokenizer.save_pretrained("<path_to_dummy_folder>")
repo.git_add()
repo.git_commit("Add model and tokenizer files")
repo.git_push()

# or use git directly in terminal
