import requests
import json
import time
import math
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm
from dotenv import load_dotenv
import os, requests
from datasets import load_dataset

# Plan: Use Issues of huggingface/datasets GitHub page to create custom dataset
# 1: poll Issues enpoint via REST API

url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=1"
response = requests.get(url)
print("response status code: ", response.status_code) # 200: successful request
print(json.dumps(response.json(), indent=2)) # json.dumps used for pretty printing


# authenticate with gh to process more then 60 requests/h
load_dotenv()
gh_token = os.getenv("GITHUB_TOKEN")
headers = {"Authorization": f"token {gh_token}"}

# download all issues from GH repo
def fetch_issues(owner="huggingface", repo="datasets", num_issues=10_000, rate_limit=5_000, issues_path=Path("../datasets/GH")):
    
    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)

    batch = []
    all_issues = []
    per_page = 100  # Number of issues to return per page
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    for page in tqdm(range(1, num_pages+1)):
        # Query with state=all to get both open and closed issues
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
        issues.raise_for_status()  # good practice for error handling
        batch.extend(issues.json())

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = []  # Flush batch for next time period
            print(f"Reached GitHub rate limit. Sleeping for one hour ...")
            time.sleep(60 * 60 + 1)

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(f"Downloaded all the issues for {repo}! Dataset stored at {issues_path}/{repo}-issues.jsonl")

fetch_issues()

# load dataset
issues_dataset = load_dataset("json", data_files="../datasets/GH/datasets-issues.jsonl", split="train")
print("custom GH issues dataset: ", issues_dataset)

