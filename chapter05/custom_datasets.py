from datasets import load_dataset
from datetime import datetime
from dotenv import load_dotenv
import json
import math
import os
import pandas as pd
from pathlib import Path
import requests
import time
from tqdm.notebook import tqdm


# Plan: Use issues of huggingface/datasets GitHub page to create custom dataset

#------------------ PULL ISSUES ------------------#
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

    request_count = 0
    batch = []
    all_issues = []
    per_page = 100  # Number of issues to return per page
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    for page in tqdm(range(num_pages)):
        # Query with state=all to get both open and closed issues
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
        issues.raise_for_status()  # good practice for error handling
        batch.extend(issues.json())

        request_count += 1

        if request_count > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = []  # Flush batch for next time period
            print(f"Reached GitHub rate limit. Sleeping for one hour ...")
            time.sleep(60 * 60 + 1)

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(f"Downloaded all the issues for {repo}! Dataset stored at {issues_path}/{repo}-issues.jsonl")


# download and clean dataset (nested entries) to be loaded correctly. Only needs to be done once
# fetch_issues()
# with open("../datasets/GH/datasets-issues.jsonl", "r") as f_in:
#     lines = [json.loads(line) for line in f_in]
# with open("../datasets/GH/datasets-issues.jsonl", "w") as f_out:
#     json.dump(lines, f_out, indent=2)

# load dataset
gh_dataset = load_dataset("json", data_files="../datasets/GH/datasets-issues.jsonl", split="train")
print("custom GH issues dataset: ", gh_dataset)


#------------------ CLEAN DATASETS ------------------#
sample = gh_dataset.shuffle().select(range(3))

# Print out the URL and pull request entries
for url, pr in zip(sample["html_url"], sample["pull_request"]):
    print(f">> URL: {url}")
    print(f">> Pull request: {pr}\n") # sometimes URLs, sometimes None

# remove redundant columns
gh_dataset = gh_dataset.remove_columns(["url", "repository_url", "labels_url", "comments_url", "events_url", "timeline_url", 
                                                "id", "node_id", "assignee", "assignees", "milestone", "performed_via_github_app",
                                                 "type", "active_lock_reason"])

# add info about pull requests
gh_dataset = gh_dataset.map(lambda x: {"is_pull_request": False if x["pull_request"] is None else True})

#filter out pull requests
issues_dataset = gh_dataset.filter(lambda x: x["is_pull_request"] is False)
print("slim issues dataset: ", issues_dataset)

#filter out issues
prs_dataset = gh_dataset.filter(lambda x: x["is_pull_request"] is True)
print("slim pull requests dataset: ", prs_dataset)


#------------------ WORKING WITH DATASETS ------------------#
# calculate average time to close an issue
# filter issues that are still open
closed_issues_ds = issues_dataset.filter(lambda x:x["closed_at"] is not None)

# convert to pandas format
closed_issues_ds.set_format("pandas")
issues_df = closed_issues_ds[:]

# parse timestamps
issues_df["created_at"] = pd.to_datetime(issues_df["created_at"])
issues_df["closed_at"] = pd.to_datetime(issues_df["closed_at"])

# calculate uptime (new key gets added automatically in pandas)
issues_df["uptime"] = (issues_df["closed_at"] - issues_df["created_at"]).dt.total_seconds() / 3600  # Stunden

# calculate mean
avg_issue_time = issues_df["uptime"].mean()

print(f"average time needed to close an issue: {avg_issue_time:.2f} hours or {avg_issue_time/24:.2f} days")


# calculate average time to close an issue -> without DataFrames
closed_prs_ds = prs_dataset.filter(lambda x:x["closed_at"] is not None)

def calc_uptime(batch):
    times = []
    for created, closed in zip(batch["created_at"], batch["closed_at"]):
            delta = (closed - created).total_seconds() / 3600
            times.append(delta)
    return {"uptime": times}

# map uptime
closed_prs_ds = closed_prs_ds.map(calc_uptime, batched=True)

# calculate avg
avg_prs_time = (sum(closed_prs_ds["uptime"]) / len(closed_prs_ds["uptime"]))
print(f"average time needed to close a pull request: {avg_prs_time:.2f} hours or {avg_prs_time/24:.2f} days")


#------------------ AUGMENTING THE DATASETS ------------------#
# get all comments under an issue
def get_comments(issue_number):
    url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"

    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            # rate limit hit
            reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            sleep_for = reset_time - int(time.time()) + 5  # buffer
            print(f"Rate limit reached. Sleeping for {sleep_for} seconds...")
            time.sleep(max(sleep_for, 0))
            continue
        elif response.status_code != 200:
            print(f"Error {response.status_code} for issue {issue_number}: {response.text}")
            return []
        else:
            break

    return [r["body"] for r in response.json()]

print("\n test comment: ", get_comments(2792))

# add comments to dataset
gh_comments_ds = gh_dataset.map(lambda x: {"comments": get_comments(x["number"])})

# save dataset
gh_comments_ds.save_to_disk("../datasets/GH/gh_comments")
# or for JSON:
# dataset.to_csv("../datasets/GH/gh_comments.csv", index=None)


#------------------ UPLOADING TO FACEHUB ------------------#
# gh_comments_ds.push_to_hub("github_issues_test")

# reload
remote_dataset = load_dataset("lewtun/github-issues", split="train")
print("confirming download: ", remote_dataset)