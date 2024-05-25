import os
from clearml import Task
from github3 import login
import sys

def post_pr_comment(task_id):
    task = Task.get_task(task_id=task_id)
    github = login(token=os.getenv("GH_TOKEN"))
    repo_name = os.getenv("GITHUB_REPOSITORY")
    pr_number = os.getenv("PR_NUMBER")  # Get PR number set in the workflow
    repo = github.repository(*repo_name.split("/"))
    pr = repo.pull_request(int(pr_number))  # Convert string PR number to int

    metrics = task.get_last_scalar_metrics()
    comment_body = "### Model Evaluation Results\n\n"
    for title, series in metrics.items():
        for ser, val in series.items():
            comment_body += f"**{title} - {ser}:** Last: {val['last']}\n"

    pr.create_comment(comment_body)

if __name__ == "__main__":
    task_id = sys.argv[1]  # Task ID needs to be passed as an argument
    post_pr_comment(task_id)