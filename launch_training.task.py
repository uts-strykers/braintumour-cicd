import argparse
from clearml import Task
from train_model import train_and_evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a ClearML task for training a model.")
    parser.add_argument("repo_url", type=str, help="URL of the repository containing the training script")
    parser.add_argument("branch_name", type=str, help="Branch name of the repository")
    parser.add_argument("commit_hash", type=str, help="Commit hash for the training script version")
    
    args = parser.parse_args()
    
    task = Task.init(
        project_name="My ML Project", task_name="Model Training on Latest Script"
    )
    # Print the task ID in a format that can be easily captured
    print(f"TASK_ID_OUTPUT: {task.id}")
    task.set_repo(repo=args.repo_url, branch=args.branch_name, commit=args.commit_hash)
    task.execute_remotely(queue_name="gitarth", exit_process=True)
    train_and_evaluate(task)