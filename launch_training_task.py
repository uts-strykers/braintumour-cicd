import argparse
from clearml import Task
from train_model import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a ClearML task for training a model.")
    parser.add_argument("--repo_url", type=str, default="https://github.com/uts-strykers/braintumour-cicd", help="URL of the repository containing the training script")
    parser.add_argument("--branch_name", type=str,default="ci-cd", help="Branch name of the repository")
    parser.add_argument("--commit_hash", type=str, default="2e4e56feffc6dde1640c7ce4983993e1709d213b", help="Commit hash for the training script version")
     
    args = parser.parse_args()
    
    task = Task.init(
        project_name="BrainScan2", task_name="TrainModel"
    )
    # Print the task ID in a format that can be easily captured
    print(f"TASK_ID_OUTPUT: {task.id}")
    task.set_repo(repo=args.repo_url, branch=args.branch_name, commit=args.commit_hash)
    #task.execute_remotely(queue_name="uts-strykers-queue", exit_process=True)

    train_model(task)#, results_dir, epochs, args)
    