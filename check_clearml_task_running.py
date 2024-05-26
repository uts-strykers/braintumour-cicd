from clearml import Task
import time
import sys


def check_task_status(task_id, timeout=3600):  # Adjust timeout as needed
    task = Task.get_task(task_id=task_id)
    start_time = time.time()

    while time.time() - start_time < timeout:
        task.reload()
        if task.status == "completed":
            print("Task completed successfully.")
            return
        elif task.status in ["failed", "stopped"]:
            print("Task failed or was stopped.")
            sys.exit(1)
        time.sleep(30)  # Poll every 30 seconds


if __name__ == "__main__":
    print(len(sys.argv))
    task_id = sys.argv[1]  # Pass the task ID as an argument
    check_task_status(task_id)