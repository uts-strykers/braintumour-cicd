import os
import numpy as np
import pandas as pd
import clearml
from clearml import Dataset, Task, TaskTypes
from clearml.automation.controller import PipelineDecorator
from clearml.automation import HyperParameterOptimizer, UniformIntegerParameterRange, GridSearch
import atexit
import tensorflow
import matplotlib.pyplot as plt
import tempfile
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard

def cleanup():
    print("Cleaning up...")

atexit.register(cleanup)

@PipelineDecorator.component(name="StartTask", return_values=["start_task_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def start_data_pipeline(dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset, Task
    
    return "startdatapipeline"

# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
@PipelineDecorator.component(name="UploadRawTrainData", return_values=["train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_one_a(start_task_id, dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset

    dataset = Dataset.get(
        dataset_id=None,  
        dataset_project=dataset_project + "Train",
        dataset_name=dataset_name + "Train",
        dataset_tags="trainrawdata",
        auto_create=True
    )

    if not os.path.isdir(dataset_root):
        raise ValueError(
            f"The specified path '{dataset_root}' is not a directory or does not exist."
        )
    
    train_raw_dataset_path = os.path.join(dataset_root, 'train')

    print(f"Uploading train raw dataset from '{train_raw_dataset_path}'.")
    dataset.add_files(train_raw_dataset_path)
    dataset.upload()
    dataset.finalize()
    
    return dataset.id

    
    
@PipelineDecorator.component(name="UploadRawValidData", return_values=["valid_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_one_b(start_task_id, dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset
    
    dataset = Dataset.get(
        dataset_id=None,  
        dataset_project=dataset_project + "Valid",
        dataset_name=dataset_name + "Valid",
        dataset_tags="validrawdata",
        auto_create=True
    )

    if not os.path.isdir(dataset_root):
        raise ValueError(
            f"The specified path '{dataset_root}' is not a directory or does not exist."
        )
    
    valid_raw_dataset_path = os.path.join(dataset_root, 'valid')

    print(f"Uploading valid raw dataset from '{valid_raw_dataset_path}'.")
    dataset.add_files(valid_raw_dataset_path)
    dataset.upload()
    dataset.finalize()
    
    return dataset.id

@PipelineDecorator.component(name="UploadRawTestData", return_values=["test_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_one_c(start_task_id, dataset_project, dataset_name, dataset_root):
    import os
    from clearml import Dataset
    
    dataset = Dataset.get(
        dataset_id=None,  
        dataset_project=dataset_project + "Test",
        dataset_name=dataset_name + "Test",
        dataset_tags="testrawdata",
        auto_create=True
    )

    if not os.path.isdir(dataset_root):
        raise ValueError(
            f"The specified path '{dataset_root}' is not a directory or does not exist."
        )
    
    test_raw_dataset_path = os.path.join(dataset_root, 'test')

    print(f"Uploading valid raw dataset from '{test_raw_dataset_path}'.")
    dataset.add_files(test_raw_dataset_path)
    dataset.upload()
    dataset.finalize()
    
    return dataset.id
 

@PipelineDecorator.component(name="ProcessTrainData", return_values=["processed_train_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_two_a(
    raw_train_dataset_id, processed_dataset_project, processed_dataset_name, dataset_root, processed_dataset_root
):
    import os
    import numpy as np
    from clearml import Dataset
    import cv2
    import glob

    # Fetch the dataset from ClearML
    dataset = Dataset.get(raw_train_dataset_id) #dataset_name=processed_dataset_name, dataset_project=processed_dataset_project)
    local_dataset_path = dataset.get_local_copy()

    # Define paths for train, valid, and test datasets
    images_path = os.path.join(local_dataset_path, "train/images")
    labels_path = os.path.join(local_dataset_path, "train/labels")
    
    print(f"Local Dataset Path: {local_dataset_path}")

    def load_labels(label_path):
        with open(label_path, 'r') as file:
            data = file.readlines()
        labels = [list(map(float, line.strip().split())) for line in data]
        return labels

    def load_dataset(images_path, labels_path):
        image_files = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
        label_files = sorted(glob.glob(os.path.join(labels_path, "*.txt")))
        
        images = [cv2.imread(img) for img in image_files]
        labels = [load_labels(lbl) for lbl in label_files]
        
        return np.array(images), np.array(labels)

    # Load datasets
    train_images, train_labels = load_dataset(images_path, labels_path)
    
    # Save datasets as NumPy arrays
    np.save(f"{processed_dataset_root}/train_images.npy", train_images)
    np.save(f"{processed_dataset_root}/train_labels.npy", train_labels)
    
    # Create a new ClearML dataset for the NumPy files
    new_dataset = Dataset.create(dataset_name=f"{processed_dataset_name}ProcessedTrainDataset" , dataset_project=processed_dataset_project)

    # Add the NumPy files to the new dataset
    new_dataset.add_files(f"{processed_dataset_root}/train_images.npy")
    new_dataset.add_files(f"{processed_dataset_root}/train_labels.npy")
   
    # Upload the new dataset to ClearML
    new_dataset.upload()

    # Finalize the dataset
    new_dataset.finalize()

    # Clean up: Remove the numpy files after upload
    os.remove(f"{processed_dataset_root}/train_images.npy")
    os.remove(f"{processed_dataset_root}/train_labels.npy")


    print("New dataset with NumPy arrays has been created and uploaded to ClearML.")

    return dataset.id

@PipelineDecorator.component(name="ProcessValidData", return_values=["processed_valid_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_two_b(
    raw_valid_dataset_id, processed_dataset_project, processed_dataset_name, dataset_root, processed_dataset_root
):
    import os
    import numpy as np
    from clearml import Dataset
    import cv2
    import glob

    # Fetch the dataset from ClearML
    dataset = Dataset.get(raw_valid_dataset_id) #dataset_name=processed_dataset_name, dataset_project=processed_dataset_project)
    local_dataset_path = dataset.get_local_copy()

    # Define paths for train, valid, and test datasets
    images_path = os.path.join(local_dataset_path, "valid/images")
    labels_path = os.path.join(local_dataset_path, "valid/labels")
    
    print(f"Local Dataset Path: {local_dataset_path}")

    def load_labels(label_path):
        with open(label_path, 'r') as file:
            data = file.readlines()
        labels = [list(map(float, line.strip().split())) for line in data]
        return labels

    def load_dataset(images_path, labels_path):
        image_files = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
        label_files = sorted(glob.glob(os.path.join(labels_path, "*.txt")))
        
        images = [cv2.imread(img) for img in image_files]
        labels = [load_labels(lbl) for lbl in label_files]
        
        return np.array(images), np.array(labels)

    # Load datasets
    valid_images, valid_labels = load_dataset(images_path, labels_path)
    
    # Save datasets as NumPy arrays
    np.save(f"{processed_dataset_root}/valid_images.npy", valid_images)
    np.save(f"{processed_dataset_root}/valid_labels.npy", valid_labels)
    
    # Create a new ClearML dataset for the NumPy files
    new_dataset = Dataset.create(dataset_name=f"{processed_dataset_name}ProcessedValidDataset" , dataset_project=processed_dataset_project)

    # Add the NumPy files to the new dataset
    new_dataset.add_files(f"{processed_dataset_root}/valid_images.npy")
    new_dataset.add_files(f"{processed_dataset_root}/valid_labels.npy")
   
    # Upload the new dataset to ClearML
    new_dataset.upload()

    # Finalize the dataset
    new_dataset.finalize()

    # Clean up: Remove the numpy files after upload
    os.remove(f"{processed_dataset_root}/valid_images.npy")
    os.remove(f"{processed_dataset_root}/valid_labels.npy")


    print("New dataset with NumPy arrays has been created and uploaded to ClearML.")

    return dataset.id


@PipelineDecorator.component(name="ProcessTestData", return_values=["processed_test_dataset_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_two_c(
    raw_test_dataset_id, processed_dataset_project, processed_dataset_name, dataset_root, processed_dataset_root
):
    import os
    import numpy as np
    from clearml import Dataset
    import cv2
    import glob

    # Fetch the dataset from ClearML
    dataset = Dataset.get(raw_test_dataset_id) #dataset_name=processed_dataset_name, dataset_project=processed_dataset_project)
    local_dataset_path = dataset.get_local_copy()

    # Define paths for train, valid, and test datasets
    images_path = os.path.join(local_dataset_path, "test/images")
    labels_path = os.path.join(local_dataset_path, "test/labels")
    
    print(f"Local Dataset Path: {local_dataset_path}")

    def load_labels(label_path):
        with open(label_path, 'r') as file:
            data = file.readlines()
        labels = [list(map(float, line.strip().split())) for line in data]
        return labels

    def load_dataset(images_path, labels_path):
        image_files = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
        label_files = sorted(glob.glob(os.path.join(labels_path, "*.txt")))
        
        images = [cv2.imread(img) for img in image_files]
        labels = [load_labels(lbl) for lbl in label_files]
        
        return np.array(images), np.array(labels)

    # Load datasets
    test_images, test_labels = load_dataset(images_path, labels_path)
    
    # Save datasets as NumPy arrays
    np.save(f"{processed_dataset_root}/test_images.npy", test_images)
    np.save(f"{processed_dataset_root}/test_labels.npy", test_labels)
    
    # Create a new ClearML dataset for the NumPy files
    new_dataset = Dataset.create(dataset_name=f"{processed_dataset_name}ProcessedTestDataset" , dataset_project=processed_dataset_project)

    # Add the NumPy files to the new dataset
    new_dataset.add_files(f"{processed_dataset_root}/test_images.npy")
    new_dataset.add_files(f"{processed_dataset_root}/test_labels.npy")
   
    # Upload the new dataset to ClearML
    new_dataset.upload()

    # Finalize the dataset
    new_dataset.finalize()

    # Clean up: Remove the numpy files after upload
    os.remove(f"{processed_dataset_root}/test_images.npy")
    os.remove(f"{processed_dataset_root}/test_labels.npy")


    print("New dataset with NumPy arrays has been created and uploaded to ClearML.")

    return dataset.id




@PipelineDecorator.component(name="MergeDataTasks", return_values=["start_model_pipeline_id"], cache=True, task_type=TaskTypes.data_processing)#, execution_queue="default")
def step_three_merge(
     process_train_dataset_id, process_valid_dataset_id, process_test_dataset_id, processed_dataset_project, processed_dataset_name
):
    
    import numpy as np
    from clearml import Dataset, Task

    task = Task.current_task()

    task.connect({
        'process_train_dataset_id': process_train_dataset_id,
        'process_valid_dataset_id': process_valid_dataset_id,
        'process_test_dataset_id': process_test_dataset_id
    })
    
    return task.id



@PipelineDecorator.component(name="TrainModel", return_values=["training_task_id", "model_id"], cache=True, task_type=TaskTypes.training)#, execution_queue="default")
def step_four( start_model_pipeline_id, dataset_name, dataset_root, processed_dataset_root, results_dir, epochs
):
    import argparse
    import os
    from ultralytics import YOLO
    import numpy as np
    from clearml import Dataset, Task, OutputModel
    import cv2
    import torch
    import tensorflow
    import matplotlib.pyplot as plt
    import tempfile

    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")


    previous_task_id = start_model_pipeline_id
    previous_task = Task.get_task(task_id=previous_task_id)
    #
    ## Retrieve the dataset IDs
    process_train_dataset_id = previous_task.get_parameters()['General/process_train_dataset_id']
    process_valid_dataset_id = previous_task.get_parameters()['General/process_valid_dataset_id']
    process_test_dataset_id = previous_task.get_parameters()['General/process_test_dataset_id']

    
    
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    task = Task.current_task()
    logger = task.get_logger()
    # Train the model
    results = model.train(data='brainscan.yaml', device=device, epochs=epochs, imgsz=640, project=results_dir, name='brain_tumor_model')

    scores = model.val()
    #print('Test score:', scores[0])
    #print('Test accuracy:', scores[1])
    #logger.report_scalar(title='evaluate', series='score', value=scores[0], iteration=epochs)
    #logger.report_scalar(
    #    title="evaluate", series="accuracy", value=scores[1], iteration=epochs
    #)
    
    # Save the trained model weights
    model_output_path = os.path.join(tempfile.gettempdir())
    #model_store = ModelCheckpoint(filepath=os.path.join(output_folder, "weight.keras"))
    #model_output_path = os.path.join(results_dir, 'brain_tumor_model', 'weights', 'best.pt')

    model_file_path= f"{model_output_path}/best.pt"

    model.save(model_file_path)

    output_model = OutputModel(task=task)#, framework="PyTorch")

    output_model.update_weights(weights_filename=model_file_path , auto_delete_file=False)

    output_model.update_design('Model training completed')
    output_model.publish()
    # Log the model ID
    model_id = output_model.id
    print(f"Task Id: {task.id} Trained model ID: {model_id}")
    if os.path.exists(f"{model_file_path}"):
        os.remove(f"{model_file_path}")

    task.connect({
        'process_train_dataset_id': process_train_dataset_id,
        'process_valid_dataset_id': process_valid_dataset_id,
        'process_test_dataset_id': process_test_dataset_id,
        'output_model': model_id,
    })

    return task.id, model_id


#@PipelineDecorator.component(name="EvaluateModel", return_values=["train_model_task_id"], cache=True, task_type=TaskTypes.training)#, execution_queue="default")
def step_five(
    train_model_task_id, model_id, processed_dataset_project, processed_dataset_name, results_dir
):
    import os
    from ultralytics import YOLO
    from clearml import Task, Dataset, Model
    import numpy as np
    print(f"Task Id: {train_model_task_id} Trained model ID: {model_id}")
    # Load the trained model
    task = Task.get_task(task_id=train_model_task_id)
    model_saved = Model(model_id=model_id)
    model_path = model_saved.get_local_copy()
    print(f"Model path: {model_path} model_id: {model_id}")

    # Initialize the model
    model = YOLO(model_path)

    # Evaluate the model on the test dataset 
    results = model.val()

    print(f"Results: {results}")
    #task.get_logger().report_table("Test Results", "Test", iteration=0, table_plot=results.pandas().to_dict())
    return train_model_task_id
    

@PipelineDecorator.component(name="HPO", return_values=["top_experiment_id"], cache=False, task_type=TaskTypes.optimizer)#, execution_queue="uts-strykers-queue")
def step_six(
    train_model_task_id, queue_name
):
    #import pkgutil
    #import zipimport
    #pkgutil.ImpImporter = zipimport.zipimporter

    from clearml import Task
    from clearml.automation import HyperParameterOptimizer, UniformParameterRange
    from clearml.automation.optuna import OptimizerOptuna

    def job_complete_callback(
        job_id,  # type: str
        objective_value,  # type: float
        objective_iteration,  # type: int
        job_parameters,  # type: dict
        top_performance_job_id,  # type: str
        ):
        print(
            "Job completed!", job_id, objective_value, objective_iteration, job_parameters
        )
        if job_id == top_performance_job_id:
            print(
                "Objective reached {}".format(
                    objective_value
                )
            )


    task = Task.current_task()  
    # Create a HyperParameterOptimizer object
    # Define Hyperparameter Space
    param_ranges = [
        UniformIntegerParameterRange(
            "Args/epochs", min_value=3, max_value=5, step_size=5
        ),
        ### you could make anything like batch_size, number of nodes, loss function, a command line argument in base task and use it as a parameter to be optimised. ###
    ]
    optimizer = HyperParameterOptimizer(
        base_task_id=train_model_task_id,
        hyper_parameters=param_ranges,
        objective_metric_title="epoch_accuracy",
        objective_metric_series="epoch_accuracy",
        objective_metric_sign="max",
        optimizer_class=GridSearch,
        max_number_of_concurrent_tasks=2,
        optimization_time_limit=60.0,
        # Check the experiments every 6 seconds is way too often, we should probably set it to 5 min,
        # assuming a single experiment is usually hours...
        pool_period_min=0.1,
        compute_time_limit=120,
        total_max_jobs=20,
        min_iteration_per_job=3,
        max_iteration_per_job=5,
        execution_queue="default"
    )

    optimizer.set_report_period(0.2)

    optimizer.set_time_limit(in_minutes=10.0)
    # Start the optimization process
    optimizer.start(job_complete_callback=job_complete_callback)

    # Wait for the optimization to finish
    optimizer.wait()

    # optimization is completed, print the top performing experiments id
    top_exp = optimizer.get_top_experiments(top_k=3)
    print([t.id for t in top_exp])
    # make sure background optimization stopped
    optimizer.stop()


    #print("Optimisation Done")
    return top_exp[0].id

@PipelineDecorator.component(name="PushModel", return_values=["processed_train_dataset_id"], cache=False, task_type=TaskTypes.service, 
                             packages=['gitpython', 'python-dotenv']) # execution_queue="default")
def step_seven (
    step_six_id, train_model_id, environment_path, repo_url,  development_branch,  
    dataset_project, queue_name
):
    import argparse
    import datetime
    import os
    import shutil

    from clearml import Model, Task
    from dotenv import load_dotenv
    from git import GitCommandError, Repo

    def configure_ssh_key(DEPLOY_KEY_PATH):
        import argparse
        import datetime
        import os
        import shutil

        from clearml import Model
        from dotenv import load_dotenv
        from git import GitCommandError, Repo

        """Configure Git to use the SSH deploy key for operations."""
        os.environ["GIT_SSH_COMMAND"] = f"ssh -i {DEPLOY_KEY_PATH} -o IdentitiesOnly=yes"


    def clone_repo(REPO_URL, branch, DEPLOY_KEY_PATH) -> tuple[Repo, str]:
        import argparse
        import datetime
        import os
        import shutil

        from clearml import Model
        from dotenv import load_dotenv
        from git import GitCommandError, Repo

        """Clone the repository."""
        configure_ssh_key(DEPLOY_KEY_PATH)
        repo_path = REPO_URL.split("/")[-1].split(".git")[0]
        try:
            repo: Repo = Repo.clone_from(
                REPO_URL, repo_path, branch=branch, single_branch=True
            )
            print(repo_path)
            print("clone repo success")
            return repo, repo_path
        except GitCommandError as e:
            print(f"Failed to clone repository: {e}")
            exit(1)

    def ensure_archive_dir(repo: Repo):
        import argparse
        import datetime
        import os
        import shutil

        from clearml import Model
        from dotenv import load_dotenv
        from git import GitCommandError, Repo

        """Ensures the archive directory exists within weights."""
        archive_path = os.path.join(repo.working_tree_dir, "weights", "archive")
        os.makedirs(archive_path, exist_ok=True)

    def archive_existing_model(repo: Repo) -> str:
        import argparse
        import datetime
        import os
        import shutil

        from clearml import Model
        from dotenv import load_dotenv
        from git import GitCommandError, Repo

        """Archives existing model weights."""

        weights_path = os.path.join(repo.working_tree_dir, "weights")
        model_file = os.path.join(weights_path, "model.pt")
        if os.path.exists(model_file):
            today = datetime.date.today().strftime("%Y%m%d")
            archived_model_file = os.path.join(weights_path, "archive", f"model-{today}.pt")
            os.rename(model_file, archived_model_file)
            return archived_model_file  # Return the path of the archived file


    def update_weights(repo: Repo, model_path):
        import argparse
        import datetime
        import os
        import shutil

        from clearml import Model
        from dotenv import load_dotenv
        from git import GitCommandError, Repo

        """Updates the model weights in the repository."""

        print("working tree dir", repo.working_tree_dir)

        weights_path = os.path.join(repo.working_tree_dir, "weights")
        ensure_archive_dir(repo)
        archived_model_file = archive_existing_model(repo)
        target_model_path = os.path.join(weights_path, "model.pt")
        #print("model path: ", model_path, " target model path: ", target_model_path)
        shutil.move(model_path, target_model_path)  # Use shutil.move for cross-device move
        # Add the newly archived model file to the Git index
        
        repo.index.add([archived_model_file])
        # Also add the new model file to the Git index
        repo.index.add([target_model_path])

    def commit_and_push(repo: Repo, model_id, DEVELOPMENT_BRANCH):
        import argparse
        import datetime
        import os
        import shutil

        from clearml import Model
        from dotenv import load_dotenv
        from git import GitCommandError, Repo

        """Commits and pushes changes to the remote repository."""
        commit_message = f"Update model weights: {model_id}"
        tag_name = f"{model_id}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        try:
            repo.index.commit(commit_message)
            repo.create_tag(tag_name, message="Model update")
            repo.git.push("origin", DEVELOPMENT_BRANCH)
            repo.git.push("origin", "--tags")
        except GitCommandError as e:
            print(f"Failed to commit and push changes: {e}")
            exit(1)

    def cleanup_repo(repo_path):
        import argparse
        import datetime
        import os
        import shutil

        from clearml import Model, Task
        from dotenv import load_dotenv
        from git import GitCommandError, Repo

        """Safely remove the cloned repository directory."""
        shutil.rmtree(repo_path, ignore_errors=True)


    print("model id", train_model_id)
    print("env path", environment_path)
    print("repo url", repo_url)
    print("development branch", development_branch)
    print("project name", dataset_project)

    #task = Task.current_task()
    #task.execute_remotely(queue_name=queue_name, exit_process=True)
    
    """Fetches the trained model using its ID and updates it in the repository."""
    load_dotenv(dotenv_path=environment_path)
    DEPLOY_KEY_PATH = os.getenv("DEPLOY_KEY_PATH")
    print("deploy key path", DEPLOY_KEY_PATH)

    # Prepare repository and SSH key
    repo, repo_path = clone_repo(repo_url, development_branch, DEPLOY_KEY_PATH)
    try:
        # Fetch the trained model
        model = Model(model_id=train_model_id)
        model_path = model.get_local_copy()

        # Update weights and push changes
        update_weights(repo, model_path)
        print("update weights success")
        commit_and_push(repo, train_model_id, development_branch)
        print("commit success")
    finally:
        print("beginning cleanup")
        cleanup_repo(repo_path)  # Ensure cleanup happens even if an error occurs
        print("cleanup success")




   
# The actual pipeline execution context
# notice that all pipeline component function calls are actually executed remotely
# Only when a return value is used, the pipeline logic will wait for the component execution to complete
@PipelineDecorator.pipeline(name="BrainScan2DataPipeline", project="BrainScan2", target_project="BrainScan2", pipeline_execution_queue="uts-strykers-queue", default_queue="uts-strykers-queue") #, version="0.0.6")
def executing_data_pipeline(dataset_project, dataset_name, dataset_root, 
                            processed_dataset_root, output_root, queue_name, results_dir,
                            environment_path,
                            development_branch,
                            repo_url, epochs):

    start_data_pipeline_id = start_data_pipeline(dataset_project, dataset_name, dataset_root)

    raw_train_dataset_id = step_one_a(start_data_pipeline_id, dataset_project, dataset_name, dataset_root)

    
    raw_validation_dataset_id = step_one_b(start_data_pipeline_id, dataset_project, dataset_name, dataset_root)
    
    
    raw_test_dataset_id = step_one_c(start_data_pipeline_id, dataset_project, dataset_name, dataset_root)
    
    
    process_train_dataset_id = step_two_a(raw_train_dataset_id, dataset_project, dataset_name, dataset_root, processed_dataset_root)
    
    
    process_valid_dataset_id = step_two_b(raw_validation_dataset_id, dataset_project, dataset_name, dataset_root, processed_dataset_root)
    
    
    process_test_dataset_id = step_two_c(raw_test_dataset_id, dataset_project, dataset_name, dataset_root, processed_dataset_root)

    start_model_pipeline_id = step_three_merge(process_train_dataset_id, process_valid_dataset_id, process_test_dataset_id, dataset_project, dataset_name)
    
    
    step_four_id, model_id = step_four(start_model_pipeline_id, dataset_name, dataset_root, processed_dataset_root, results_dir, epochs)

    
    #step_five_id = step_five(step_four_id, model_id, dataset_name, dataset_root, results_dir)

    #step_six_id = step_six(step_five_id, queue_name)

    #step_seven_id = step_seven(step_five_id, model_id, environment_path, repo_url, development_branch, dataset_project, queue_name="default")

   



if __name__ == "__main__":
    # set the pipeline steps default execution queue (per specific step we can override it with the decorator)
    PipelineDecorator.set_default_execution_queue('uts-strykers-queue')
    # Run the pipeline steps as subprocesses on the current machine, great for local executions
    # (for easy development / debugging, use `PipelineDecorator.debug_pipeline()` to execute steps as regular functions)
    PipelineDecorator.run_locally()
    #PipelineDecorator.debug_pipeline()
    # Start the pipeline execution logic.
    
    #base_path = "/Users/soterojrsaberon/BlastAsia/braintumour-ml/brainscan2"
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    executing_data_pipeline(
        dataset_project="BrainScan2",
        dataset_name="BrainScan2",
        dataset_root=f"{base_path}/datasets/brain-tumor",
        processed_dataset_root=f"{base_path}/datasets/processed",
        output_root=f"{base_path}/output",
        queue_name="uts-strykers-queue", 
        results_dir=f"{base_path}/models",
        environment_path=f"{base_path}/.env",
        development_branch="development",
        repo_url="git@github.com:uts-strykers/braintumour-api.git",
        epochs=1
    )

     

    print("process completed")

