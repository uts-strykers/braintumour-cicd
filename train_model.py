import os
import numpy as np
import pandas as pd
import clearml
from clearml.automation.controller import PipelineDecorator
import argparse
import tempfile
from ultralytics import YOLO
from clearml import Dataset, Task, OutputModel, TaskTypes
import torch
import yaml

def create_data_yaml():

    # Define the class names and number of classes
    class_names = ['negative', 'positive']
    nc = len(class_names)

    # Get the root directory of the script
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the paths for train and validation images
    train_path = 'train/images'
    val_path =  'valid/images'
    test_path =  'test/images'

    # Create the data dictionary
    data = {
        'path': os.path.join(root_dir,"mydata", "brain-tumor"),
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': nc,
        'names': class_names
    }

    # Define the path to save the data.yaml file
    yaml_path = os.path.join(root_dir, 'data.yaml')

    # Save the data dictionary to a yaml file (overwrite if it exists)
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

    print(f"data.yaml file has been created at: {yaml_path}")

def get_path_for_saved_model():
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a directory for saving the model
    save_dir = os.path.join(root_dir, 'models')
    os.makedirs(save_dir, exist_ok=True)

    # Define the path to save the model
    save_path = os.path.join(save_dir, 'best.pt')
    return save_path, save_dir

def train_model(task, project_name="BrainScan2", task_name="TrainModel", 
                results_dir="/Users/soterojrsaberon/BlastAsia/braintumour-ml/brainscan2/models", 
                epochs = 2, args=None):
    #epochs = 2
    #results_dir = "/Users/soterojrsaberon/BlastAsia/braintumour-ml/brainscan2/models"
    #task: Task = Task.init(
    #    project_name=project_name,
    #    task_name=task_name)
    
    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    
    #    print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    #    print("Using MPS")
    else:
        device = torch.device('cpu')
    #    print("Using CPU")

    model_file_path, project_dir = get_path_for_saved_model()

    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    task = Task.current_task()
    logger = task.get_logger()

    create_data_yaml()
    # Train the model
    results = model.train(data='data.yaml', project=project_dir, device=device, epochs=epochs, imgsz=640, name='brain_tumor_model')

    scores = model.val()
    
    # Save the trained model weights
    #model_output_path = os.path.join(tempfile.gettempdir())

    #model_file_path= f"{model_output_path}/best.pt"
    #model_file_path = get_path_for_saved_model()
    
    #root_dir = os.path.dirname(os.path.abspath(__file__))
    model.save(model_file_path)

    output_model = OutputModel(task=task)#, framework="PyTorch")

    output_model.update_weights(weights_filename=model_file_path , auto_delete_file=False)

    output_model.update_design('Model training completed')
    output_model.publish()
    # Log the model ID
    model_id = output_model.id
    #print(f"Task Id: {task.id} Trained model ID: {model_id}")
    if os.path.exists(f"{model_file_path}"):
        os.remove(f"{model_file_path}")

    #return task.id

   
 
if __name__ == "__main__":
    base_path = "/Users/soterojrsaberon/BlastAsia/braintumour-ml/brainscan2"
    
    results_dir=f"{base_path}/models"

    parser = argparse.ArgumentParser(
        description="Train BrainScan data with YoloV8."
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="BrainScan",
        help="Name for the project",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="TrainModel",
        help="Name for the task",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=results_dir,
        help="Results directdory",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of training epochs"
    )
    args = parser.parse_args()

    train_model(
        project_name=args.project_name,
        task_name=args.task_name,
        epochs=args.epochs,  
        results_dir=args.results_dir,      
        args=args
    ) 