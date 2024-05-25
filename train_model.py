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

def train_model( project_name, task_name,  epochs, args=None):
   
    task: Task = Task.init(
        project_name=project_name,
        task_name=task_name)
   
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

    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    task = Task.current_task()
    logger = task.get_logger()
    # Train the model
    results = model.train(data='brainscan.yaml', device=device, epochs=epochs, imgsz=640, project=results_dir, name='brain_tumor_model')

    scores = model.val()
    
    # Save the trained model weights
    model_output_path = os.path.join(tempfile.gettempdir())

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

    return task.id, model_id

   

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
        args.project_name,
        args.task_name,
        args.epochs,        
        args=args
    )