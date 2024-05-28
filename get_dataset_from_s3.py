""" import boto3
import os
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
"""

def get_path_for_saved_model():
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a directory for saving the model
    save_dir = os.path.join(root_dir, 'mydata')
    os.makedirs(save_dir, exist_ok=True)

    # Define the path to save the model
  
    return  save_dir

"""
def download_s3_folder(bucket_name, s3_folder, local_dir):
    try:
        # Initialize a session using Amazon S3
        s3 = boto3.client('s3')

        # Ensure the local directory exists
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # List objects within the specified folder
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder)

        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Form the full local path
                    local_file_path = os.path.join(local_dir, os.path.relpath(obj['Key'], s3_folder))
                    print(local_file_path)
                    # Ensure the local directory exists
                    if not os.path.exists(os.path.dirname(local_file_path)):
                        os.makedirs(os.path.dirname(local_file_path))

                    # Download the file
                    print(f"Downloading {obj['Key']} to {local_file_path}")
                    s3.download_file(bucket_name, obj['Key'], local_file_path)
                    print(f"Downloaded {obj['Key']} to {local_file_path}")

    except NoCredentialsError:
        print("Error: AWS credentials not found.")
    except PartialCredentialsError:
        print("Error: Incomplete AWS credentials provided.")
    except ClientError as e:
        print(f"Error: {e}")


# Download the S3 folder
    download_s3_folder(bucket_name, s3_folder, local_dir)
    
 
if __name__ == "__main__":
    bucket_name = 'uts-aws-storage'
    s3_folder = 'strykers/dataset/datasets'
    local_dir = get_path_for_saved_model()
    print(local_dir)
        
    download_s3_folder(bucket_name, s3_folder, local_dir) """
import boto3
import os
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

def download_s3_folder(bucket_name, s3_folder, local_dir):
    try:
        # Initialize a session using Amazon S3
        s3 = boto3.client('s3',
                          aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                          aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))

        
        # Ensure the local directory exists
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # List objects within the specified folder (prefix)
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder)

        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Form the full local path
                    local_file_path = os.path.join(local_dir, os.path.relpath(obj['Key'], s3_folder))

                    # Ensure the local directory exists
                    if not os.path.exists(os.path.dirname(local_file_path)):
                        os.makedirs(os.path.dirname(local_file_path))

                    # Download the file
                    print(f"Downloading {obj['Key']} to {local_file_path}")
                    s3.download_file(bucket_name, obj['Key'], local_file_path)
                    print(f"Downloaded {obj['Key']} to {local_file_path}")

    except NoCredentialsError:
        print("Error: AWS credentials not found.")
    except PartialCredentialsError:
        print("Error: Incomplete AWS credentials provided.")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDenied':
            print("Error: Access Denied. Please check your AWS permissions.")
        else:
            print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

# Define your bucket name, S3 folder (prefix), and local directory
bucket_name = 'uts-aws-storage'
s3_folder = 'strykers/Dataset/datasets'
local_dir = get_path_for_saved_model()
print(local_dir)
        

# Download the S3 folder
download_s3_folder(bucket_name, s3_folder, local_dir)