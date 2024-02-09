import os
import shutil
import boto3

def upload_folder_to_s3_and_remove(local_folder_path, s3_bucket_name, s3_folder_name):
    s3 = boto3.client('s3', aws_access_key_id="AKIAXJJIYLL3OGXGRT6X", aws_secret_access_key="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")
    for root, dirs, files in os.walk(local_folder_path):
        for filename in files:
            local_file_path = os.path.join(root, filename)
            s3_file_path = os.path.relpath(local_file_path, local_folder_path)
            s3_file_path = os.path.join(s3_folder_name, s3_file_path)
            s3.upload_file(local_file_path, s3_bucket_name, s3_file_path)
    
    # Remove the local folder after uploading
    shutil.rmtree(local_folder_path)
    print(f"Deleted local folder: {local_folder_path}")

# Example usage
local_folder_path = "models/444764"
s3_bucket_name = "static.styl.in"
s3_folder_name = f"s3://static.styl.in/models/444764"

upload_folder_to_s3_and_remove(local_folder_path, s3_bucket_name, s3_folder_name)
