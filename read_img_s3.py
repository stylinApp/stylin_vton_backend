import boto3
from io import BytesIO
import os

# Replace 'your_access_key' and 'your_secret_key' with your AWS access key and secret key
s3 = boto3.client('s3', aws_access_key_id='your_access_key', aws_secret_access_key='your_secret_key')

def save_image_locally(s3_url, local_folder):
    # Extracting bucket and key from the S3 URL
    bucket, key = s3_url.split('/')[2], '/'.join(s3_url.split('/')[3:])

    # Download image from S3
    response = s3.get_object(Bucket=bucket, Key=key)
    image_data = response['Body'].read()

    # Save image locally
    local_filename = os.path.join(local_folder, os.path.basename(key))
    with open(local_filename, 'wb') as local_file:
        local_file.write(image_data)

    return local_filename

# Example usage
s3_image_url = 'https://your-s3-bucket.s3.amazonaws.com/path/to/your/image.jpg'
local_folder_path = 'local_folder'

local_file_path = save_image_locally(s3_image_url, local_folder_path)
print(f"Image saved to: {local_file_path}")
