import subprocess
import os
import requests
import os
import shutil
import boto3


def hit_api(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("API request successful")
            return response.json()
        else:
            response = f"Error: API request failed with status code {response.status_code}"
            return response
    except requests.RequestException as e:
        response = f"Error: {e}"
        return response
def download_images(command):
    try:
        subprocess.run(command, check=True, shell=True)
        print("Images downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
def preprocess(command):
    try:
        subprocess.run(command, check=True, shell=True)
        print("Mask created successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        
def train_script(command):
    try:
        subprocess.run(command, check=True, shell=True)
        print("Training Completed")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

def update_data(api_url,patch_data):
    response = requests.patch(api_url, json=patch_data)
    headers = {
    "Content-Type": "application/json"
    }

    response = requests.patch(api_url, json=patch_data, headers=headers)


    # Check if the request was successful (status code 2xx)
    if response.status_code // 100 == 2:
        print(f"PATCH request successful. Response: {response.text}")
    else:
        print(f"PATCH request failed. Status Code: {response.status_code}, Response: {response.text}")

def upload_folder_to_s3_and_remove(local_folder_path, s3_bucket_name, s3_folder_name):
    s3 = boto3.client('s3', aws_access_key_id="AKIAXJJIYLL3OGXGRT6X", aws_secret_access_key="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")
    for root, dirs, files in os.walk(local_folder_path):
        for filename in files:
            local_file_path = os.path.join(root, filename)
            s3_file_path = os.path.relpath(local_file_path, local_folder_path)
            s3_file_path = os.path.join(s3_folder_name, s3_file_path)
            s3.upload_file(local_file_path, s3_bucket_name, s3_file_path)
    
    # Remove the local folder after uploading
    # shutil.rmtree(local_folder_path)
    # print(f"Deleted local folder: {local_folder_path}")
    



if __name__ == "__main__":
    #Take Data from api
    import json
    while True:
        try:
            api_url = 'http://stylin-prod.ap-south-1.elasticbeanstalk.com/user/api/image/training/'
            response_data = hit_api(api_url)
            image_url = response_data['image_data']
            ProductID = response_data["product_id"]
            image_url = json.dumps(image_url)
            ProductName = response_data['product_name']
            ListedProducts = ['blazers', 'casual_shirts', 'casual_trousers', 'formal_trousers', 'jackets', 'jeans', 'casual_shoes', 'formal_shirts', 'sandals', 'belts', 'sweaters', 'formal_shoes', 'sweatshirts', 'sunglasses', 'sports_shoes', 'track_pants', 'shorts', 't_shirts', 'coats', 'suits', 'cummerband', 'boots', 'flip_flops', 'kurta_sets', 'pyjamas', 'kurtas', 'nehru_jackets', 'sherwanis', 'track_pants_and_joggers', 'patiala_pants', 'active_t_shirts', 'tracksuits', 'Co_Ords', 'track_pants_and_shorts', 'churidars', 'salwars', 'dhoti_pants', 'shackets']
            InpaintingProducts = ['blazers', 'shirts', 'trousers', 'jackets', 'jeans', 'sweaters','sweatshirts','track_pants', 'shorts', 't_shirts', 'coats', 'suits', 'kurta_sets', 'pyjamas', 'kurtas', 'nehru_jackets', 'sherwanis', 'track_pants_and_joggers', 'patiala_pants', 'active_t_shirts', 'tracksuits', 'track_pants_and_shorts', 'churidars', 'salwars', 'dhoti_pants', 'shackets']
            if ProductName in InpaintingProducts:
                ExcludedProducts = ['jeans','shoes','sunglasses']
                if ProductName in ExcludedProducts:
                    ProductName = ProductName.capitalize()
                else:
                    ProductName = ProductName[:-1]
                    if ProductName == 't_shirt' or ProductName == 'active_t_shirt':
                        ProductName = 'T-shirt'
                    elif ProductName == 'track_pant':
                        ProductName = 'Trouser'
                    else:
                        ProductName = ProductName.capitalize()

                print("+++++++++++++++++++++++++++++++++++++++++++",ProductName)
                Prompt = response_data['product_description']


                #Download Script
                out_path = 'ProductImages/'+str(ProductID)
                isExist = os.path.exists(out_path)
                if os.path.exists(out_path):
                    print("Product is trained")
                else:
                    os.mkdir(out_path)
                download_command = f"python download_prod.py '{image_url}' --output-directory {out_path}"
                download_images(download_command)

                # #Preprocess Script
                print('-------------------',ProductName)
                preprocess_command = f'python preprocess_sam.py --instance_data_dir {out_path} --instance_prompt {ProductName}'
                preprocess(preprocess_command)

                # #Traning Script
                model_path = f'models/{str(ProductID)}'
                if os.path.exists(model_path):
                    print("Product is trained")
                else:
                    os.mkdir(model_path)

                print("this is prompt ------------------",Prompt)
                accelerate_command = f'accelerate launch --num_processes 1 train_lora_org.py --instance_data_dir {out_path} --output_dir {model_path} --instance_prompt "{Prompt}" --resolution=512 --train_batch_size=1 --learning_rate=1e-4   --max_train_steps=2000   --checkpointing_steps 1000'
            
            
                try:
                    train_script(accelerate_command)
                # ## Upload to S3 and remove from local
                except:
                    print('Error in training')
                    break
                local_folder_path = f"models/{str(ProductID)}"
                s3_bucket_name = "static.styl.in"
                s3_folder_name = f"models/V1/{str(ProductID)}"
                upload_folder_to_s3_and_remove(local_folder_path, s3_bucket_name, s3_folder_name)
                ProductID = str(ProductID)
                ai_model_version_no = ProductID+'_V.0.1'

                patch_data = {
                    "ai_model_version_no":ai_model_version_no,
                    "ai_model_generated":True,
                    "product_id":ProductID
                }

                update_data(api_url,patch_data)
        except:
            print("Pipeline failed ")
            break
        else:
            print("Try on not available for this product")
