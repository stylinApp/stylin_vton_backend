import subprocess

def download_images(image_urls, output_directory='.'):
    for url in image_urls:
        try:
            # Construct the wget command
            wget_command = ['wget', '-P', output_directory, url]
            
            # Run the wget command
            subprocess.run(wget_command, check=True)
            
            print(f"Downloaded: {url}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    # Replace the following list with the URLs of the images you want to download
    image_urls = [
        'https://assets.myntassets.com/h_720,q_90,w_540/v1/assets/images/13802596/2024/1/30/afe6d2a2-6561-4e9b-8e82-ab0fee4797f21706615444844-Nautica-Men-Pure-Cotton-Colourblocked-Round-Neck-T-shirt-374-1.jpg',
        'https://assets.myntassets.com/h_720,q_90,w_540/v1/assets/images/13802596/2024/1/30/f56926e6-4e39-4b9c-bf8f-980d7a00da641706615444810-Nautica-Men-Pure-Cotton-Colourblocked-Round-Neck-T-shirt-374-6.jpg'
        
    ]

    # Specify the directory where you want to save the images (default is the current directory)
    output_directory = 'data/upper_wear_nautica'

    # Create the output directory if it doesn't exist
    subprocess.run(['mkdir', '-p', output_directory])

    # Download the images
    download_images(image_urls, output_directory)
