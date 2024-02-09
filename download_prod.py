import argparse
import subprocess
import json

def download_images(image_urls, output_directory='.'):
    for url in image_urls:
        try:
            # Construct the wget command
            wget_command = ['wget', '-P', output_directory, url]

            # Run the wget command
            subprocess.run(wget_command, check=True)

            # print(f"Downloaded: {url}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {url}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download images from a list of URLs.')
    parser.add_argument('image_urls', nargs=1, help='List of image URLs to download as a JSON-formatted string')
    parser.add_argument('--output-directory', '-o', default='.', help='Output directory for downloaded images')

    args = parser.parse_args()

    # Convert the JSON-formatted string to a list of URLs
    try:
        image_urls = json.loads(args.image_urls[0])
    except json.JSONDecodeError:
        print("Error: Invalid JSON format for image URLs.")
        return

    # Create the output directory if it doesn't exist
    subprocess.run(['mkdir', '-p', args.output_directory])

    # Download the images
    download_images(image_urls, args.output_directory)

if __name__ == "__main__":
    main()


# python script_name.py '["url1", "url2", "url3"]' --output-directory 'data/trouser'
