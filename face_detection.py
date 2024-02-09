import face_recognition
from PIL import Image
import os
import cv2

def detect_and_crop_face(source_image_path, target_image, output_image_path):
    # Load the source image and find face locations
    
    source_image = face_recognition.load_image_file(source_image_path)
    # source_image = source_image.resize(512,512)
    height, width, channels = source_image.shape

    source_image = cv2.resize(source_image,(512,512))
    source_face_locations = face_recognition.face_locations(source_image)

    if len(source_face_locations) == 0:
        print("No face found in the source image.")
        return

    # Get the face location in the source image
    top, right, bottom, left = source_face_locations[0]

    # Calculate the face width and height
    face_width = right - left
    face_height = bottom - top

    # Load the target image
    # target_image = Image.open(target_image_path)
    target_image = target_image

    # Crop the face from the source image
    source_face = Image.fromarray(source_image[top:bottom, left:right])

    # Resize the source face to match the size in the target image
    source_face = source_face.resize((face_width, face_height))

    # Paste the source face onto the target image at the original face location
    target_image.paste(source_face, (left, top))


    # Create the output directory if it doesn't exist
    # os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    # Save the result
    target_image = target_image.resize((width, height))
    # target_image.save(output_image_path)

    print(f"Face pasted successfully. Result saved to {output_image_path}")
    return target_image

# # Example usage:
# source_image_path = 'path/to/source/image.jpg'  # Replace with the path to your source image
# target_image_path = 'path/to/target/image.jpg'  # Replace with the path to your target image
# output_image_path = 'output/result_image.jpg'  # Specify the output path



# Example usage:
# source_image_path = 'data/test_img/IMG_8031-PhotoRoom.jpg'  # Replace with the path to your source image
# target_image_path = 'out/Test_Img/IMG_8031-PhotoRoom.jpg'  # Replace with the path to your target image
# output_image_path = 'output/result_image3.jpg'  # Specify the output path

# detect_and_crop_face(source_image_path, target_image_path, output_image_path)
