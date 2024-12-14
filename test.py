import os
from PIL import Image

# Process images and save results
def process_images(image_folder):

    results = []

    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
            image_path = os.path.join(image_folder, image_file)
            print(f"Processing: {image_path}")

            try:
                # Open the image using PIL
                raw_image = Image.open(image_path)

                # Check if the image is in RGB format
                if raw_image.mode == 'RGB':
                    print(f"Image {image_file} is in RGB format.")
                else:
                    print(f"Image {image_file} is not in RGB format. It is in {raw_image.mode} format.")
                    
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

print('Processing Images')
IMAGE_FOLDER = "images"  # Update this with your folder path
OUTPUT_CSV = "output_results.csv"
process_images(IMAGE_FOLDER)
print('Processed Images')
