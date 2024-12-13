import argparse
import os
import csv
from skingpt4.common.config import Config
from skingpt4.common.registry import registry

# Define the argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Batch process images with SkinGPT4.")
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use for model inference.")
    return parser.parse_args()

# Initialize the chat object
def initialize_chat(cfg_path, gpu_id):
    config = Config(cfg_path)
    registry.register("config", config)
    config.merge_with_args({"gpu_id": gpu_id})
    chat = registry.get_model("Chat")
    return chat

# Process images and save results
def process_images(chat):
    images_path = "images"  # Default path for images directory
    output_csv = "output.csv"  # Default path for output CSV file

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images directory not found: {images_path}")

    image_files = [f for f in os.listdir(images_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not image_files:
        raise FileNotFoundError("No valid image files found in the directory.")

    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Image Name", "Output"])

        for image_file in image_files:
            image_path = os.path.join(images_path, image_file)

            try:
                chat.upload_img(image_path)
                response = chat.ask("Analyze this skin condition.")
                output = response.get("answer", "No answer available.")
            except Exception as e:
                output = f"Error: {str(e)}"

            writer.writerow([image_file, output])
            print(f"Processed {image_file}: {output}")

# Main function
def main():
    args = parse_args()

    chat = initialize_chat(args.cfg_path, args.gpu_id)
    process_images(chat)

if __name__ == "__main__":
    main()
