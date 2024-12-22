import argparse
import os
import random
import csv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont
from skingpt4.common.config import Config
from skingpt4.common.dist_utils import get_rank
from skingpt4.common.registry import registry
from skingpt4.conversation.conversation import Chat, CONV_VISION

# Imports modules for registration
from skingpt4.datasets.builders import *
from skingpt4.models import *
from skingpt4.processors import *
from skingpt4.runners import *
from skingpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="specify the GPU to load the model."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecate), change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def add_caption_to_image(image_path, caption, output_folder):
    """Add a caption to the bottom of an image and save the modified image with multi-line support."""
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Use PIL default font
        font = ImageFont.load_default()

        # Calculate text size and wrap the caption into multiple lines if needed
        max_width = img.width - 20  # Margin for padding
        lines = []
        current_line = ""

        for word in caption.split():
            # Add word to current line
            test_line = f"{current_line} {word}".strip()
            # Check if the line is too wide
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]

            if text_width <= max_width:
                current_line = test_line  # Continue adding to the current line
            else:
                # If the line is too wide, start a new line
                if current_line:
                    lines.append(current_line)
                current_line = word

        # Add the last line
        if current_line:
            lines.append(current_line)

        # Calculate total text height
        text_height = sum(
            [
                draw.textbbox((0, 0), line, font=font)[3]
                - draw.textbbox((0, 0), line, font=font)[1]
                for line in lines
            ]
        )

        # Create a new image with extra space for the caption
        new_height = (
            img.height + text_height + len(lines) * 10 + 20
        )  # Add padding between lines
        new_img = Image.new("RGB", (img.width, new_height), (255, 255, 255))
        new_img.paste(img, (0, 0))
        draw = ImageDraw.Draw(new_img)
        # Draw the multi-line caption at the bottom center
        text_y = img.height + 10  # Padding to the bottom
        for line in lines:
            # Calculate the position for each line (centered)
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (img.width - text_width) // 2  # Center the text horizontally
            draw.text((text_x, text_y), line, font=font, fill="black")
            text_y += bbox[3] - bbox[1] + 10  # Move down for the next line

        # Save the new image
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        new_img.save(output_path)
        print(f"Caption added and saved to {output_path}")
    except Exception as e:
        print(f"Error adding caption to {image_path}: {e}")


def process_images_from_csv(csv_file, chat, output_csv, output_folder):
    """Process all images in a CSV file and save results to a CSV and images with captions."""
    conv = CONV_VISION.copy()

    results = []

    # Read the CSV file and get the image paths
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # Grabbing only the filename from the CSV (the first column)
            image_filename = row[0].split("/")[-1]  # This extracts just the file name from the path
            image_path = os.path.join("scin_images", image_filename)  # Build the path to the file in the 'scin_images' folder
            
            if os.path.exists(image_path):
                conv = CONV_VISION.copy()
                print(f"Processing: {image_path}")

                # Upload the image and ask the question
                img_list = []
                chat.upload_img(image=image_path, conv=conv, img_list=img_list)
                chat.ask(
                    "Could you describe the skin disease in this image for me?", conv
                )

                # Get the model's answer
                response, _ = chat.answer(
                    conv=conv, img_list=img_list, max_new_tokens=300
                )

                # Store the result
                results.append({"Image": image_path, "Description": response})

                # Add caption to image
                add_caption_to_image(image_path, response, output_folder)

    # Write the results to a CSV file
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Image", "Description"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_csv}")


print("Initializing Chat")
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to("cuda:{}".format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
    vis_processor_cfg
)
chat = Chat(model, vis_processor, device="cuda:{}".format(args.gpu_id))
print("Initialization Finished")
print("Processing Images")

# Input and output paths
CSV_FILE = "sampled_image_ids.csv"  # Path to your CSV file
OUTPUT_CSV = "output_results.csv"
OUTPUT_FOLDER = "output_images"  # Folder for images with captions

# Process images from CSV
process_images_from_csv(CSV_FILE, chat, OUTPUT_CSV, OUTPUT_FOLDER)
print("Processed Images")
