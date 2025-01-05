import argparse
import os
import random
import csv
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
from typing import Union, Tuple, Optional
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont
import logging

# imports modules for registration
from skingpt4.common.config import Config
from skingpt4.common.dist_utils import get_rank
from skingpt4.common.registry import registry
from skingpt4.conversation.conversation import Chat, CONV_VISION
from skingpt4.tasks import *
from skingpt4.runners import *
from skingpt4.processors import *
from skingpt4.models import *
from skingpt4.datasets.builders import *
from skingpt4.classification.classification_task import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("skingpt4_processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments for the SkinGPT-4 demo.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="specify the gpu to load the model."
    )
    args = parser.parse_args()
    logger.info("Parsed arguments: %s", args)
    return args

def load_model(checkpoint_path, device):
    """
    Load and initialize the classification model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint file
        device (str): Device to load the model on ('cuda:0', 'cpu', etc.)

    Returns:
        ClassificationTask: Loaded and initialized model in evaluation mode
    """
    model = ClassificationTask.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, target_size=(810, 1080)):
    """
    Preprocess an image for model inference.

    Args:
        image_path (str): Path to the input image file
        target_size (tuple): Desired size for the image (width, height), default: (810, 1080)

    Returns:
        tuple: Contains:
            - PIL.Image: Original image opened in RGB mode
            - torch.Tensor: Preprocessed image tensor of shape (1, 3, height, width)
    """
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, image_tensor

def predict(model, image_tensor, device):
    """
    Perform model inference on a preprocessed image tensor.

    Args:
        model (ClassificationTask): The loaded classification model
        image_tensor (torch.Tensor): Preprocessed image tensor of shape (1, 3, height, width)
        device (str): Device to perform inference on ('cuda:0', 'cpu', etc.)

    Returns:
        tuple: Contains:
            - int: Predicted class index
            - numpy.ndarray: Softmax probabilities for each class
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        _, predicted_class = torch.max(output, 1)
        predicted_class = predicted_class.item()
    return predicted_class, probabilities


def process_single_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (512, 512),
) -> Optional[Image.Image]:
    """
    Process a single image for LLM input.

    Args:
        image_path: Path to the image file
        target_size: Desired output size (width, height)

    Returns:
        PIL.Image or None if processing fails
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error("Image file does not exist: %s", image_path)
            return None

        logger.info("Processing image: %s", image_path)

        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
                logger.debug("Converted image mode to RGB")

            # Calculate resize dimensions maintaining aspect ratio
            orig_width, orig_height = img.size
            ratio = min(target_size[0] / orig_width, target_size[1] / orig_height)
            new_size = (int(orig_width * ratio), int(orig_height * ratio))

            logger.debug("Resizing from %s to %s", img.size, new_size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Create new image with padding
            new_img = Image.new("RGB", target_size, (0, 0, 0))
            paste_x = (target_size[0] - new_size[0]) // 2
            paste_y = (target_size[1] - new_size[1]) // 2
            new_img.paste(img, (paste_x, paste_y))

            logger.info("Successfully processed image to size %s", target_size)
            return new_img

    except Exception as e:
        logger.error("Error processing image %s: %s", image_path, str(e))
        return None


def process_images(
    csv_file: str, model: any, output_csv: str, output_folder: str, device: str
) -> None:
    """
    Process all images in a folder and save results to a CSV.

    Args:
        image_folder: Directory containing source images
        chat: Chat instance for LLM processing
        output_csv: Path to output CSV file
    """
    logger.info("Starting batch processing of images from %s", csv_file)

    results = []

    try:
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                image_filename = row[0].split("/")[
                    -1
                ]  # This extracts just the file name from the path
                image_path = os.path.join(
                    "images", "test", image_filename
                )  # Build the path to the file in the 'scin_images' folder

                if os.path.exists(image_path):
                    conv = CONV_VISION.copy()
                    print(f"Processing: {image_path}")

                    try:
                        processed_image = process_single_image(image_path)
                        if processed_image is None:
                            continue

                        img_list = []

                        _, image_tensor = preprocess_image(image_path)

                        predicted_class, probabilities = predict(model, image_tensor, device)
                        print(predicted_class.type())
                        
                        results.append({"Image": image_path, "Class": predicted_class+1})

                        logger.debug("Successfully processed %s", image_path)

                    except Exception as e:
                        logger.error("Error processing %s: %s", image_path, str(e))
                        continue

        # Save results
        if results:
            try:
                with open(
                    output_csv, mode="w", newline="", encoding="utf-8"
                ) as csvfile:
                    writer = csv.DictWriter(
                        csvfile, fieldnames=["Image", "Class"]
                    )
                    writer.writeheader()
                    writer.writerows(results)
                logger.info("Results saved to %s", output_csv)
            except Exception as e:
                logger.error("Error saving results to CSV: %s", str(e))
        else:
            logger.warning("No results to save")

    except Exception as e:
        logger.error("Fatal error in process_images: %s", str(e))
        raise


def main():
    """
    Main function to run the SkinGPT-4 demo application.
    """
    logger.info("Starting SkinGPT-4 demo application")
    chat = None
    device = None

    try:
        args = parse_args()
        device = f"cuda:{args.gpu_id}" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
        
        # Setup device
        if torch.cuda.is_available():
            device = f"cuda:{args.gpu_id}"
            torch.cuda.set_device(args.gpu_id)
        else:
            device = "cpu"
            logger.warning("CUDA not available, using CPU")

        logger.info("Initializing Chat on device: %s", device)

        # Initialize model
        model = load_model("/workspace/archive/results/pair_5_clinicals/ckpts/epoch=15-step=256.ckpt", device)
        logger.info("Loaded model")

        # Process images
        csv_file = "sampled_image.csv"
        output_csv = "output_results.csv"
        output_folder = "output_images"

        process_images(csv_file, model, output_csv, output_folder, device)

    except Exception as e:
        logger.error("Fatal error: %s", str(e), exc_info=True)
        raise

    finally:
        if chat is not None and hasattr(chat, "model"):
            try:
                chat.model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Successfully cleaned up resources")
            except Exception as e:
                logger.error("Error during cleanup: %s", str(e))


if __name__ == "__main__":
    main()
