import argparse
import os
import random
import csv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from skingpt4.common.config import Config
from skingpt4.common.dist_utils import get_rank
from skingpt4.common.registry import registry
from skingpt4.conversation.conversation import Chat, CONV_VISION
from pathlib import Path
from typing import Union, Tuple, Optional
from tqdm import tqdm

# imports modules for registration
from skingpt4.datasets.builders import *
from skingpt4.models import *
from skingpt4.processors import *
from skingpt4.runners import *
from skingpt4.tasks import *


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('skingpt4_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments for the SkinGPT-4 demo.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - cfg-path: Path to configuration file
            - gpu-id: GPU device ID to use
            - options: Additional configuration overrides
    """
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True,
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    """
    Setup random seeds for reproducibility across all libraries.

    Args:
        config: Configuration object containing run_cfg.seed setting

    Returns:
        None

    Note:
        Sets seeds for random, numpy, and PyTorch libraries
        Disables CUDNN benchmarking for deterministic behavior
    """
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


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

    Note:
        Performs the following operations:
        1. Opens and converts image to RGB
        2. Resizes while maintaining aspect ratio
        3. Pads to target size with black borders
    """

    try:
        logger.info("Processing image: %s", image_path)

        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
                logger.debug("Converted image mode to RGB")

            orig_width, orig_height = img.size
            ratio = min(target_size[0] / orig_width,
                        target_size[1] / orig_height)
            new_size = (int(orig_width * ratio), int(orig_height * ratio))

            logger.debug("Resizing from %s to %s", img.size, new_size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            new_img = Image.new("RGB", target_size, (0, 0, 0))

            paste_x = (target_size[0] - new_size[0]) // 2
            paste_y = (target_size[1] - new_size[1]) // 2
            new_img.paste(img, (paste_x, paste_y))

            logger.info("Successfully processed image to size %s", target_size)
            return new_img

    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Error processing image %s: %s", image_path, str(e))
        return None


def process_images(image_folder: str, chat: Chat, output_csv: str) -> None:
    """
    Process all images in a folder and save results to a CSV.

    Args:
        image_folder: Directory containing source images
        chat: Chat instance for LLM processing
        output_csv: Path to output CSV file

    Returns:
        None

    Note:
        Processes each supported image file (.png, .jpg, .jpeg, .bmp)
        Saves results in CSV format with columns: Image, Description
        Logs progress and errors during processing
    """
    logger.info("Starting batch processing of images from %s", image_folder)
    conv = CONV_VISION.copy()
    img_list = []
    results = []

    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
            image_path = os.path.join(image_folder, image_file)
            logger.info("Processing: %s", image_path)

            try:
                # Process the image before sending to LLM
                processed_image = process_single_image(image_path)
                if processed_image is None:
                    logger.error(
                        "Skipping %s due to processing failure", image_file)
                    continue

                # Upload the processed image and ask the question
                _ = chat.upload_img(processed_image, conv, img_list)
                chat.ask("Describe this condition", conv)

                # Get the model's answer
                response, _ = chat.answer(
                    conv, img_list=[], max_new_tokens=300)
                logger.info(
                    "Successfully processed and analyzed %s", image_file)

                # Store the result
                results.append({"Image": image_file, "Description": response})

            except Exception as e:
                logger.error("Error processing %s: %s", image_file, str(e))
                continue

    # Write results to CSV
    try:
        with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["Image", "Description"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info("Results successfully saved to %s", output_csv)
    except Exception as e:
        logger.error("Error saving results to CSV: %s", str(e))


def check_accuracy(predictions_csv: str, ground_truth_csv: str) -> float:
    """
    Compare LLM predictions against ground truth labels.

    Args:
        predictions_csv: Path to CSV file containing LLM predictions
        ground_truth_csv: Path to CSV file containing correct labels

    Returns:
        float: Accuracy score between 0 and 1

    Note:
        Expects CSV files with 'Image' and 'Description' columns
        Calculates exact match accuracy between predictions and ground truth
        Returns 0.0 if processing fails
    """
    try:
        predictions = {}
        ground_truth = {}

        with open(predictions_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                predictions[row['Image']] = row['Description']

        with open(ground_truth_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ground_truth[row['Image']] = row['Description']

        correct = 0
        total = 0

        for image_name, true_label in ground_truth.items():
            if image_name in predictions:
                if predictions[image_name] == true_label:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0
        logger.info("Accuracy: %d/%d (%0.2f%%)",
                    correct, total, accuracy * 100)

        return accuracy
    except Exception as e:
        logger.error("Error checking accuracy: %s", str(e))
        return 0.0


def main():
    """
    Main function to run the SkinGPT-4 demo application.

    Args:
        None

    Returns:
        None

    Flow:
        1. Parse command line arguments
        2. Initialize configuration and seeds
        3. Setup model and chat interface
        4. Process images from specified folder
        5. Calculate and report accuracy
        6. Handles errors with appropriate logging
    """
    logger.info("Starting SkinGPT-4 demo application")

    args = parse_args()
    cfg = Config(args)
    setup_seeds(cfg)

    try:
        logger.info('Initializing Chat')
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(
            'cuda:{}'.format(args.gpu_id))

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(
            vis_processor_cfg.name).from_config(vis_processor_cfg)
        chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
        logger.info('Chat initialization finished')

        # Process images
        image_folder = "images"
        output_csv = "output_results.csv"
        ground_truth_csv = "ground_truth.csv"
        process_images(image_folder, chat, output_csv)
        logger.info('Image processing completed')

        accuracy = check_accuracy(output_csv, ground_truth_csv)
        logger.info("Final accuracy: %0.2f%%", accuracy * 100)

    except Exception as e:
        logger.error("Fatal error: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    main()
