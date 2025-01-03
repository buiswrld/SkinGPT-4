import argparse
import os
import random
import csv
import numpy as np
import torch
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
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="specify the gpu to load the model."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    logger.info("Parsed arguments: %s", args)
    return args


def setup_seeds(config):
    """
    Setup random seeds for reproducibility.

    Args:
        config: Configuration object containing seed settings
    """
    seed = config.run_cfg.seed + get_rank()
    logger.info("Setting random seed to: %d", seed)

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
    if not isinstance(image_path, (str, Path)):
        logger.error("Invalid image_path type: %s", type(image_path))
        return None

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
    csv_file: str, chat: Chat, output_csv: str, output_folder: str
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
                    "images", image_filename
                )  # Build the path to the file in the 'scin_images' folder

                if os.path.exists(image_path):
                    conv = CONV_VISION.copy()
                    print(f"Processing: {image_path}")

                    try:
                        processed_image = process_single_image(image_path)
                        if processed_image is None:
                            continue

                        img_list = []

                        chat.upload_img(processed_image, conv, img_list)
                        chat.ask(
                            "Could you describe the skin disease in this image for me?",
                            conv,
                        )

                        response, _ = chat.answer(
                            conv, img_list=img_list, max_new_tokens=300
                        )
                        results.append({"Image": image_path, "Description": response})

                        add_caption_to_image(image_path, response, output_folder)

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
                        csvfile, fieldnames=["Image", "Description"]
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


def check_accuracy(predictions_csv: str, ground_truth_csv: str) -> float:
    """
    Compare LLM predictions against ground truth labels.

    Args:
        predictions_csv: Path to CSV file containing LLM predictions
        ground_truth_csv: Path to CSV file containing correct labels

    Returns:
        float: Accuracy score between 0 and 1
    """
    try:
        if not os.path.exists(predictions_csv):
            logger.error("Predictions file not found: %s", predictions_csv)
            return 0.0

        if not os.path.exists(ground_truth_csv):
            logger.error("Ground truth file not found: %s", ground_truth_csv)
            return 0.0

        predictions = {}
        ground_truth = {}

        with open(predictions_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            predictions = {row["Image"]: row["Description"] for row in reader}

        with open(ground_truth_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ground_truth = {row["Image"]: row["Description"] for row in reader}

        if not predictions or not ground_truth:
            logger.warning("Empty predictions or ground truth data")
            return 0.0

        correct = sum(
            1
            for k in ground_truth
            if k in predictions and predictions[k] == ground_truth[k]
        )
        total = len(ground_truth)

        accuracy = correct / total if total > 0 else 0
        logger.info("Accuracy: %d/%d (%0.2f%%)", correct, total, accuracy * 100)

        return accuracy

    except Exception as e:
        logger.error("Error checking accuracy: %s", str(e))
        return 0.0


def main():
    """
    Main function to run the SkinGPT-4 demo application.
    """
    logger.info("Starting SkinGPT-4 demo application")
    chat = None
    device = None

    try:
        args = parse_args()

        if not os.path.exists(args.cfg_path):
            raise FileNotFoundError(f"Config file not found: {args.cfg_path}")

        cfg = Config(args)
        # setup_seeds(cfg)

        # Setup device
        if torch.cuda.is_available():
            device = f"cuda:{args.gpu_id}"
            torch.cuda.set_device(args.gpu_id)
        else:
            device = "cpu"
            logger.warning("CUDA not available, using CPU")

        logger.info("Initializing Chat on device: %s", device)

        # Initialize model
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(device)

        # Initialize processor
        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(
            vis_processor_cfg.name
        ).from_config(vis_processor_cfg)

        # Initialize chat
        chat = Chat(model, vis_processor, device=device)
        logger.info("Chat initialization finished")

        # Process images
        csv_file = "sampled_image.csv"
        output_csv = "output_results.csv"
        ground_truth_csv = "ground_truth.csv"
        output_folder = "output_images"

        process_images(csv_file, chat, output_csv, output_folder)

        if os.path.exists(ground_truth_csv):
            accuracy = check_accuracy(output_csv, ground_truth_csv)
            logger.info("Final accuracy: %0.2f%%", accuracy * 100)
        else:
            logger.warning("Ground truth file not found, skipping accuracy check")

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
