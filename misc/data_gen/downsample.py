import cv2
import os
import argparse

def downsample(image_path: str, output_path: str, degree: int):
    img = cv2.imread(image_path)
    orig_size = (img.shape[1], img.shape[0])

    for step in range(degree):
        img = cv2.pyrDown(img)
    
    img = cv2.resize(img, orig_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path, img)


def process_images(input_dir: str, output_dir: str, degree: int):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            downsample(input_path, output_path, degree)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(descrption='Downsample images to a specified degree')
    parser.add_argument('--degree', type=int, default=3, help='Degree of downsampling (call pyrdown n degrees times)')
    args = parser.parse_args()
    degree = args.degree
    input_dir = "images"
    output_dir = f"images/downsampled_d{degree}"
    process_images(input_dir, output_dir, degree)