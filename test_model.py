import argparse
import torch
from PIL import Image
import numpy as np
from skingpt4.classification.classification_task import ClassificationTask
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Test Model")
    parser.add_argument("--gpu-id", type=int, default=0, help="Specify the GPU to load the model.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the model checkpoint file")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image file")
    return parser.parse_args()

def load_model(checkpoint_path, device):
    model = ClassificationTask.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, target_size=(810, 1080)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image, image_tensor

def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        _, predicted_class = torch.max(output, 1)
        predicted_class = predicted_class.item()
    return predicted_class, probabilities

def main():
    args = parse_args()
    device = f"cuda:{args.gpu_id}" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    
    model = load_model(args.ckpt_path, device) 
    image, image_tensor = preprocess_image(args.image_path)
    predicted_class, probabilities = predict(model, image_tensor, device)
    
    classes = ["Eczema", "Allergic Contact Dermatitis", "Urticaria", "Psoriasis", "Impetigo", "Tinea"]
    print(f"Predicted class: {classes[predicted_class]}")
    print(f"Probabilities: {probabilities}")
    for idx, prob in enumerate(probabilities):
        print(f"Class index {idx} ({classes[idx]}): {prob}")

if __name__ == "__main__":
    main()