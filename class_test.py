import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from skingpt4.models.skin_gpt4 import skingpt4
from skingpt4.classification.classification_task import ClassificationTask

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])

params = {
    "vit_model": "eva_clip_g",
    "q_former_model": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
    "img_size": 224,
    "drop_path_rate": 0,
    "use_grad_checkpoint": False,
    "vit_precision": "fp16",
    "freeze_vit": True,
    "freeze_qformer": True,
    "num_query_token": 32,
    "low_resource": False,
    "device_8bit": 0,
    "loss_fn": "CE",
    "dataset_path": "dataset/csv/test_csv.csv",
}
model = ClassificationTask(params)
model.eval()

test_images = [
    "dataset/images/test_image.png",
    "dataset/images/test_image2.png",
]

for image_path in test_images:
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0) 
    image = image.half()
    with torch.no_grad():
        logits = model({"image": image})
        predicted_class = torch.argmax(logits, dim=1).item()
        print(f"Image: {image_path}, Predicted Class: {predicted_class}")