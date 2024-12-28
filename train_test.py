# FILE: class_test.py

import torch
from torchvision import transforms
from PIL import Image
from pytorch_lightning import Trainer
from skingpt4.models.skin_gpt4 import skingpt4
from skingpt4.classification.classification_task import ClassificationTask

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),  # Suppress the warning by setting antialias=True
    transforms.ToTensor(),
])

# Define the parameters
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

## base model, no training
print("base model:")
model = ClassificationTask(params)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

test_images = [
    "dataset/images/test_image.png",
    "dataset/images/test_image2.png",
]

for image_path in test_images:
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0) 
    image = image.to(device).half()
    with torch.no_grad():
        logits = model({"image": image})
        print(f"Logits for {image_path}: {logits}") 
        print(f"Logits shape for {image_path}: {logits.shape}") 
        probs = torch.softmax(torch.tensor(probs), dim=1).numpy()
        predicted_class = torch.argmax(probs, dim=1)
        print(f"Image: {image_path}, Predicted Class: {predicted_class}")

### display values after training to observe difference
print("after training")
model_2 = ClassificationTask(params)

trainer = Trainer(
    max_epochs = 5,
)

print("trainer fitting")
trainer.fit(model_2)

model_2.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_2 = model_2.to(device)

test_images = [
    "dataset/images/test_image.png",
    "dataset/images/test_image2.png",
]

for image_path in test_images:
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0) 
    image = image.to(device).half()
    with torch.no_grad():
        logits = model_2({"image": image})
        print(f"Logits for {image_path}: {logits}") 
        print(f"Logits shape for {image_path}: {logits.shape}") 
        probs = torch.softmax(torch.tensor(probs), dim=1).numpy()
        predicted_class = torch.argmax(probs, dim=1)
        print(f"Image: {image_path}, Predicted Class: {predicted_class}")
