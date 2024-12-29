import torch
from torchvision import transforms
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from skingpt4.models.skin_gpt4 import skingpt4
from skingpt4.classification.classification_task import ClassificationTask
import wandb  # Import the W&B library

wandb.init(project="skingpt4", entity="prodbui") 

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
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

model = ClassificationTask(params)

trainer = Trainer(
    max_epochs=5,
    logger=wandb.log,
)

print("Trainer fitting")
trainer.fit(model)

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
        
        wandb.log({
            "image": wandb.Image(image_path),
            "logits": logits.cpu().numpy(),
        })
        
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1)
        print(f"Image: {image_path}, Predicted Class: {predicted_class}")

        wandb.log({
            "predicted_class": predicted_class.item(),
            "image_path": image_path,
        })

wandb.finish()
