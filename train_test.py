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

model = ClassificationTask(params)

trainer = Trainer(
    gpus=1 if torch.cuda.is_available() else 0,
    max_epochs=10,
    progress_bar_refresh_rate=20,
)

trainer.fit(model)