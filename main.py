import os
import fire
from pytorch_lightning import Trainer

from skingpt4.models.skin_gpt4 import skingpt4

from util import init_exp_folder, Args
from skingpt4.classification import (get_task,
                       load_task,
                       get_ckpt_callback, 
                       get_early_stop_callback,
                       get_logger)

def train(
        
        # lightning params
        gpus=1,
        accelerator=None,
        logger_type='wandb', 
        save_dir="../archive/results",
        exp_name="demo", #TODO ~ Customize
        proj_name="skingpt", #TODO ~ Customize
        patience=10,
        gradient_clip_val=0.5,
        limit_train_batches=16.0, #TODO ~ Customize
        weights_summary=None,
        max_epochs=1,

        #util params
        task='classification',#TODO ~ Customize
        loss_fn="CE",

        ## model params
        
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        low_resource=False,
        device_8bit=0,

        #misc
        pretrained=True,
        num_classes=6, #TODO ~ Customize
        stochastic_weight_avg=True,
        tb_path="./tb",

        dataset_path="data/data.csv" #TODO ~ customize 
        ):
    
    """
    gpus=gpus,
                        accelerator=accelerator,
                        logger=get_logger(logger_type, save_dir, exp_name, proj_name),
                        callbacks=[get_early_stop_callback(patience),
                                    get_ckpt_callback(save_dir, exp_name)],
                        weights_save_path=os.path.join(save_dir, exp_name),
                        gradient_clip_val=gradient_clip_val,
                        limit_train_batches=limit_train_batches,
                        weights_summary=weights_summary,
                        max_epochs=max_epochs
    """

    """
    Run the training experiment.

    Args:
        save_dir: Path to save the checkpoints and logs
        exp_name: Name of the experiment
        model: Model name
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
        pretrained: Whether or not to use the pretrained model
        num_classes: Number of classes
        accelerator: Distributed computing mode
        logger_type: 'wandb' or 'test_tube'
        gradient_clip_val:  Clip value of gradient norm
        limit_train_batches: Proportion of training data to use
        max_epochs: Max number of epochs
        patience: number of epochs with no improvement after
                  which training will be stopped.
        stochastic_weight_avg: Whether to use stochastic weight averaging.
        tb_path: Path to global tb folder
        loss_fn: Loss function to use
        weights_summary: Prints a summary of the weights when training begins.

    Returns: None

    """
    args = Args(locals())
    init_exp_folder(args)

    #model instance
    task = get_task(args)

    trainer = Trainer(gpus=gpus,
                      accelerator=accelerator,
                      logger=get_logger(logger_type, save_dir, exp_name, proj_name),
                      callbacks=[get_early_stop_callback(patience),
                                 get_ckpt_callback(save_dir, exp_name)],
                      weights_save_path=os.path.join(save_dir, exp_name),
                      gradient_clip_val=gradient_clip_val,
                      limit_train_batches=limit_train_batches,
                      weights_summary=weights_summary,
                      max_epochs=max_epochs)
    trainer.fit(task)


def test(ckpt_path,
         gpus=4,
         **kwargs):
    """
    Run the testing experiment.

    Args:
        ckpt_path: Path for the experiment to load
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
    Returns: None

    """
    task = load_task(ckpt_path, **kwargs)
    trainer = Trainer(gpus=gpus)
    trainer.test(task)


if __name__ == "__main__":
    fire.Fire()

"""
import torch
from torchvision import transforms
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from skingpt4.models.skin_gpt4 import skingpt4
from skingpt4.classification.classification_task import ClassificationTask
import wandb  # Import the W&B library

wandb.init(
    project="skingpt4",    
    config={
    "epochs": 10,
    }) 

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

"""