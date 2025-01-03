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
        accelerator='cuda',
        logger_type='wandb', 
        save_dir="../archive/results",
        exp_name="demo", #TODO ~ Customize
        proj_name="skingpt",
        patience=20,
        gradient_clip_val=0.5,
        limit_train_batches=16.0, #TODO ~ Customize
        weights_summary=None,
        max_epochs=200,

        #util params
        task='classification',
        loss_fn="CE",
        use_mlp_head = False,
        learning_rate = 5e-4,
        classes=('Eczema', 'Allergic Contact Dermatitis', 'Urticaria', 'Psoriasis', 'Impetigo', 'Tinea')#TODO ~ Customize
        num_classes=6, #TODO ~ Customize
        oversample=False,

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

    trainer = Trainer(devices=gpus,
                      accelerator=accelerator,
                      logger=get_logger(logger_type, save_dir, exp_name, proj_name),
                      callbacks=[get_early_stop_callback(patience),
                                 get_ckpt_callback(save_dir, exp_name)],
                      default_root_dir=os.path.join(save_dir, exp_name),
                      gradient_clip_val=gradient_clip_val,
                      limit_train_batches=limit_train_batches,
                      enable_model_summary=weights_summary,
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
    trainer = Trainer(devices=gpus)
    trainer.test(task)


if __name__ == "__main__":
    fire.Fire()