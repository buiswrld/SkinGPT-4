import argparse
from util import Args
from skingpt4.models import load_model
from skingpt4.models import skingpt4

def get_model(model_args: dict):
    model_args_ = model_args

    if isinstance(model_args, argparse.Namespace):
        model_args_ = Args(vars(model_args))

    if model_args_.get('task') == "classification":
        model = skingpt4.from_configs(model_args)
 
    return model 