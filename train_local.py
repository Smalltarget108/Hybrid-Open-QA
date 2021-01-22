import sagemaker
import os
from sagemaker.pytorch import PyTorch
from pytz import timezone
from datetime import datetime
import shutil
import time

role = "arn:aws:iam::630054792439:role/service-role/AmazonSageMaker-ExecutionRole-20200927T153620"


estimator = PyTorch(debugger_hook_config=False, # IMPORTANT: sagemaker debugger is CANCER!!!!!
                    entry_point='run_fusion_in_decoder.py',
                    source_dir='/home/ec2-user/projects/T5_experiments/src/',
                    role=role,
                    train_instance_count=1,
                    train_instance_type='local_gpu',
                    image_name='630054792439.dkr.ecr.us-east-1.amazonaws.com/pytorch-1.6.0-py3',
                    output_path='file:///home/ec2-user/efs/sagemaker-output/hybridQA',
                    framework_version='1.6.0',
                    hyperparameters={'config': '_mixed_question_textual_input'}
                    )

input_dict = {
    'train': 'file:///home/ec2-user/efs/hybridQA',
}

estimator.fit(input_dict, wait=True)
