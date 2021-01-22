import sagemaker
import os
from sagemaker.pytorch import PyTorch
from pytz import timezone
from datetime import datetime
import shutil
import time


role = "arn:aws:iam::630054792439:role/service-role/AmazonSageMaker-ExecutionRole-20200927T153620"

run_book = [
    ('ott-qa', '_ottqa_question_both_input'),
]

for task, config_name in run_book:
    time.sleep(3)
    estimator = PyTorch(debugger_hook_config=False, # IMPORTANT: sagemaker debugger is CANCER!!!!!
                        entry_point='run_fusion_in_decoder.py',
                        source_dir='/home/ec2-user/projects/Hybrid-Open-QA/src/',
                        role=role,
                        train_instance_count=1,
                        train_instance_type='ml.p3dn.24xlarge',
                        # image_name='630054792439.dkr.ecr.us-east-1.amazonaws.com/t5-training',
                        # image_name='630054792439.dkr.ecr.us-east-1.amazonaws.com/fid-py3',
                        # image_name='630054792439.dkr.ecr.us-east-1.amazonaws.com/pytorch-1.6.0-py3',
                        image_uri='630054792439.dkr.ecr.us-east-1.amazonaws.com/pytorch-1.6.0',
                        output_path='s3://hanboli-research/hybridQA/sm_output/',
                        code_location='s3://hanboli-research/hybridQA/sm_output/',
                        # framework_version='1.6.0',
                        # py_version='py3',
                        train_max_run=2 * 24 * 60 * 60,
                        train_volume_size=500,
                        hyperparameters={'config': config_name}
                        )

    input_dict = {
        'train': f's3://hanboli-research/hybridQA/{task}',
    }

    print("start training...")
    localtime = datetime.now(timezone('America/Los_Angeles'))
    fmt = '%m%d%H%M'
    st = localtime.strftime(fmt)
    job_name = (st + config_name + '-' + task).replace('_', '-')
    estimator.fit(input_dict, job_name=job_name, wait=True)
