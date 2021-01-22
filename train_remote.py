import sagemaker
import os
from sagemaker.pytorch import PyTorch
from pytz import timezone
from datetime import datetime
import shutil
import time


role = "arn:aws:iam::922834146316:role/service-role/AmazonSageMaker-ExecutionRole-20190523T192589"

run_book = [
    ('ott-qa', '_ottqa_question_both_input'),
]

for task, config_name in run_book:
    time.sleep(3)
    estimator = PyTorch(debugger_hook_config=False, # IMPORTANT: sagemaker debugger is CANCER!!!!!
                        entry_point='run_fusion_in_decoder.py',
                        source_dir='src/',
                        role=role,
                        instance_count=2,
                        instance_type='ml.p3dn.24xlarge',
                        image_uri='922834146316.dkr.ecr.us-east-1.amazonaws.com/hybrid-qa',
                        output_path='s3://henghui-vertical-intern-east1/hybridQA/sm_output/',
                        code_location='s3://henghui-vertical-intern-east1/hybridQA/sm_output/',
                        framework_version='1.6.0',
                        py_version='py3',
                        max_run=2 * 24 * 60 * 60,
                        volume_size=500,
                        hyperparameters={'config': config_name}
                        )

    input_dict = {
        'train': f's3://henghui-vertical-intern-east1/hybridQA/{task}',
    }

    print("start training...")
    localtime = datetime.now(timezone('America/Los_Angeles'))
    fmt = '%m%d%H%M'
    st = localtime.strftime(fmt)
    job_name = (st + config_name + '-' + task).replace('_', '-')
    estimator.fit(input_dict, job_name=job_name, wait=True)
