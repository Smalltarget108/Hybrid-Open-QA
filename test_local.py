import sagemaker
import os
from sagemaker.pytorch import PyTorch
from pytz import timezone
from datetime import datetime
import shutil
import time


role = "arn:aws:iam::630054792439:role/service-role/AmazonSageMaker-ExecutionRole-20200927T153620"

run_book = [
    # ('ott-qa', '_ottqa_question_both_input'),
    # ('regular', '_wikisql_question_both_input_no_sql_supervision'),
    # ('regular', '_squad_question_both_input'),
    # ('open-nq', '_nq_question_both_input'),
    # ('nq-wikisql', '_nq_wikisql_question_both_input'),
    # ('ottqa-wikisql', '_ottqa_wikisql_question_both_input'),
    # ('all', '_all_question_both_input'),
    # ('all', '_all_question_both_input_no_sql_supervision'),
    # ('ottqa_wikisql', '_ottqa_wikisql_question_both_input_no_sql_supervision'),
    ('nq_wikisql', '_nq_wikisql_both_input_no_sql_supervision'),
    # ('nq', '_nq_question_text_input'),
    # ('ott-qa', '_ottqa_question_text_input'),
    # ('wikisql', '_wikisql_question_text_input'),
    # ('', '_opensquad_question_table_input_no_sql'),
    # ('', '_nq_question_table_input_no_sql'),
    # ('', '_ottqa_question_table_input_no_sql'),
    # ('', '_wikisql_question_table_input_no_sql'),
    # ('', '_mix_squawiki_question_table_input_no_sql'),
]

# nq_experiments = [
#     ('', '_nq_question_text_input'),
#     ('', '_nq_question_both_input'),
#     ('', '_nq_question_table_input_no_sql'),
# ]

bushiyan = [
    ('', '_squad_question_table_input'),
    # ('', '_wikisql_question_text_input'),
]

table_with_sql = [
    ('01262254-squad-wikisql-question-table-input-sql', '_squad_wikisql_question_table_input_sql'),
    # ('', '_nq_wikisql_question_table_input_sql'),
    # ('', '_ott_wikisql_question_table_input_sql'),
]

# both_with_sql = [
#     ('', '_nq_wikisql_question_both_input_sql'),
#     ('', '_ottqa_wikisql_question_both_input_sql'),
# ]

for ckpt_path, config_name in table_with_sql:
    time.sleep(3)
    print(config_name)
    estimator = PyTorch(debugger_hook_config=False, # IMPORTANT: sagemaker debugger is CANCER!!!!!
                        entry_point='run_fid_inference_sm.py',
                        source_dir='/home/ec2-user/projects/Hybrid-Open-QA/src/',
                        role=role,
                        train_instance_count=1,
                        train_instance_type='local_gpu',
                        image_uri='630054792439.dkr.ecr.us-east-1.amazonaws.com/pytorch-1.6.0-cuda11',
                        # image_uri='630054792439.dkr.ecr.us-east-1.amazonaws.com/pytorch-1.7.1', # cuda 11.0
                        output_path='file:///home/ec2-user/efs/sagemaker-output/hybridQA',
                        # framework_version='1.6.0',
                        # py_version='py3',
                        hyperparameters={
                            'config': config_name,
                            'num_beams': 3,
                            'num_return_sequences': 3,
                            'test_batch_size': 32,
                            }
                        )

    input_dict = {
        'train': 'file:///home/ec2-user/efs/hybridQA/FID-inference',
        'test': 'file:///home/ec2-user/efs/mix-squa-wiki/sm_output/' + ckpt_path + '/output',
    }

    print("start testing...")
    localtime = datetime.now(timezone('America/Los_Angeles'))
    fmt = '%m%d%H%M%S'
    st = localtime.strftime(fmt)
    job_name = (st + config_name).replace('_', '-')
    estimator.fit(input_dict, job_name=job_name, wait=True)
