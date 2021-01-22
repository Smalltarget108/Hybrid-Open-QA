ARG REGION=us-east-1

# FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:1.6.0-gpu-py3
FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:1.6.0-gpu-py36-cu110-ubuntu18.04

# Add the dependencies.
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install transformers==3.0.2
RUN pip install pytorch-lightning==1.1.4
RUN pip install omegaconf==2.0.5
RUN pip install jsonlines
RUN pip install wandb
