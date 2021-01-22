ARG REGION=us-east-1

FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:1.6.0-gpu-py3

# # Apex
# WORKDIR /tmp/unique_for_apex
# # uninstall Apex if present, twice to make absolutely sure :)
# RUN pip uninstall -y apex || :
# RUN pip uninstall -y apex || :
# # SHA is something the user can touch to force recreation of this Docker layer,
# # and therefore force cloning of the latest version of Apex
# RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
# WORKDIR /tmp/unique_for_apex/apex
# RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
# WORKDIR /workspace

# Add the transformer package. This package is very important for NLP research.
RUN pip install transformers==3.0.2
RUN pip install pytorch-lightning==1.1.4
RUN pip install omegaconf==2.0.5
RUN pip install jsonlines
RUN pip install wandb
