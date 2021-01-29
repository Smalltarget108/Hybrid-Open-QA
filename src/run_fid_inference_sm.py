import glob
import os
import argparse
import numpy as np
import logging
import glob
import jsonlines
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import recall_score
from typing import Iterable, Optional, Tuple
import torch
import torch.nn.functional as F
# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers, seed_everything
import pdb

from utils_fusion_in_decoder import generate_dataloader
from run_fusion_in_decoder import T5

import tarfile

# transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--num_return_sequences', type=int, default=3)
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f'/opt/ml/code/model_arcifacts/fusion_config{args.config}.yaml')

    cfg.data.data_dir = os.environ['SM_CHANNEL_TRAIN'] # /opt/ml/input/data/train/

    checkpoint_dir = os.environ['SM_CHANNEL_TEST'] # /opt/ml/input/data/test/
    print(os.listdir(checkpoint_dir)) # [model.tar.gz]
    cfg.model.checkpoint_dir = os.path.join(os.environ['SM_MODEL_DIR'], 'ckpt')
    os.makedirs(cfg.model.checkpoint_dir, exist_ok=True)
    # decompress
    print("extracting files...\n")
    with tarfile.open(os.path.join(checkpoint_dir, 'model.tar.gz'), 'r') as f:
        f.extractall(cfg.model.checkpoint_dir)

    print('after decompression')
    print(os.listdir(cfg.model.checkpoint_dir)) # []

    cfg.model.checkpoint_dir = os.path.join(cfg.model.checkpoint_dir, 'ckpt', cfg.model.model_name)
    # e.g. /opt/ml/input/data/test/ckpt/t5-base/

    print(os.listdir(cfg.model.checkpoint_dir))

    cfg.data.output_dir = os.path.join(os.environ['SM_MODEL_DIR'], 'output')
    os.makedirs(cfg.data.output_dir, exist_ok=True)

    if args.test_batch_size is not None:
        cfg.optim.test_batch_size = args.test_batch_size

    best_val_loss = 10000.
    checkpoints = os.path.join(cfg.model.checkpoint_dir, "*val_loss*.ckpt")
    for checkpoint_file in glob.glob(checkpoints):
        print(checkpoint_file)
        try:
            val_loss = float(checkpoint_file.split('=')[-1].replace(".ckpt", ""))
        except:
            continue
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_file = checkpoint_file
    print(f"Loading the checkpoint: {best_checkpoint_file}")
    cfg.model.model_checkpoint = best_checkpoint_file

    # overwrite
    cfg.data.num_beams = args.num_beams
    cfg.data.num_return_sequences = args.num_return_sequences

    # set seed
    seed_everything(cfg.optim.seed)

    tokenizer = T5Tokenizer.from_pretrained(
        cfg.model.tokenizer_name if cfg.model.tokenizer_name else cfg.model.model_name,
        cache_dir=cfg.model.cache_dir,
        use_fast=cfg.model.use_fast,
    )

    model_t5 = T5(cfg, tokenizer)

    print("Evaluation starts")

    test_dataloader = generate_dataloader(
        data_dir = cfg.data.data_dir,
        tokenizer = tokenizer,
        max_source_length = cfg.data.max_source_length,
        max_target_length = cfg.data.max_target_length,
        overwrite_cache = cfg.data.overwrite_cache,
        mode = "test",
        batch_size = cfg.optim.test_batch_size,
        question_type = cfg.data.question_type,
        passage_type = cfg.data.passage_type,
    )

    torch.cuda.empty_cache()

    best_checkpoint_file = cfg.model.model_checkpoint

    # load model
    best_checkpoint = torch.load(best_checkpoint_file, map_location=lambda storage, loc: storage)
    model_t5.load_state_dict(best_checkpoint['state_dict'])

    # test using Trainer test function
    cfg.trainer.precision = 32
    trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True))
    trainer.test(model_t5, test_dataloader)


if __name__ == "__main__":
    main()
