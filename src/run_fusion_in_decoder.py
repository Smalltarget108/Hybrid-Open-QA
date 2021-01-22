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
import json

from utils_fusion_in_decoder import generate_dataloader

# transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    # get_cosine_schedule_with_warmup,
    # get_cosine_with_hard_restarts_schedule_with_warmup,
)

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


NUM_GPU = os.environ['SM_NUM_GPUS']
HOSTS = json.loads(os.environ['SM_HOSTS'])
CURRENT_HOST = os.environ['SM_CURRENT_HOST']

# setting up environment variable for pytorch lightning distributed training
os.environ['MASTER_ADDR'] = HOSTS[0]
os.environ['MASTER_PORT'] = str(6105)
os.environ['WORLD_SIZE'] = str(len(HOSTS))
os.environ['NODE_RANK'] = str(HOSTS.index(CURRENT_HOST))
os.environ['LOCAL_RANK'] = str(0)


class T5(pl.LightningModule):

    def __init__(self, cfg: DictConfig):
        super(T5, self).__init__()

        self.hparams = cfg
        self.learning_rate = cfg.optim.learning_rate

        self.config = T5Config.from_pretrained(
            cfg.model.config_name if cfg.model.config_name else cfg.model.model_name,
            cache_dir=cfg.model.cache_dir,
        )
        self.tokenizer = T5Tokenizer.from_pretrained(
            cfg.model.tokenizer_name if cfg.model.tokenizer_name else cfg.model.model_name,
            cache_dir=cfg.model.cache_dir,
            use_fast=cfg.model.use_fast,
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            cfg.model.model_name,
            from_tf=False,
            config=self.config,
            cache_dir=cfg.model.cache_dir,
        )

        # # add special tokens
        # additional_special_tokens_dict = {
        #     'additional_special_tokens': ['[[', ']]', '==>']
        # }
        # self.tokenizer.add_special_tokens(additional_special_tokens_dict)
        # self.model.resize_token_embeddings(len(self.tokenizer))

        # calculate total training steps
        # self.t_total = len(self.train_dataloader()) * cfg.trainer.max_epochs
        self.t_total = cfg.trainer.max_steps

        if cfg.optim.warmup_steps < 1.0:
            self.warmup_steps = int(cfg.optim.warmup_steps * self.t_total)
        else:
            self.warmup_steps = int(cfg.optim.warmup_steps)
            assert self.warmup_steps < self.t_total

        logger.info(f'Number of warmup steps: {self.warmup_steps}, total number of training steps: {self.t_total}')


    def forward(self, input_ids, attention_mask, labels):
        # input_ids: (batch_size, num_inputs, max_source_length)
        batch_size = input_ids.size(0)
        input_ids = input_ids.reshape(-1, self.hparams.data.max_source_length) # (b * num_inputs, max_source_length)
        attention_mask_tmp = attention_mask.reshape(-1, self.hparams.data.max_source_length) # (b * num_inputs, max_source_length)

        # encode the question + context
        encoder_outputs = self.model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask_tmp,
            )

        # concat all the encoder hidden states
        hidden_states = encoder_outputs[0]
        encoder_outputs = (hidden_states.reshape(batch_size, -1, self.config.d_model), *encoder_outputs[1:])
        attention_mask = attention_mask.reshape(batch_size, -1) # (b, num_inputs * max_source_length)

        # fusion-in decoder
        outputs = self.model(input_ids=None,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                labels=labels
            )

        return outputs


    @torch.no_grad()
    def _generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_specific_kwargs
    ) -> torch.LongTensor:

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 3, "Input prompt should be of shape (batch_size, num_inputs, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            )
            pad_token_id = eos_token_id

        # current position and vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
            self.config.is_encoder_decoder
            and hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id

            assert (
                decoder_start_token_id is not None
            ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
            assert hasattr(self.model, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.model.get_encoder), "{} should be a method".format(self.model.get_encoder)

            # get encoder and store encoder outputs
            encoder = self.model.get_encoder()

            input_ids_tmp = input_ids.reshape(-1, self.hparams.data.max_source_length) # (b * num_inputs, max_source_length)
            attention_mask_tmp = attention_mask.reshape(-1, self.hparams.data.max_source_length) # (b * num_inputs, max_source_length)

            encoder_outputs: tuple = encoder(input_ids_tmp, attention_mask=attention_mask_tmp)
            encoder_outputs = (encoder_outputs[0].reshape(batch_size, -1, self.config.d_model), *encoder_outputs[1:])

            attention_mask = attention_mask.reshape(batch_size, -1) # (b, num_inputs * max_source_length)
            input_ids = input_ids.reshape(batch_size, -1)

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
            ).type_as(input_ids)
            cur_len = 1

            assert (
                batch_size == encoder_outputs[0].shape[0]
            ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .type_as(input_ids)
            )
            # expand encoder_outputs
            encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

        else:
            encoder_outputs = None
            cur_len = input_ids.shape[-1]

        assert (
            cur_len < max_length
        ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

        if num_beams > 1:
            output = self.model._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )
        else:
            output = self.model._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )

        return output


    def generate(self, input_ids, attention_mask):

        outputs = self._generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.hparams.data.max_generation_length,
            do_sample=False,
            # top_p=0.95,
            num_beams=self.hparams.data.num_beams,
            num_return_sequences=self.hparams.data.num_return_sequences,
        )
        if outputs.size(1) < self.hparams.data.max_generation_length:
            outputs = torch.cat((outputs, torch.zeros(outputs.size(0), self.hparams.data.max_generation_length-outputs.size(1)).type_as(outputs)), dim=1)

        return outputs

    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, labels = batch

        # fwd
        outputs = self.forward(input_ids, attention_mask, labels)

        # loss
        loss = outputs[0]

        return {'loss': loss}

    def training_step_end(self, outputs):
        loss = outputs['loss'].mean()
        # logs
        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, labels = batch

        # fwd
        outputs = self.forward(input_ids, attention_mask, labels)
        generated = self.generate(input_ids, attention_mask)

        # loss
        loss = outputs[0]

        labels[labels==-100] = 0
        return {'val_loss': loss, 'generations': generated, 'labels': labels}

    def validation_step_end(self, outputs):
        loss  = outputs['val_loss'].mean()
        outputs['val_loss'] = loss

        return outputs

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        preds  = torch.cat([x['generations'] for x in outputs], dim=0).cpu().numpy().tolist()
        labels = torch.cat([x['labels'] for x in outputs], dim=0).cpu().numpy().tolist()
        for pred, label in zip(preds[:5:self.hparams.num_return_sequences], labels[:5]):
            pred_str = self.tokenizer.decode(pred)
            label_str = self.tokenizer.decode(label)
            logger.info(f"generated: {pred_str}")
            logger.info(f"tgt: {label_str}")

        tensorboard_logs = {'val_loss': avg_loss}

        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, labels = batch

        # fwd
        outputs = self.forward(input_ids, attention_mask, labels)
        generated = self.generate(input_ids, attention_mask)

        # loss
        loss = outputs[0]

        labels[labels==-100] = 0

        assert len(generated) == self.hparams.data.num_return_sequences * len(labels)

        return {'test_loss': loss, 'generations': generated, 'labels': labels}

    def test_step_end(self, outputs):
        loss  = outputs['test_loss'].mean()
        outputs['test_loss'] = loss

        return outputs

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        preds  = torch.cat([x['generations'] for x in outputs], dim=0).cpu().numpy().tolist()
        labels = torch.cat([x['labels'] for x in outputs], dim=0).cpu().numpy().tolist()

        assert len(preds) == self.hparams.data.num_return_sequences * len(labels)

        new_labels = []
        for label in labels:
            new_labels.extend([label] * self.hparams.data.num_return_sequences)
        labels = new_labels

        assert len(preds) == len(labels)

        results = []
        for pred, label in zip(preds, labels):
            # pdb.set_trace()
            pred_str = self.tokenizer.decode(pred)
            label_str = self.tokenizer.decode(label)
            results.append(
                {
                    "tgt": label_str,
                    "gen": pred_str,
                }
            )

        if not os.path.exists(self.hparams.data.output_dir):
            os.makedirs(self.hparams.data.output_dir)

        with jsonlines.open(os.path.join(self.hparams.data.output_dir, f"{self.hparams.model.model_name}_{self.hparams.data.max_source_length}_generated_sql.jsonl"), "w") as writer:
            writer.write_all(results)

        tensorboard_logs = {'test_loss': avg_loss}

        return {'avg_test_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.optim.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.hparams.optim.adam_epsilon)
        scheduler = {
             'scheduler': None,
             'monitor': 'val_loss', # Default: val_loss
             'interval': 'step', # step or epoch
             'frequency': 1
        }

        scheduler['scheduler'] = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.t_total
        )

        return [optimizer], [scheduler]


    def train_dataloader(self):
        train_dataloader = generate_dataloader(
            data_dir = self.hparams.data.data_dir,
            tokenizer = self.tokenizer,
            max_source_length = self.hparams.data.max_source_length,
            max_target_length = self.hparams.data.max_target_length,
            overwrite_cache = self.hparams.data.overwrite_cache,
            mode = "train",
            batch_size = self.hparams.optim.train_batch_size,
            question_type = self.hparams.data.question_type,
            passage_type = self.hparams.data.passage_type,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = generate_dataloader(
            data_dir = self.hparams.data.data_dir,
            tokenizer = self.tokenizer,
            max_source_length = self.hparams.data.max_source_length,
            max_target_length = self.hparams.data.max_target_length,
            overwrite_cache = self.hparams.data.overwrite_cache,
            mode = "dev",
            batch_size = self.hparams.optim.dev_batch_size,
            question_type = self.hparams.data.question_type,
            passage_type = self.hparams.data.passage_type,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = generate_dataloader(
            data_dir = self.hparams.data.data_dir,
            tokenizer = self.tokenizer,
            max_source_length = self.hparams.data.max_source_length,
            max_target_length = self.hparams.data.max_target_length,
            overwrite_cache = self.hparams.data.overwrite_cache,
            mode = "test",
            batch_size = self.hparams.optim.test_batch_size,
            question_type = self.hparams.data.question_type,
            passage_type = self.hparams.data.passage_type,
        )
        return test_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f'/opt/ml/code/model_arcifacts/fusion_config{args.config}.yaml')

    cfg.data.data_dir = os.environ['SM_CHANNEL_TRAIN']
    cfg.data.output_dir = os.path.join(os.environ['SM_MODEL_DIR'], 'output')
    os.makedirs(cfg.data.output_dir, exist_ok=True)
    cfg.model.checkpoint_dir = os.path.join(os.environ['SM_MODEL_DIR'], 'ckpt')
    os.makedirs(cfg.model.checkpoint_dir, exist_ok=True)

    # set seed
    seed_everything(cfg.optim.seed)

    # checkpoint
    checkpoint_dir = os.path.join(cfg.model.checkpoint_dir, cfg.model.model_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_callback = ModelCheckpoint(
        monitor='avg_val_loss',
        filepath=os.path.join(checkpoint_dir, '{epoch}-{val_loss:.4f}'),
        mode='min',
        save_last=True,
        save_top_k=3,
    )

    model_t5 = T5(cfg)
    if cfg.model.model_checkpoint:
        logger.info(f"Loading the checkpoint {cfg.model.model_checkpoint} and continue training")
        model_checkpoint = torch.load(cfg.model.model_checkpoint, map_location=lambda storage, loc: storage)
        model_dict = model_checkpoint['state_dict']
        model_t5.load_state_dict(model_dict)

    # training and testing
    if cfg.do_train:
        logger.info("Training starts")
        # tb_logger = loggers.WandbLogger(save_dir=cfg.optim.logging_dir, project='fusion in decoder')
        trainer = pl.Trainer(
            # logger=tb_logger,
            checkpoint_callback=checkpoint_callback,
            gpus=int(NUM_GPU),
            accelerator='ddp2',
            num_nodes=len(HOSTS),
            **OmegaConf.to_container(cfg.trainer, resolve=True),
        )
        trainer.fit(model_t5)
        # trainer.test(model_t5)

    if cfg.do_eval:
        logger.info("Evaluation starts")
        best_checkpoint_file = None
        if cfg.model.model_checkpoint == None:
            # find best checkpoint
            best_val_loss = 10000.
            for checkpoint_file in glob.glob(os.path.join(checkpoint_dir, "*val_loss*.ckpt")):
                try:
                    val_loss = float(checkpoint_file.split('=')[-1].replace(".ckpt", ""))
                except:
                    continue
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint_file = checkpoint_file
            logger.info(f"Loading the checkpoint: {best_checkpoint_file}")
        else:
            best_checkpoint_file = cfg.model.model_checkpoint

        # load model
        if best_checkpoint_file is not None:
            best_checkpoint = torch.load(best_checkpoint_file, map_location=lambda storage, loc: storage)
            model_t5.load_state_dict(best_checkpoint['state_dict'])

        # test using Trainer test function
        trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True))
        trainer.test(model_t5)


if __name__ == "__main__":
    main()
