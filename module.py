import argparse
import torch
import time
import pytorch_lightning as pl
import numpy as np
import math

from utils import (
    pickle_save,
    freeze_params,
    assert_all_frozen,
    save_json,
    lmap,
    flatten_list,
    NLLLoss,
    universal_sentence_embedding,
)
from torch.nn import functional as F
from torch import nn
from pathlib import Path
from transformers import (
    AutoConfig,
    PretrainedConfig,
    AutoTokenizer,
    PreTrainedTokenizer
)
from collections import defaultdict
from dataset import KnowledgeSeq2SeqDataset
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any
from utils import sequence_loss
from transformers.optimization import (
    AdamW,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.models.bart.modeling_bart import shift_tokens_right
from metrics import (
    f1_metric
)
from prefixTuning import PrefixTuningBart, PrefixTuningGPT2
from prefixBart import PrefixBartForConditionalGeneration
from prefixGPT2 import PrefixGPT2ForConditionalGeneration
from callback import logging

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    
    
}


class BaseTransformer(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        num_labels=None,
        mode="base",
        config=None,
        tokenizer=None,
        model=None,
        **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()

        self.save_hyperparameters(hparams)  
        self.step_count = 0  
        self.output_dir = Path(self.hparams.output_dir)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        if config is None:   
            self.config = AutoConfig.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                **({"num_labels": num_labels} if num_labels is not None else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )
        else:
            self.config: PretrainedConfig = config
        
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer: PreTrainedTokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage=None):
        if stage == "fit":
            self.dataset_size = len(self.train_dataloader().dataset)
        elif stage == "validate":
            self.dataset_size = len(self.val_dataloader().dataset)
        elif stage == "test":
            self.dataset_size = len(self.test_dataloader().dataset)

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps()
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
        )

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_end(outputs)

    def total_steps(self) -> int:
        num_devices = max(1, self.hparams.gpus)  
        effective_batch_size = self.hparams.train_batch_size * self.hparams.gradient_accumulation_steps * num_devices
        return (self.dataset_size / effective_batch_size) * self.hparams.num_train_epochs

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False):
        raise NotImplementedError("You must implement this for your task")

    def train_dataloader(self):
        return self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:   

        save_path = self.output_dir.joinpath("best_tfmr")
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


class PrefixDialogModule(BaseTransformer):

    mode = "question-answering"
    loss_names = ["loss"]
    metric_names = ["loss"]
    default_val_metric = "loss"

    def __init__(self, hparams, **kwargs):

        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)

        self.eval_type = hparams.eval_type  
        self.dirpath = self.hparams.output_dir
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.preds_save_path = Path(self.output_dir) / "preds.txt"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)  
        self.step_count = 0
        self.metrics = defaultdict(list)
        self.preds: List = []
        self.tuning_mode = self.hparams.tuning_mode
        self.vocab_size = len(self.tokenizer)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        self.bow_loss_fn = NLLLoss(reduction="mean", ignore_index=self.tokenizer.pad_token_id)
        if self.tuning_mode == "pt2":
            self.loss_names = ['loss', 'ce_loss']

        self.model_type = None
        if "bart" in self.hparams.model_name_or_path:
            self.model_type = "bart"
            PrefixTuning = PrefixTuningBart
        elif "gpt" in self.hparams.model_name_or_path:
            self.model_type = "gpt"
            PrefixTuning = PrefixTuningGPT2
        else:
            assert False

        if self.model_type == "bart":
            self.model = PrefixBartForConditionalGeneration.from_pretrained(
                self.hparams.model_name_or_path,
                config=self.config,
                cache_dir=cache_dir,
            )
        elif self.model_type == "gpt":
            self.model = PrefixGPT2ForConditionalGeneration.from_pretrained(
                self.hparams.model_name_or_path,
                config=self.config,
                cache_dir=cache_dir,
            )

        freeze_params(self.model)
        assert_all_frozen(self.model)

        self.prefix_model = None
        self.prefix_model_2 = None

        if self.tuning_mode == "pt1":

            assert self.hparams.pfxKlgModel_name_or_path is None
            self.prefix_model = PrefixTuning(
                self.config, hparams=hparams,
            )

        elif self.tuning_mode == "pt2":

            assert self.hparams.pfxKlgModel_name_or_path is not None, "Requiring model pretrained on first stage..."
            logging.info('---  loading from {}  ---'.format(hparams.pfxKlgModel_name_or_path))

            self.prefix_model = PrefixTuning.from_pretrained(
                self.hparams.pfxKlgModel_name_or_path,
                cache_dir=cache_dir,
                config=self.config,
                hparams=self.hparams,
            )

            freeze_params(self.prefix_model)
            assert_all_frozen(self.prefix_model)

            self.prefix_model_2 = PrefixTuning(self.config, hparams=hparams, qkv_trans=True)  

        else:

            assert False

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            max_knowledge_length=self.hparams.max_knowledge_length,
            prefix=self.model.config.prefix or "",
        )
        if not self.eval_type:
            n_observations_per_split = {
                "train": self.hparams.n_train,
                f"valid": self.hparams.n_val,
                f"test": self.hparams.n_test,
            }
        else:  
            n_observations_per_split = {
                "train": self.hparams.n_train,
                f"valid_{self.eval_type}": self.hparams.n_val,
                f"test_{self.eval_type}": self.hparams.n_test,
            }

        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        if not self.eval_type:
            self.target_lens = {
                "train": self.hparams.max_target_length,
                f"valid": self.hparams.max_target_length,
                f"test": self.hparams.max_target_length,
            }
        else:

            self.target_lens = {
                "train": self.hparams.max_target_length,
                f"valid_{self.eval_type}": self.hparams.max_target_length,
                f"test_{self.eval_type}": self.hparams.max_target_length
            }

        self.num_workers = hparams.num_workers

        self.dataset_class = KnowledgeSeq2SeqDataset

        self.already_saved_batch = False

        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams

        self.eval_max_length = self.hparams.eval_max_gen_length
        self.eval_min_length = self.hparams.eval_min_gen_length

        self.val_metric = self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric

    def get_dataset(self, type_path) -> KnowledgeSeq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            preseqlen=self.hparams.preseqlen,
            tuning_mode=self.tuning_mode,
            
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:

        dataset = self.get_dataset(type_path)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            sampler=None,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:

        if not self.eval_type:
            return self.get_dataloader(f"valid", batch_size=self.hparams.eval_batch_size)
        return self.get_dataloader(f"valid_{self.eval_type}", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        if not self.eval_type:
            return self.get_dataloader(f"test", batch_size=self.hparams.eval_batch_size)
        return self.get_dataloader(f"test_{self.eval_type}", batch_size=self.hparams.eval_batch_size)

    def save_readable_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        readable_batch = {
            k: self.tokenizer.batch_decode(v.tolist()) if "mask" not in k else v.shape for k, v in batch.items()
        }
        save_json(readable_batch, Path(self.output_dir) / "text_batch.json")
        save_json({k: v.tolist() for k, v in batch.items()}, Path(self.output_dir) / "tok_batch.json")

        self.already_saved_batch = True
        return readable_batch

    def forward(self, input_ids, **kwargs):

        if self.tuning_mode == "pt1":

            return self.prefix_model(input_ids, self.model, **kwargs)

        elif self.tuning_mode == "pt2":

            klg_prompt_dict = self.prefix_model.get_prompt(bsz=input_ids.size(0), return_dict=True)

            emb_weight = self.model.transformer.wte.weight if isinstance(self.model, PrefixGPT2ForConditionalGeneration) \
                else self.model.model.shared.weight

            assert emb_weight.requires_grad is False

            prefix_past_kv_list, prefix_key_padding_mask, merge_attn_output_bow_logits = \
                self.prefix_model_2.get_prompt_2(input_ids.size(0), klg_prompt_dict, emb_weight)

            return self.prefix_model_2(
                input_ids=input_ids,
                model=self.model,
                _prefix_past_kv_list=prefix_past_kv_list,
                _prefix_key_padding_mask=prefix_key_padding_mask,
                merge_attn_output_bow_logits=merge_attn_output_bow_logits,
                **kwargs
            )

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.pad
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]

        if isinstance(self.model, PrefixBartForConditionalGeneration):
            decoder_input_ids = shift_tokens_right(
                tgt_ids, pad_token_id,
                decoder_start_token_id=self.model.config.decoder_start_token_id
            )
        else:  

            decoder_input_ids = shift_tokens_right(
                tgt_ids, pad_token_id,
                decoder_start_token_id=self.tokenizer.bos_token_id
            )
            decoder_input_mask = decoder_input_ids.ne(self.pad).type_as(src_mask)
            decoder_input_mask[:, 0] = 1  
            decoder_input_ids = torch.cat([src_ids, decoder_input_ids], dim=-1)
            src_mask = torch.cat([src_mask, decoder_input_mask], dim=-1)
            tgt_src_ids = torch.full(src_ids.size(), fill_value=self.pad if not self.training else -100, requires_grad=False).to(src_ids.device)
            tgt_ids = torch.cat([tgt_src_ids, tgt_ids], dim=-1).to(src_ids.device)
            target_mask = tgt_ids != (self.pad if not self.training else -100)

        if not self.already_saved_batch:  
            batch["decoder_input_ids"] = decoder_input_ids
            self.save_readable_batch(batch)

        if isinstance(self.model, PrefixBartForConditionalGeneration):
            outputs = self(
                input_ids=src_ids,
                attention_mask=src_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
            )
        else:
            outputs = self(
                input_ids=decoder_input_ids,
                attention_mask=src_mask,
                use_cache=False,
            )

        lm_logits = outputs["logits"]

        if isinstance(self.model, PrefixBartForConditionalGeneration):
            ce_loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)
            assert lm_logits.shape[-1] == self.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        else:  
            ce = lambda logit, target: F.cross_entropy(logit, target, reduce=False)
            loss_fn = lambda logits, targets: sequence_loss(logits, targets, ce, mask=target_mask)  
            loss = loss_fn(lm_logits, tgt_ids).mean()

        
        if self.tuning_mode == "pt2":
            bow_labels = batch['bow_labels']  
            merge_attn_output_bow_logits = outputs['merge_attn_output_bow_logits'].log()  
            bsz, pre_seq_len = merge_attn_output_bow_logits.size()[:2]
            prefix_mask = torch.ones([bsz, pre_seq_len], requires_grad=False).type_as(bow_labels).to(bow_labels.device)
            merge_attn_output_bow_logits = universal_sentence_embedding(merge_attn_output_bow_logits, prefix_mask)  
            merge_attn_output_bow_logits = merge_attn_output_bow_logits.unsqueeze(1).repeat(1, bow_labels.size(-1), 1)
            bow_loss = 0.1 * self.bow_loss_fn(input=merge_attn_output_bow_logits, target=bow_labels)

            ce_loss = loss.detach()
            self.log("ce_loss", ce_loss, prog_bar=True)
            self.log('bow_loss', bow_loss, prog_bar=True)

            loss = loss + bow_loss

            return (loss, ce_loss)

        return (loss, )

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:

        loss_tensors = self._step(batch)
        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}

        
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
        logs["bs"] = batch["input_ids"].shape[0]
        logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()

        return {"loss": loss_tensors[0], "log": logs}

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        gen_text = [" ".join(i.split()) for i in gen_text]
        return lmap(str.strip, gen_text)

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()
        bsz = batch["input_ids"].shape[0]

        loss_tensors = self._step(batch)

        if self.tuning_mode == "pt1":
            prefix_past_kv_list, prefix_key_padding_mask = self.prefix_model.get_prompt(bsz=bsz)
        else:
            klg_prompt_dict = self.prefix_model.get_prompt(bsz=bsz, return_dict=True)
            emb_weight = self.model.transformer.wte.weight if isinstance(self.model, PrefixGPT2ForConditionalGeneration) else self.model.model.shared.weight
            assert emb_weight.requires_grad is False
            prefix_past_kv_list, prefix_key_padding_mask, _ = self.prefix_model_2.get_prompt_2(bsz, klg_prompt_dict, emb_weight)

        if isinstance(self.model, PrefixBartForConditionalGeneration):

            generated_ids = self.model.generate(
                batch["input_ids"],
                pref_past_kv_list=prefix_past_kv_list,
                pref_key_padding_mask=prefix_key_padding_mask,
                attention_mask=batch["attention_mask"],
                use_cache=True,   
                decoder_start_token_id=self.model.config.decoder_start_token_id,
                num_beams=self.eval_beams,
                max_length=self.eval_max_length,
                min_length=self.eval_min_length,
                no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
                pad_token_id=self.pad
            )

        else:  

            batch_size = batch["input_ids"].size(0)
            input_ids = torch.full([batch_size, 1], fill_value=self.config.bos_token_id).to(batch["input_ids"].device)
            input_ids = torch.cat([batch["input_ids"], input_ids], dim=-1)

            attention_mask = torch.ones([batch_size, 1]).to(input_ids.device)  
            attention_mask = torch.cat([batch["attention_mask"], attention_mask], dim=-1)

            generated_ids = self.model.generate(
                input_ids,
                pref_past_kv_list=prefix_past_kv_list,
                pref_key_padding_mask=prefix_key_padding_mask,
                attention_mask=attention_mask,
                use_cache=True,  
                num_beams=self.eval_beams,
                max_length=batch["input_ids"].size(-1) + self.eval_max_length,
                min_length=batch["input_ids"].size(-1) + self.eval_min_length,
                no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
                pad_token_id=self.pad,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            generated_ids = generated_ids[:, batch["input_ids"].size(-1):]

        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["labels"])
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target)
        return base_metrics

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:

        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"] if 'ce_loss' not in losses else losses["ce_loss"]
        generative_metrics = {
            k: np.array([x[k].detach().cpu() if type(x[k]) is torch.Tensor else x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metric_val = (
            generative_metrics[self.val_metric] if self.val_metric in generative_metrics else losses[self.val_metric]
        )
        metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
        generative_metrics.update({k: v.item() for k, v in losses.items()})
        losses.update(generative_metrics)

        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count

        preds = flatten_list([x["preds"] for x in outputs])
        target = flatten_list([x["target"] for x in outputs])

        f1 = f1_metric(preds, target)
        ppl = math.exp(loss)

        self.log('f1', f1)
        self.log('loss', loss)

        all_metrics["F1"] = f"{f1 * 100.:.2f}"
        all_metrics["PPL"] = f"{ppl:.4f}"

        self.metrics[prefix].append(all_metrics)  

        self.preds = [" ||| ".join(_pair) for _pair in zip(preds, target)]  

        return {
            "log": all_metrics,
            "preds": self.preds,
            
            f"{prefix}_{self.val_metric}": metric_tensor,
        }

    def test_epoch_end(self, outputs):
        output_dict = self.validation_epoch_end(outputs, prefix="test")
        self.preds = output_dict["preds"]

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        if self.tuning_mode == "pt1":
            model = self.prefix_model
        if self.tuning_mode == "pt2":
            model = self.prefix_model_2

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.opt = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
        )
        scheduler = self.get_lr_scheduler()

        return [self.opt], [scheduler]

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.tuning_mode == "pt1":
            save_path = self.output_dir.joinpath("best_pt1")
            self.prefix_model.config.save_step = self.step_count
            self.prefix_model.save_pretrained(save_path)
        elif self.tuning_mode == "pt2":
            save_path = self.output_dir.joinpath("best_pt2")
            self.prefix_model_2.config.save_step = self.step_count
            self.prefix_model_2.save_pretrained(save_path)

        self.tokenizer.save_pretrained(save_path)
