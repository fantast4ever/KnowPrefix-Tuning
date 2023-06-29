import argparse
import os
import logging
from pathlib import Path
import pytorch_lightning as pl

from utils import pickle_save, check_output_dir
from transformers import (
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
)

from module import BaseTransformer, arg_to_scheduler, PrefixDialogModule
from callback import LoggingCallback, Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}


arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())



def generic_train(
    model: BaseTransformer,
    args: argparse.Namespace,
    early_stopping_callback=None,
    logger=False,  
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    ckpt_path=None,
    **extra_train_kwargs
):
    pl.seed_everything(args.seed)

    
    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)

    
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            monitor="loss",
            mode="min",
            save_top_k=args.save_top_k,
        )

    if early_stopping_callback:
        extra_callbacks.append(early_stopping_callback)

    if logging_callback is None:
        logging_callback = LoggingCallback()

    train_params = {}

    
    if args.fp16:
        train_params["precision"] = 16
        train_params["amp_level"] = args.fp16_opt_level

    if args.gpus > 1:
        train_params["strategy"] = "ddp"   

    train_params["accumulate_grad_batches"] = args.gradient_accumulation_steps
    train_params["accelerator"] = extra_train_kwargs.get("accelerator", None)
    train_params["profiler"] = extra_train_kwargs.get("profiler", None)
    train_params["max_epochs"] = args.num_train_epochs

    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary=None,
        callbacks=[logging_callback, checkpoint_callback] + extra_callbacks,
        logger=logger,
        enable_checkpointing=True,
        **train_params,
    )

    if args.do_train:
        trainer.fit(model, ckpt_path=ckpt_path)

    return trainer


def main(args, model=None) -> PrefixDialogModule:

    Path(args.output_dir).mkdir(exist_ok=True)  
    check_output_dir(args, expected_items=3)   
    

    args.gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    if model is None:
        model: PrefixDialogModule = PrefixDialogModule(args)  

    
    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    else:
        es_callback = False

    lower_is_better = args.val_metric == "loss"
    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(
            args.output_dir, model.val_metric, args.save_top_k, lower_is_better
        ),
        early_stopping_callback=es_callback,
        logger=False,   
        ckpt_path=args.resumed_ckpt_file
    )

    pickle_save(model.hparams, model.output_dir / "hparams.pkl")

    if not args.do_predict:
        return model

    if args.do_train and (args.resumed_ckpt_file is not None):
        
        args.resumed_ckpt_file = None

    logging.info(model.hparams)
    trainer.test(model=model, ckpt_path=args.resumed_ckpt_file)

    return model


def add_args(parser):

    
    model_group = parser.add_argument_group("model parameters")
    model_group.add_argument(
        "--max_source_length",
        default=32,   
        type=int,
    )
    model_group.add_argument(
        "--max_knowledge_length",
        default=32,  
        type=int,
    )
    model_group.add_argument(
        "--max_target_length",
        default=16,   
        type=int,
    )
    model_group.add_argument(
        "--freeze_encoder",
        action="store_true",
    )
    model_group.add_argument(
        "--freeze_embeds",
        action="store_true",
    )
    model_group.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        default=True,
    )
    model_group.add_argument(
        "--n_train",
        type=int,
        default=-1,
        required=False,
    )
    model_group.add_argument(
        "--n_val",
        type=int,
        default=-1,
        required=False,
    )
    model_group.add_argument(
        "--n_test",
        type=int,
        default=-1,
        required=False,
    )
    model_group.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        required=False,
    )
    model_group.add_argument(
        "--eval_beams",
        type=int,
        default=None,
        required=False,
    )
    model_group.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=None,
        required=False,
    )
    model_group.add_argument(
        "--val_metric",
        type=str,
        default="loss",
        choices=["loss", "f1"],
    )
    model_group.add_argument(
        "--eval_max_gen_length",
        default=32,
        type=int,
    )
    model_group.add_argument(
        "--eval_min_gen_length",
        default=2,
        type=int,
    )
    model_group.add_argument(
        "--save_top_k",
        type=int,
        default=1,
        required=False,
    )
    model_group.add_argument(
        "--early_stopping_patience",
        type=int,
        default=-1,
        required=False,
    )

    base_transformer_group = parser.add_argument_group("base transformer parameters")
    base_transformer_group.add_argument(
        "--model_name_or_path",
        default="",
        type=str,
    )
    base_transformer_group.add_argument(
        "--pfxKlgModel_name_or_path",
        default=None,
    )
    base_transformer_group.add_argument(
        "--resumed_ckpt_file",
        default=None,
    )
    base_transformer_group.add_argument(
        "--cache_dir",
        default="",
        type=str,
    )
    base_transformer_group.add_argument(
        "--config_name",
        default="",
        type=str,
    )
    base_transformer_group.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
    )
    base_transformer_group.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
    )
    base_transformer_group.add_argument(
        "--lr_scheduler",
        default="linear",
        choices=arg_to_scheduler_choices,
        type=str,
    )
    base_transformer_group.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
    )
    base_transformer_group.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
    )
    base_transformer_group.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
    )
    base_transformer_group.add_argument(
        "--num_workers",
        default=1,
        type=int,
    )
    base_transformer_group.add_argument(
        "--num_train_epochs",
        default=5,
        type=int,
    )
    base_transformer_group.add_argument(
        "--train_batch_size",
        default=4,  
        type=int
    )
    base_transformer_group.add_argument(
        "--eval_batch_size",
        default=4,
        type=int
    )

    generic_group = parser.add_argument_group("generic parameters")
    generic_group.add_argument(
        "--data_dir",
        default="",
        type=str,
    )

    generic_group.add_argument(
        "--output_dir",
        default="",
        type=str,
    )

    generic_group.add_argument(
        "--fp16",
        action="store_true",
    )

    generic_group.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2"
    )

    generic_group.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
    )
    generic_group.add_argument(
        "--do_train",
        action="store_true",
    )
    generic_group.add_argument(
        "--do_predict",
        default=True,
        action="store_true",
    )
    generic_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,   
    )
    generic_group.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    task_group = parser.add_argument_group("task-specific parameters")
    task_group.add_argument(
        "--tuning_mode",
        type=str,
        choices=["pt1", "pt2"],
        default="pt1"
    )
    task_group.add_argument(
        "--eval_type",
        type=str,
        default="",
        choices=["seen", "unseen", ""]
    )
    prefix_group = parser.add_argument_group("task-specific parameters")
    prefix_group.add_argument(
        "--mid_dim",
        type=int,
        default=128,
    )
    prefix_group.add_argument(
        "--prefix_dropout",
        type=float,
        default=0.0
    )
    prefix_group.add_argument(
        "--preseqlen",
        type=int,
        default=10,
    )

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()   
    parser = pl.Trainer.add_argparse_args(parser)   
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
