# KnowPrefix-Tuning #

---

This repo contains the code of the paper "**KnowPrefix-Tuning: A Two-Stage Prefix-Tuning Framework for Knowledge-Grounded Dialogue Generation**", [ECML-PKDD 2023](https://2023.ecmlpkdd.org/)

---

## Requirments ##

This code is tested on Python 3.8, Pytorch 1.7.1, and transformers 4.18.0

`pip install -r requirements.txt`

---

## Prepare Data ##
We conduct experiments on Wizard of Wikipedia and CmuDog datasets.
The data preprocessing procedure follows [PARLAI](https://parl.ai/).
The processing details are under `parlai_wow` and `parlai_cmudog` folders.

Accessing the processed datasets directly through the following links: WoW; CMU-DoG

---

## Training ##
The KnowPrefix-Tuning is trained in two stages.
In the first stage, the knowledge prefix is optimized by generating the evidence knowledge piece regarding dialogue context.
In the second stage, a response generator generates a knowledgeable response grounding on the learned knowledge prefix and the dialogue context.



**Stage 1: Knowledge Prefix Generation**
<pre>
export CUDA_VISIBLE_DEVICES=0,1,...
MODEL="bart-large or gpt2-large"
DATASET="wow or cmu_dog"
python ../finetune.py \
    --max_source_length 128 \
    --max_knowledge_length 128 \
    --max_target_length 64 \
    --eval_max_gen_length 128 \
    --model_name_or_path "HuggingFace PLMs" \
    --data_dir "The preprocessed dataset" \
    --output_dir " " \
    --learning_rate 3e-5 \
    --lr_scheduler "linear" \
    --warmup_steps 2000 \
    --num_train_epochs 10 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --n_train -1 \
    --n_val -1 \
    --n_test -1 \
    --tuning_mode "pt1" \
    --eval_type "seen" \
    --mid_dim 800 \
    --preseqlen 20 \
    --do_train
</pre>

**Stage 2: Knowledgeable Response Generation:**
<pre>
export CUDA_VISIBLE_DEVICES=0,1,...
MODEL="bart-large or gpt2-large"
DATASET="wow or cmu_dog"
python ../finetune.py \
    --max_source_length 128 \
    --max_knowledge_length 128 \
    --max_target_length 64 \
    --eval_max_gen_length 32 \
    --model_name_or_path "HuggingFace PLMs" \
    --pfxKlgModel_name_or_path "Model trained on Stage 1. (/path/best_pt1)" \
    --data_dir "The preprocessed dataset" \
    --output_dir " " \
    --learning_rate 3e-5 \
    --lr_scheduler "linear" \
    --warmup_steps 2000 \
    --num_train_epochs 30 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --n_train -1 \
    --n_val -1 \
    --n_test -1 \
    --tuning_mode "pt2" \
    --eval_type "seen" \
    --mid_dim 800 \
    --preseqlen 20 \
    --do_train
</pre>

---

## Inference ##
Once the training is finished, run the following script to conduct the inference stage.

<pre>
export CUDA_VISIBLE_DEVICES=0
MODEL="bart-large or gpt2-large"
DATASET="wow or cmu_dog"
python ../finetune.py \
    --max_source_length 128 \
    --max_knowledge_length 128 \
    --max_target_length 64 \
    --eval_max_gen_length 32 \
    --resumed_ckpt_file ".ckpt file trained on stage 2 (/path/xxx.ckpt)" \
    --model_name_or_path "HuggingFace PLMs" \
    --pfxKlgModel_name_or_path "Model trained on Stage 1." \
    --data_dir "The preprocessed dataset" \
    --output_dir " " \
    --eval_batch_size 16 \
    --n_train -1 \
    --n_val -1 \
    --n_test -1 \
    --tuning_mode "pt2" \
    --eval_type "seen" \
    --mid_dim 800 \
    --preseqlen 20
</pre>

---

## Acknowledgement ##
We thank all the anonymous reviewers for their insightful comments.



