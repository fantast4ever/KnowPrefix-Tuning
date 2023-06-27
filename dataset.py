import os
import string
import torch
import json
import linecache
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict
from transformers import BartTokenizer
from transformers.utils import cached_property
from utils import pickle_load, trim_batch
from sampler import SortishSampler, DistributedSortishSampler
from nltk.corpus import stopwords


class AbstractSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")  
        if os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None:   
            self.src_lens = self.src_lens[:n_obs]

        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")


class LegacySeq2SeqDataset(AbstractSeq2SeqDataset):

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """Call tokenizer on src and tgt_lines"""
        index = index + 1  
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        source_inputs = self.encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = self.encode_line(self.tokenizer, tgt_line, self.max_target_length)
        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": target_ids,
        }

    def encode_line(self, tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
        """Only used by LegacyDataset"""
        return tokenizer(
            [line],
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            **self.dataset_kwargs,
        )

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["labels"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": y,
        }
        return batch


class KnowledgeSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        max_knowledge_length,
        type_path="train",
        n_obs=None,
        prefix="",
        
        preseqlen=10,
        tuning_mode="ft",
        model_type="bart",
        **dataset_kwargs
    ):
        super().__init__()

        self.data_file = Path(data_dir).joinpath(type_path + ".json")
        self.sep_token = tokenizer.sep_token
        self.preseqlen = preseqlen
        self.tuning_mode = tuning_mode
        self.model_type = model_type
        self.max_source_length = max_source_length
        self.stop_words = stopwords.words('english')

        self._data = []
        self._src_lens = []
        self.load(self.data_file)

        if self.tuning_mode == "pt2":
            self.max_target_length = max_target_length
        else:
            self.max_target_length = max_knowledge_length

        assert min(self._src_lens) > 0, f"found empty line."
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix else ""

        if n_obs is not None:
            self._src_lens = self._src_lens[:n_obs]

        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

    def get_bow_response(self, response):
        response = response.translate(str.maketrans('', '', string.punctuation))
        tokenized_response = nltk.word_tokenize(response)
        response = [w for w in tokenized_response if w not in self.stop_words]
        return " ".join(response)

    def load(self, data_file):
        with open(data_file, mode="r", encoding="utf-8") as frp:
            for _l in frp.readlines():
                _l = json.loads(_l)
                history = f" {self.sep_token} ".join(_l["history"])
                golden_knowledge = _l["knowledge"][0]
                response = _l["response"]

                bow_response = self.get_bow_response(response)

                self._src_lens.append(len(history))

                self._data.append(
                    {
                        "history": history,
                        "knowledge": golden_knowledge,
                        "response": response,
                        "bow_response": bow_response
                    }
                )

    def __len__(self):
        return len(self._src_lens)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self._src_lens, batch_size, shuffle=shuffle)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        indexed_data = self._data[index]  

        history = indexed_data["history"]
        knowledge = indexed_data["knowledge"]
        tgt_line = indexed_data["response"]
        bow_tgt = indexed_data['bow_response']
        source_line = self.prefix + history

        if self.tuning_mode == "pt1":
            tgt_line = knowledge

        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        source_inputs = self.encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = self.encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()

        if self.tuning_mode == "pt2":
            bow_target = self.encode_line(self.tokenizer, bow_tgt, self.max_target_length)
            bow_target_ids = bow_target['input_ids'].squeeze()
            return {
                "input_ids": source_ids,
                "attention_mask": src_mask,
                "labels": target_ids,
                "bow_targets": bow_target_ids
            }

        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": target_ids,
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        pad_token_id = self.pad_token_id
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["labels"] for x in batch])

        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        y = trim_batch(target_ids, pad_token_id)

        if self.tuning_mode == "pt2":

            bow_target_ids = torch.stack([x['bow_targets'] for x in batch])
            bow_y = trim_batch(bow_target_ids, pad_token_id)

            return {
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "labels": y,
                'bow_labels': bow_y
            }

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": y,
        }

    def encode_line(self, tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
        """Only used by LegacyDataset"""

        return tokenizer(
            [line],
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            **self.dataset_kwargs,
        )
