"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

import numpy
import pandas as pd
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class SFTDataset(Dataset):

    def __init__(self, parquet_files: str | ListConfig, tokenizer: str|PreTrainedTokenizer, config: DictConfig):
        prompt_key: str = config.get("prompt_key", "prompt")
        prompt_dict_keys: list[str] = config.get("prompt_dict_keys", None)
        response_key: str = config.get("response_key", "response")
        response_dict_keys: list[str] = config.get("response_dict_keys", None)
        max_length: int = config.get("max_length", 1024)
        truncation: str = config.get("truncation", "error")
        use_shm: bool = config.get("use_shm", False)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        assert truncation in ["error", "left", "right"]
        self.truncation: str = truncation
        self.use_shm: bool = use_shm

        if not isinstance(parquet_files, ListConfig):
            parquet_files: list[str] = [parquet_files]

        self.parquet_files: list[str]|ListConfig = parquet_files
        if isinstance(tokenizer, str):
            tokenizer: PreTrainedTokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key: list[str] = prompt_key if isinstance(prompt_key, tuple | list) else [prompt_key]
        self.response_key: list[str] = response_key if isinstance(response_key, tuple | list) else [response_key]
        self.prompt_dict_keys: list[str] = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys: list[str] = response_dict_keys if response_dict_keys else []

        self.max_length: int = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls: pd.Series):

            while isinstance(ls, pd.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes: list[pd.DataFrame] = []
        for parquet_file in self.parquet_files:
            dataframe: pd.DataFrame = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe: pd.DataFrame = pd.concat(dataframes)
        self.prompts: pd.Series = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts: list[str] = self.prompts.tolist()
        self.responses: pd.Series = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
        self.responses: list[str] = self.responses.tolist()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer: PreTrainedTokenizer = self.tokenizer

        prompt: str = self.prompts[item]
        response: str = self.responses[item]

        prompt_chat: list[dict[str, str]] = [{"role": "user", "content": prompt}]

        prompt_chat_str: str = tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        response_chat_str: str = response + tokenizer.eos_token

        prompt_ids_output: dict[str, torch.Tensor] = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids: torch.Tensor = prompt_ids_output["input_ids"][0]
        prompt_attention_mask: torch.Tensor = prompt_ids_output["attention_mask"][0]

        response_ids_output: dict[str, torch.Tensor] = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids: torch.Tensor = response_ids_output["input_ids"][0]
        response_attention_mask: torch.Tensor = response_ids_output["attention_mask"][0]

        prompt_length: int = prompt_ids.shape[0]
        response_length: int = response_ids.shape[0]

        input_ids: torch.Tensor = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask: torch.Tensor = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        sequence_length: int = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids: torch.Tensor = (
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
                * self.tokenizer.pad_token_id
            )
            padded_attention_mask: torch.Tensor = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids: torch.Tensor = torch.cat((input_ids, padded_input_ids))
            attention_mask: torch.Tensor = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids: torch.Tensor = compute_position_id_with_mask(attention_mask)

        loss_mask: torch.Tensor = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
