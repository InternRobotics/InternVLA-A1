from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer

import torch

from lerobot.utils.constants import (
    OBS_STATE, 
    ACTION, 
    OBS_IMAGES, 
    OBS_LANGUAGE_TOKENS, 
    OBS_LANGUAGE_ATTENTION_MASK, 
)
from lerobot.transforms.core import DataTransformFn, DataDict


@DataTransformFn.register_subclass("internvl3_tokenizer_internvla_a1")
@dataclass
class InternVL3TokenizerTransformFn(DataTransformFn):
    pretrained_model_name_or_path: str = 'OpenGVLab/InternVL3-1B'
    max_length: int = 48
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)

    def __call__(self, data: DataDict) -> DataDict: 
        lang_inputs = self.tokenizer(
            data[self.task_key], 
            max_length=self.max_length, 
            padding_side=self.padding_side, 
            padding=self.padding, 
            truncation=self.truncation, 
        )

        data[OBS_LANGUAGE_TOKENS] = torch.tensor(lang_inputs.input_ids)
        data[OBS_LANGUAGE_ATTENTION_MASK] = torch.tensor(lang_inputs.attention_mask)

        return data
    

@DataTransformFn.register_subclass("unify_internvla_a1_inputs")
@dataclass
class UnifyInternVLAA1InputsTransformFn(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict: 
        data = {
            OBS_STATE: data[OBS_STATE], 
            ACTION: data[ACTION], 
            f"{OBS_IMAGES}.image0": data[f"{OBS_IMAGES}.image0"], 
            f"{OBS_IMAGES}.image1": data[f"{OBS_IMAGES}.image1"], 
            f"{OBS_IMAGES}.image2": data[f"{OBS_IMAGES}.image2"], 
            f"{OBS_IMAGES}.image0_mask": data[f"{OBS_IMAGES}.image0_mask"], 
            f"{OBS_IMAGES}.image1_mask": data[f"{OBS_IMAGES}.image1_mask"], 
            f"{OBS_IMAGES}.image2_mask": data[f"{OBS_IMAGES}.image2_mask"], 
            # "task": data["task"], 
            OBS_LANGUAGE_TOKENS: data[OBS_LANGUAGE_TOKENS], 
            OBS_LANGUAGE_ATTENTION_MASK: data[OBS_LANGUAGE_ATTENTION_MASK], 
        }
        return data


if __name__ == "__main__":
    sample = {
        "task": "This is a test sample.", 
    }
    tokenizer = InternVL3TokenizerTransformFn()
    output = tokenizer(sample)
    print(output)
