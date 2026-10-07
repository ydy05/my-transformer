import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if not tokenizer_path.exists():
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )

        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang),
            trainer=trainer
        )

        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):

    # English -> Chinese
    ds_raw = load_dataset(
        "Helsinki-NLP/news_commentary",
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train"
    )

    # 创建英文 tokenizer
    tokenizer_src = get_or_build_tokenizer(
        config,
        ds_raw,
        config["lang_src"]
    )

    # 创建中文 tokenizer
    tokenizer_tgt = get_or_build_tokenizer(
        config,
        ds_raw,
        config["lang_tgt"]
    )

    # 90% training, 10% validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(
        ds_raw,
        [train_ds_size, val_ds_size]
    )

    return train_ds_raw, val_ds_raw, tokenizer_src, tokenizer_tgt
