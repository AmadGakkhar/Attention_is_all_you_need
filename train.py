import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_path"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer


def get_ds(config):
    ds = load_dataset(
        "opus_books", f'{config["src_lang"]}-{config["trg_lang"]}', split="train"
    )

    # Build Tokenizer

    tokenizer_src = get_or_build_tokenizer(config, ds, config["src_lang"])
    tokenizer_trg = get_or_build_tokenizer(config, ds, config["trg_lang"])

    # Trani test split

    train_ds_size = int(0.9 * len(ds))
    val_ds_size = int(0.1 * len(ds))

    train_ds, val_ds = random_split(ds, [train_ds_size, val_ds_size])
