import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset, causal_mask

from pathlib import Path


def get_all_sentences(ds, lang):
    """
    Generator that yields all sentences in the given dataset `ds` for language `lang`.

    Args:
        ds: A dataset containing sentences in multiple languages.
        lang: The language for which to yield sentences.

    Yields:
        str: The next sentence in the given language.
    """
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    """
    Retrieves or builds a tokenizer for the specified language.

    This function checks if a tokenizer already exists for the given language
    at the specified path. If it does not exist, it creates a new tokenizer
    using the WordLevel model with a specified set of special tokens and a
    minimum frequency. The tokenizer is trained using all sentences from the
    provided dataset for the specified language. The tokenizer is then saved
    for future use. If the tokenizer already exists, it is loaded from the file.

    Args:
        config (dict): Configuration containing the path template for the tokenizer.
        ds: A dataset containing sentences in multiple languages.
        lang (str): The language for which to build or retrieve the tokenizer.

    Returns:
        Tokenizer: A tokenizer object for the specified language.
    """
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
    """
    Loads a dataset and prepares tokenizers for source and target languages.

    This function loads the specified dataset, constructs tokenizers for both
    source and target languages, and then splits the dataset into training and
    validation sets.

    Args:
        config (dict): Configuration containing source and target language codes.

    Returns:
        tuple: A tuple containing the training and validation datasets.
    """
    ds = load_dataset(
        "opus_books", f'{config["src_lang"]}-{config["trg_lang"]}', split="train"
    )

    # Build Tokenizer

    tokenizer_src = get_or_build_tokenizer(config, ds, config["src_lang"])
    tokenizer_trg = get_or_build_tokenizer(config, ds, config["trg_lang"])

    # Trani test split

    train_ds_size = int(0.9 * len(ds))
    val_ds_size = int(0.1 * len(ds))

    train_ds_raw, val_ds_raw = random_split(ds, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_trg,
        config["src_lang"],
        config["trg_lang"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_trg,
        config["src_lang"],
        config["trg_lang"],
        config["seq_len"],
    )

    max_len_src = 0
    max_len_trg = 0

    for item in ds:
        src_ids = tokenizer_src.encode(item["translation"][config["src_lang"]]).ids
        trg_ids = tokenizer_trg.encode(item["translation"][config["trg_lang"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_trg = max(max_len_trg, len(trg_ids))

    print("Max length of source sentence: ", max_len_src)
    print("Max length of target sentence: ", max_len_trg)

    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg
