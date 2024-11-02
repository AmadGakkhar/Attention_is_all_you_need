import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset, causal_mask
from tqdm import tqdm

from pathlib import Path

from model import build_transformer

from torch.utils.tensorboard import SummaryWriter

from config import get_weights_file_path, get_config


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
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
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
    val_ds_size = len(ds) - train_ds_size

    print("Total Samples ", len(ds))
    print("Train Samples ", train_ds_size)
    print("Val Samples ", val_ds_size)

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


def get_model(config, vocab_src_len, vocab_trg_len):
    model = build_transformer(
        vocab_src_len,
        vocab_trg_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )
    return model


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg = get_ds(config)

    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size()
    )
    model = model.to(device)

    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    initial_epoch = 0
    global_step = 0

    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Loading model weights from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_trg.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epoch"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch["encoder_input"].to(device)  # (batch, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (batch, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(
                device
            )  # (batch, 1, seq_len, seq_len)

            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (batch, seq_len, d_model)

            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (batch, seq_len, d_model)

            proj_output = model.project(
                decoder_output
            )  # (batch, seq_len, trg_vocab_size)

            label = batch["label"].to(device)  # (batch, seq_len)

            loss = loss_fn(
                proj_output.view(-1, tokenizer_trg.get_vocab_size()), label.view(-1)
            )

            batch_iterator.set_postfix({"Loss": f"{loss.item():6.3f}"})

            # Log loss in Tensorboard

            writer.add_scalar("Train Loss", loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            get_weights_file_path(config, f"{epoch:02d}"),
        )


if __name__ == "__main__":
    config = get_config()
    train_model(config)
