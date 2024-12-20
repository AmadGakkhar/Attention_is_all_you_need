from pathlib import Path


def get_config():
    return {
        "batch_size": 2,
        "num_epoch": 20,
        "lr": 0.0001,
        "seq_len": 350,
        "d_model": 512,
        "src_lang": "en",
        "trg_lang": "it",
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": None,
        "tokenizer_path": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(",") / model_folder / model_filename)
