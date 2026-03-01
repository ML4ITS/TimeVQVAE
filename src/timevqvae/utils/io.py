import os
import tempfile

import torch

from timevqvae.utils.paths import get_root_dir


def save_model(models_dict: dict, dirname="saved_models", id: str = ""):
    """Save model state_dicts under repository root, with fallback to temp dir."""
    try:
        if not os.path.isdir(get_root_dir().joinpath(dirname)):
            os.mkdir(get_root_dir().joinpath(dirname))

        id_ = id[:] if id == "" else f"-{id}"
        for model_name, model in models_dict.items():
            torch.save(model.state_dict(), get_root_dir().joinpath(dirname, model_name + id_ + ".ckpt"))
    except PermissionError:
        dirname = tempfile.gettempdir()
        print(
            f"\nThe trained model is saved in the following temporary dirname due to some permission error: {dirname}.\n"
        )

        id_ = id[:] if id == "" else f"-{id}"
        for model_name, model in models_dict.items():
            torch.save(model.state_dict(), get_root_dir().joinpath(dirname, model_name + id_ + ".ckpt"))
