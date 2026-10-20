"""Device selection shared by the whole project."""
import os

# Let PyTorch fall back to the CPU for operations MPS does not implement.
# This only takes effect if it is set before torch is first imported, so it
# lives here, at the top of the module every script imports first.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402  (must come after the environment variable above)


def get_device() -> torch.device:
    """Return the best available device: cuda, then mps (Apple Silicon), then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
