import torch


def resolve_device(name: str = "auto") -> torch.device:
    """Résout 'auto' vers le meilleur accélérateur disponible."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
