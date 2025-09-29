# models/__init__.py
from .ffnn import FeedforwardNN
from .mlp_dropout import MLPDropout
from .simple_cnn import SimpleCNN

def get_model(name, input_dim=None, **kwargs):
    if name == "ffnn":
        return FeedforwardNN(input_dim, **kwargs)
    elif name == "mlp_dropout":
        return MLPDropout(input_dim, **kwargs)
    elif name == "simple_cnn":
        return SimpleCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")
