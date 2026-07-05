from .baseline import FrameCNNTemporalAvg
from .cnn_lstm import FrameCNNLSTM
from .r3d import R3DClassifier
from .swin3d import Swin3DClassifier

MODEL_REGISTRY = {
    "baseline": FrameCNNTemporalAvg,
    "cnn_lstm": FrameCNNLSTM,
    "r3d": R3DClassifier,
    "swin3d": Swin3DClassifier,
}

# Which tensor layout / normalization each model expects.
MODEL_FAMILY = {
    "baseline": "2d",
    "cnn_lstm": "2d",
    "r3d": "3d",
    "swin3d": "3d",
}

MODEL_NORMALIZE = {
    "baseline": "tanh",
    "cnn_lstm": "tanh",
    "r3d": "tanh",
    "swin3d": "imagenet",
}
