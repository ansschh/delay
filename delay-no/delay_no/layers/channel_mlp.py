import torch.nn as nn

class ChannelMLP(nn.Sequential):
    """
    Channel MLP used for lifting and projection operations in the FNO architecture.
    """
    def __init__(self, in_ch, hidden, out_ch, act=nn.GELU, dropout=0.0):
        super().__init__(
            nn.Linear(in_ch, hidden),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_ch)
        )
