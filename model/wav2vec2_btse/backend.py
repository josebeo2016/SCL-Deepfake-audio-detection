import torch
import torch.nn as nn
import torch.nn.functional as F
# try:
#     from linear import MLP
#     from xlsr import SSLModel
# except:
#     from .linear import MLP
#     from .xlsr import SSLModel
try:
    from model.wav2vec2_btse.linear import MLP
    from model.xlsr import SSLModel
except:
    from .linear import MLP
    from ..xlsr import SSLModel

class Model(nn.Module):
    def __init__(self, config, device):
        """
        config: dict, configuration for the model
        device: torch.device, device to run the model
        """
        super(Model, self).__init__()
        self.device=device
        self.is_train = True
        self.ssl_model = SSLModel(device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.mlp = MLP(128, 128, 3, 2, 0.5)
        self.out_dim = 128
        
    def forward(self, x):
        """
        x: tensor, (batch, audio_len) for raw audio input
        """
        # XLS-R 300M frontend
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1), self.is_train) #(bs,frame_number,feat_dim)
        # Linear Layer to reduce dimension
        x_feat = self.LL(x_ssl_feat) #(bs,frame_number,input_dim)
        # MLP backend
        logits, emb = self.mlp(x_feat) #(bs,num_classes) & (bs,out_dim)
        
        return logits, emb, x_feat
    
