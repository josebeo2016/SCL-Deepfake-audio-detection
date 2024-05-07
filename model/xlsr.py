import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import os

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        
        cp_path = os.path.join(BASE_DIR,'../pretrained/xlsr2_300m.pt')
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.model.to(device)

        self.out_dim = 1024
        return

    def extract_feat(self, input_data, is_train=True):
        
        # put the model to GPU if it not there
        # if next(self.model.parameters()).device != input_data.device \
        #    or next(self.model.parameters()).dtype != input_data.dtype:
        #     self.model.to(input_data.device, dtype=input_data.dtype)
        #     self.model.train()
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
            
        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        # print(emb.shape)
        return emb