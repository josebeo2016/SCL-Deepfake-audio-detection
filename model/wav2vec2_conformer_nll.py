import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import os
from model.loss_metrics import supcon_loss
from .xlsr import SSLModel

from .conformer import ConformerBlock
from torch.nn.modules.transformer import _get_clones

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

############################
## Conformer model
############################
class MyConformer(nn.Module):
  def __init__(self, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=16, n_encoders=1):
    super(MyConformer, self).__init__()
    self.dim_head=int(emb_size/heads)
    self.dim=emb_size
    self.heads=heads
    self.kernel_size=kernel_size
    self.n_encoders=n_encoders
    self.encoder_blocks=_get_clones( ConformerBlock( dim = emb_size, dim_head=self.dim_head, heads= heads, 
    ff_mult = ffmult, conv_expansion_factor = exp_fac, conv_kernel_size = kernel_size),
    n_encoders)
    self.class_token = nn.Parameter(torch.rand(1, emb_size))
    self.fc5 = nn.Linear(emb_size, 2)

  def forward(self, x): # x shape [bs, tiempo, frecuencia]
    x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])#[bs,1+tiempo,emb_size]
    for layer in self.encoder_blocks:
            x = layer(x) #[bs,1+tiempo,emb_size]
    embedding=x[:,0,:] #[bs, emb_size]
    out=self.fc5(embedding) #[bs,2]
    return out, embedding


class Model(nn.Module):
    def __init__(self, args, device, is_train = True):
        super().__init__()
        self.device = device
        self.is_train = is_train
        self.flag_fix_ssl = args['flag_fix_ssl']
        self.contra_mode = args['contra_mode']
        self.loss_type = args['loss_type']
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, args['conformer']['emb_size'])
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.selu = nn.SELU(inplace=True)
        
        self.backend=MyConformer(**args['conformer'])
        self.loss_CE = nn.CrossEntropyLoss()
        
        self.sim_metric_seq = lambda mat1, mat2: torch.bmm(
            mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)
        # Post-processing
        
    def _forward(self, x):
        #-------pre-trained Wav2vec model fine tunning ------------------------##

        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1), self.is_train) #(bs,frame_number,feat_dim)
        x = self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim)
        feats = x
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)

        # output [batch, 2]
        # emb [batch, emb_size]
        output, emb = self.backend(x)
        output = F.log_softmax(output, dim=1)
        if (self.is_train):
            return output, feats, emb
        return output
    
    def forward(self, x_big):
        
        if (self.is_train):
            # x_big is a tensor of [1, length, bz]
            # convert to [bz, length]
            x_big = x_big.squeeze(0).transpose(0,1)
            output, feats, emb = self._forward(x_big)
            return output, feats, emb
        else:
            # in inference mode, we don't need the emb
            # the x_big now is a tensor of [bz, length]
            return self._forward(x_big)
        
    
    def loss(self, output, feats, emb, labels, config):
        
        real_bzs = output.shape[0]
        n_views = 1.0
        
        # print("output.shape", output.shape)
        # print("labels.shape", labels.shape)
        L_CE = self.loss_CE(output, labels)
        
        # reshape the feats to match the supcon loss format
        feats = feats.unsqueeze(1)
        # print("feats.shape", feats.shape)
        L_CF1 = supcon_loss(feats, labels=labels, contra_mode=self.contra_mode, sim_metric=self.sim_metric_seq)
        
        # reshape the emb to match the supcon loss format
        emb = emb.unsqueeze(1)
        emb = emb.unsqueeze(-1)
        # print("emb.shape", emb.shape)
        L_CF2 = supcon_loss(emb, labels=labels, contra_mode=self.contra_mode, sim_metric=self.sim_metric_seq)
        if self.loss_type == 1:
            return {'L_CE':L_CE, 'L_CF1':L_CF1, 'L_CF2':L_CF2}
        elif self.loss_type == 2:
            return {'L_CE':L_CE, 'L_CF1':L_CF1}
        elif self.loss_type == 3:
            return {'L_CE':L_CE, 'L_CF2':L_CF2}
        # ablation study
        elif self.loss_type == 4:
            return {'L_CE':L_CE}
        elif self.loss_type == 5:
            return {'L_CF1':L_CF1, 'L_CF2':L_CF2}
        
    def loss_function_(self, batch_size, anchor_name, anchor, positive, negative):
        # anchor: tensor (b,H) where H is the feature dimension, b is the batch size
        # positive: tensor (b,K,H) where  K is the number of augmentations
        # negative: tensor (b,S,H) where S is the number of vocoded samples

        total = torch.cat((positive, negative), dim=0) # Combining positive and negative samples
        loss = 0

        # iterate over the batch
        for i in range(batch_size):
            a = anchor[i,:].unsqueeze(0) # Selecting anchor sample
            P = positive[i*batch_size:(i+1)*batch_size,:] # getting positive samples for anchor i
            N = negative[i*batch_size:(i+1)*batch_size,:] # getting negative samples for anchor i

            # term 1
            num1 = torch.exp(torch.mm(a, P.t())) # f(x_i, x_p) = x_i* x_p
            den1 = torch.exp(torch.mm(a, total[i*batch_size:(i+1)*batch_size,:].t())).sum()
            term_1 = - torch.log(num1 / den1+ 1e-7).sum()

            # term 2
            term_2 = 0
            for j in range(N.shape[0]):
                for k in range(j+1,N.shape[0]):
                    v1 = N[j,:].unsqueeze(0) # getting vocoded sample j
                    v2 = N[k,:].unsqueeze(0) # getting vocoded sample k
                    num2 = torch.exp(torch.mm(v1, v2.t())) # f(x_v, x_q) = x_v * x_q
                    den2 = torch.exp(torch.mm(v1, total[i*batch_size:(i+1)*batch_size,:].t())).sum()
                    term_2 += - torch.log(num2 / den2 + 1e-7)
            term_2 /=  N.shape[0] * (N.shape[0] - 1)

            loss += term_1 + term_2

        loss /= batch_size

        return loss
    
