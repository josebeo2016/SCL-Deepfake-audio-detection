import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import os
from .loss_metrics import supcon_loss

___author__ = "PhucDT"
__email__ = "phucdt@soongsil.ac.kr"

############################
## FOR fine-tuned SSL MODEL
############################

BASE_DIR=os.path.dirname(os.path.abspath(__file__))

class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        
        cp_path = os.path.join(BASE_DIR,'pretrained/xlsr2_300m.pt')
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device

        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        
        if True:
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
                
            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)['x']
            # print(emb.shape)
        return emb
class DropoutForMC(nn.Module):
    """Dropout layer for Bayesian model
    THe difference is that we do dropout even in eval stage
    """
    def __init__(self, p, dropout_flag=True):
        super(DropoutForMC, self).__init__()
        self.p = p
        self.flag = dropout_flag
        return
        
    def forward(self, x):
        return torch.nn.functional.dropout(x, self.p, training=self.flag)

class BackEnd(nn.Module):
    """Back End Wrapper
    """
    def __init__(self, input_dim, out_dim, num_classes, 
                 dropout_rate, dropout_flag=True):
        super(BackEnd, self).__init__()

        # input feature dimension
        self.in_dim = input_dim
        # output embedding dimension
        self.out_dim = out_dim
        # number of output classes
        self.num_class = num_classes
        
        # dropout rate
        self.m_mcdp_rate = dropout_rate
        self.m_mcdp_flag = dropout_flag
        
        # a simple full-connected network for frame-level feature processing
        self.m_frame_level = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.LeakyReLU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag),
            
            nn.Linear(self.in_dim, self.in_dim),
            nn.LeakyReLU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag),
            
            nn.Linear(self.in_dim, self.out_dim),
            nn.LeakyReLU(),
            torch.nn.Dropout(self.m_mcdp_rate)
        )

        # linear layer to produce output logits 
        self.m_utt_level = nn.Linear(self.out_dim, self.num_class)
        
        return

    def forward(self, feat):
        """ logits, emb_vec = back_end_emb(feat)

        input:
        ------
          feat: tensor, (batch, frame_num, feat_feat_dim)

        output:
        -------
          logits: tensor, (batch, num_output_class)
          emb_vec: tensor, (batch, emb_dim)
        
        """
        # through the frame-level network
        # (batch, frame_num, self.out_dim)
        feat_ = self.m_frame_level(feat)
        
        # average pooling -> (batch, self.out_dim)
        feat_utt = feat_.mean(1)
        
        # output linear 
        logits = self.m_utt_level(feat_utt)
        return logits, feat_utt

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
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.selu = nn.SELU(inplace=True)
        
        self.loss_CE = nn.CrossEntropyLoss()
        self.backend = BackEnd(128, 128, 2, 0.5, True)
        
        self.sim_metric_seq = lambda mat1, mat2: torch.bmm(
            mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)
        # Post-processing
        
    def _forward(self, x):
        #-------pre-trained Wav2vec model fine tunning ------------------------##

        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim)
        feats = x
        x = nn.ReLU()(x)
        
        # output [batch, 2]
        # emb [batch, 128]
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
    
