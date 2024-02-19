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
from torch.nn.functional import cosine_similarity

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

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
            DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag),
            
            nn.Linear(self.in_dim, self.in_dim),
            nn.LeakyReLU(),
            DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag),
            
            nn.Linear(self.in_dim, self.out_dim),
            nn.LeakyReLU(),
            DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag))

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
        self.K = args['K']
        self.S = args['S']
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
        
        # post-processing on front-end features
        # x = x.transpose(1, 2)   #(bs,feat_out_dim,frame_number)
        output, emb = self.backend(x)
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
        # print("feats.shape", feats.shape)
        # print("emb.shape", emb.shape)
        L_CE = self.loss_CE(output, labels)
        
        # print("feats.shape", feats.shape)
        L_CF1 = 1/real_bzs * loss_SCL(feats, K=self.K, S=self.S)
        
        # print("emb.shape", emb.shape)
        L_CF2 = 1/real_bzs * loss_SCL(emb, K=self.K, S=self.S)
        if self.loss_type == 1:
            return {'L_CE':L_CE, 'L_CF1':L_CF1, 'L_CF2':L_CF2}
        elif self.loss_type == 2:
            return {'L_CE':L_CE, 'L_CF1':L_CF1}
        elif self.loss_type == 3:
            return {'L_CE':L_CE, 'L_CF2':L_CF2}
        
def sim_metric_seq(mat1, mat2):
    if len(mat1.shape) == 2:
        mat1 = mat1.unsqueeze(-1)
        mat2 = mat2.unsqueeze(-1)
    return torch.bmm(mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)
def loss_SCL(batch, K, S, t=0.07):
    """
    Computes the loss based on the given input batch, K, and S.

    Args:
    - batch (torch.Tensor): Input batch tensor of shape [1 + K + M + S, H ...]
    - K (int): Number of augmented samples
    - S (int): Number of negative samples

    Returns:
    - loss (torch.Tensor): Computed loss value
    """
    bsz = batch.shape[0]  # Get batch size
    device = batch.device  # Get device

    M = len(batch) - 1 - K - S  # Calculate the number of other real samples
    remove_rows = list(range(bsz))
    remove_rows[1:1+K+M] = [] # positive sample rows
    
    logits_mat = sim_metric_seq(batch, batch)
    # print("logits_mat\n", logits_mat)
    logits_mat = logits_mat[remove_rows]
    self_mask = torch.ones((bsz,bsz))
    self_mask.diagonal().fill_(0) # mask on each data itself
    self_mask[:,0].fill_(0) # no need to compute the loss of the anchor
    self_mask = self_mask[remove_rows]
    # print("self_mask\n",self_mask)
    self_mask = self_mask.to(device)
    # Numerous stability improvements
    logits_max, _ = torch.max(logits_mat * self_mask, dim=1, keepdim=True)
    logits_mat_ = logits_mat - logits_max.detach()
    # compute log_prob
    exp_logits = torch.exp(logits_mat_ * self_mask) * self_mask
    # divide by the sum of exp_logits along the row
    log_prob = logits_mat_ - torch.log(exp_logits.sum(1, keepdim=True))
    # print("log_prob\n", log_prob)
    
    mask_ = torch.ones((bsz,bsz))
    mask_.diagonal().fill_(0) 
    # mask_[1:1+K+M,:].fill_(0) # mask on augmented and positive samples
    mask_[0,1+K+M:].fill_(0) # mask on anchor to negative samples
    mask_[1+K+M:,1:1+K+M].fill_(0) # mask on negative samples to augmented and positive samples
    mask_ = mask_.to(device)
    
    # print("remove_rows", remove_rows)   
    
    mask_ = mask_[remove_rows]
    # log_prob = log_prob[remove_rows]
    # print("mask_\n", mask_)
    
    mean_log_prob_pos = (mask_ * log_prob).sum(1) / mask_.sum(1)
    # print("mean_log_prob_pos.shape", mean_log_prob_pos.shape)
    # print(mean_log_prob_pos)
    # loss
    loss = - mean_log_prob_pos
    loss = loss.mean()

    return loss