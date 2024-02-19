import torch
import torch.nn as nn


class MLP(nn.Module):
    """Back End Wrapper
    """
    def __init__(self, input_dim, out_dim, num_layers, num_classes, 
                 dropout_rate):
        super(MLP, self).__init__()

        # input feature dimension
        self.in_dim = input_dim
        # output embedding dimension
        self.out_dim = out_dim
        # number of output classes
        self.num_class = num_classes
        # number of layers
        self.num_layers = num_layers
        
        # dropout rate
        self.m_mcdp_rate = dropout_rate
        
        # a simple full-connected network for frame-level feature processing
        self.m_frame_level = nn.Sequential()
        for i in range(self.num_layers-1):
            self.m_frame_level.add_module("linear_{}".format(i), nn.Linear(self.in_dim, self.in_dim))
            self.m_frame_level.add_module("relu_{}".format(i), nn.LeakyReLU())
            self.m_frame_level.add_module("dropout_{}".format(i), nn.Dropout(self.m_mcdp_rate))
        
        # last layer
        self.m_frame_level.add_module("linear_{}".format(self.num_layers-1), nn.Linear(self.in_dim, self.out_dim))
        self.m_frame_level.add_module("relu_{}".format(self.num_layers-1), nn.LeakyReLU())
        self.m_frame_level.add_module("dropout_{}".format(self.num_layers-1), nn.Dropout(self.m_mcdp_rate))
        
        
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
        # feat_ = self.m_frame_level(feat)
        
        # average pooling -> (batch, self.out_dim)
        if len(feat_.shape) == 3:
            feat_utt = feat_.mean(1)
        # feat_utt = feat_.mean(1)
        else:
            feat_utt = feat_
        
        # output linear 
        logits = self.m_utt_level(feat_utt)
        return logits, feat_utt