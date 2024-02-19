from audioop import bias
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter
import math
try:
    from model.loss_metrics import supcon_loss
    from model.wav2vec2_btse.biosegment import Wav2bioCNN
except:
    from ..loss_metrics import supcon_loss
    from ..wav2vec2_btse.biosegment import Wav2bioCNN

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

try:
    
    import transformer
    import commons
    import cnns2s
    from backend import Model as backend
# import biosegment
except:
    from . import commons
    from . import transformer
    from . import cnns2s
    from .backend import Model as backend


___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

__modifier__ = "Phucdt"


#####################
# BIO SCORING
#####################


def Mask_Generate(lengths, max_lengths= None, dtype= torch.float):
        '''
        lengths: [Batch]
        '''
        mask = torch.arange(max_lengths or torch.max(lengths))[None, :].to(lengths.device) < lengths[:, None]    # [Batch, Time]
        return mask.unsqueeze(1).to(dtype)  # [Batch, 1, Time]

class BioEmbedding(nn.Module):
    def __init__(self, device, num_bios, out_channels) -> None:
        super(nn.Module, self).__init__()
        self.out_channels = out_channels
        self.num_embeddings = num_bios
        self.emb = nn.Embedding(
            num_embeddings=num_bios,
            embedding_dim=out_channels
        )
    def forward(self, x):
        """

        Args:
            x (tensor): Biological sound segmentation vector [B, T_bio]


        Returns:
            _type_: _description_ 
        """
        x = self.emb(x).transpose(2,1)# [B, out_channels, T_bio]
        

# https://github.com/bentrevett/pytorch-seq2seq/
class bioEncoderRNN(nn.Module):
    def __init__(self, d_args, device) -> None:
        super(bioEncoderRNN, self).__init__()

        self.device=device
        
        self.bio_emb = nn.Embedding(d_args['n_bios'], d_args['bio_dim'])
        self.bio_dim = d_args['bio_dim'] 

        # nn.init.normal_(self.bio_emb.weight, 0.0, d_args['bio_dim']**-0.5)
        
        # length scoring == # fc1 out features
        self.rnn = nn.GRU(d_args['bio_dim'], d_args['bio_rnn'], 1, batch_first=True)
        
        
        self.bio_scoring = nn.Linear(in_features = d_args['bio_rnn'],
			out_features = d_args['nb_fc_node'],bias=True)
        
    def forward(self, bio, bio_lengths):
        
        bio = self.bio_emb(bio) # [b, bio_length, bio_dim]
        # print(bio.size())
        bio_lengths = bio_lengths.cpu().numpy()
 
        bio = nn.utils.rnn.pack_padded_sequence(
                        bio, bio_lengths, batch_first=True)

        self.rnn.flatten_parameters()
        bio, hidden = self.rnn(bio)
        bio, _ = nn.utils.rnn.pad_packed_sequence(
            bio, batch_first=True)
        
        # hidden [b, bio_dim]
        bio_scoring = self.bio_scoring(hidden[-1,:,:])
        # bio_scoring = torch.tanh(self.bio_scoring(hidden[-1,:,:]))
        return bio_scoring

class bioEncoderRNNsmall(nn.Module):
    def __init__(self, d_args, device) -> None:
        super(bioEncoderRNNsmall, self).__init__()

        self.device=device
        
        self.bio_emb = nn.Embedding(d_args['n_bios'], d_args['bio_dim'])
        self.bio_dim = d_args['bio_dim'] 

        # nn.init.normal_(self.bio_emb.weight, 0.0, d_args['bio_dim']**-0.5)
        
        # length scoring == # fc1 out features
        self.rnn = nn.GRU(d_args['bio_dim'], d_args['bio_rnn'], 1, batch_first=True)
        
        
        self.bio_scoring = nn.Linear(in_features = d_args['bio_rnn'],
			out_features = d_args['bio_out'],bias=True)
        
    def forward(self, bio, bio_lengths):
        
        bio = self.bio_emb(bio) # [b, bio_length, bio_dim]
        # print(bio.size())
        bio_lengths = bio_lengths.cpu().numpy()
 
        bio = nn.utils.rnn.pack_padded_sequence(
                        bio, bio_lengths, batch_first=True)

        self.rnn.flatten_parameters()
        bio, hidden = self.rnn(bio)
        bio, _ = nn.utils.rnn.pad_packed_sequence(
            bio, batch_first=True)
        
        # last hidden [b, bio_dim]
        bio_scoring = self.bio_scoring(hidden[-1,:,:])
        # bio_scoring = torch.tanh(self.bio_scoring(hidden[-1,:,:]))
        return bio_scoring


class bioEncoderConv(nn.Module):
    def __init__(self, d_args, device) -> None:
        super(bioEncoderConv, self).__init__()

        self.device=device       
        self.conv = cnns2s.Encoder(d_args['n_bios'],
                                   d_args['bio_dim'],
                                   d_args['bio_hid'],
                                   d_args['n_layers'],
                                   device=device
                                   )
        self.bio_scoring = nn.Linear(in_features = d_args['bio_dim'],
			out_features = d_args['bio_out'],bias=True)
        
    def forward(self, bio, bio_lengths):
        
        bio, _ = self.conv(bio)
        # print(bio.size())
        bio = bio[:,-1,:]
        bio_scoring = self.bio_scoring(bio)
        # print(bio_scoring.size())
        return bio_scoring


class bioEncoderTransformer(nn.Module):
    def __init__(self, d_args, device):
        super(bioEncoderTransformer, self).__init__()

        self.device=device
        self.bio_dim = d_args['bio_dim']
        self.bio_embedding = nn.Embedding(d_args['n_bios'], d_args['bio_dim'])
        nn.init.normal_(self.bio_embedding.weight, 0.0, d_args['bio_dim']**-0.5)

        self.encoder = transformer.Encoder(
                        d_args['bio_dim'],
                        d_args['pf_dim'],
                        d_args['n_heads'],
                        d_args['n_layers'],
                        )
        # self.bio_scoring = nn.Linear(in_features = d_args['bio_rnn'],
        #         out_features = d_args['nb_fc_node'],bias=True)
        self.bio_scoring= nn.Conv1d(d_args['bio_dim'], d_args['bio_out'], 1)
    
    def forward(self, bio, bio_lengths):
        bio = self.bio_embedding(bio) * math.sqrt(self.bio_dim) # [b, bio_lengths, bio_dim]
        bio = torch.transpose(bio, 1, -1) # [b, bio_dim, bio_lengths]
        bio_mask = torch.unsqueeze(commons.sequence_mask(bio_lengths, bio.size(2)), 1).to(bio.dtype)

        bio = self.encoder(bio * bio_mask, bio_mask) # [b, bio_dim, bio_lengths]

        bio_scoring = self.bio_scoring(bio) * bio_mask

        return bio_scoring[:,:,-1] # [b, nb_fc_node]
        # return bio_scoring # for gru


class bioEncoderTransformersmall(nn.Module):
    def __init__(self, d_args, device):
        super(bioEncoderTransformersmall, self).__init__()

        self.device=device
        self.bio_dim = d_args['bio_dim']
        self.bio_embedding = nn.Embedding(d_args['n_bios'], d_args['bio_dim'])
        nn.init.normal_(self.bio_embedding.weight, 0.0, d_args['bio_dim']**-0.5)

        self.encoder = transformer.Encoder(
                        d_args['bio_dim'],
                        d_args['pf_dim'],
                        d_args['n_heads'],
                        d_args['n_layers'],
                        )
        # self.bio_scoring = nn.Linear(in_features = d_args['bio_rnn'],
        #         out_features = d_args['nb_fc_node'],bias=True)
        self.bio_scoring= nn.Conv1d(d_args['bio_dim'], d_args['bio_out'], 1)
    
    def forward(self, bio, bio_lengths):
        bio = self.bio_embedding(bio) * math.sqrt(self.bio_dim) # [b, bio_lengths, bio_dim]
        bio = torch.transpose(bio, 1, -1) # [b, bio_dim, bio_lengths]
        bio_mask = torch.unsqueeze(commons.sequence_mask(bio_lengths, bio.size(2)), 1).to(bio.dtype)

        bio = self.encoder(bio * bio_mask, bio_mask) # [b, bio_dim, bio_lengths]

        bio_scoring = self.bio_scoring(bio) * bio_mask

        return bio_scoring[:,:,-1] # [b, bio_out]


class bioEncoderlight(nn.Module):
    def __init__(self, d_args, device):
        super(bioEncoderlight, self).__init__()

        self.device=device
        # self.bio_dim = d_args['bio_dim']
        self.bio_embedding = nn.Embedding(d_args['n_bios'], d_args['bio_dim'])


        self.conv1 = nn.Conv1d(d_args['bio_dim'], 256, 1)
        self.conv2 = nn.Conv1d(256, 512, 1)
        
        self.fc1 = nn.Conv1d(512, d_args['nb_fc_node'], 1)
        # self.fc2 = nn.Linear(d_args['nb_fc_node'],d_args['nb_fc_node'], bias=True)
        
    
    def forward(self, bio, bio_lengths):
        bio = self.bio_embedding(bio) # [b, len, bio_dim]
        bio = torch.transpose(bio, 1, -1) # [b, bio_dim, len]
        bio = self.conv1(bio)
        bio = self.conv2(bio)
        bio = self.fc1(bio)
        # bio = self.fc2(bio)
        # bio_scoring = torch.transpose(bio_scoring, 1, -1) # [b, len, bio_dim]
        # print (bio.size())
        return bio[:,:,-1] # [b, nb_fc_node]

############################
# Countermeasure Model
############################ 
        
class Model(nn.Module):
    def __init__(self, args, device, is_train = True):
        super(Model, self).__init__()
        self.device=device
        self.is_add = args['is_add']
        self.backend = backend(args, self.device)
        self.is_train = True
        self.loss_type = args['loss_type']
        self.Wav2bioCNN = Wav2bioCNN(device=self.device)
        
        #PHUCDT
        # self.bioScoring = bioEncoderConv(args, self.device)
        # self.bioScoring = bioEncoderRNN(args, self.device) #add
        # self.bioScoring = bioEncoderTransformer(args, device) #add
        # self.bioScoring = bioEncoderlight(args, self.device)
        # self.bioScoring = bioEncoderRNNsmall(args, self.device) #concat
        self.bioScoring = bioEncoderTransformersmall(args, self.device) #concat
        
        if self.is_add:
            # ADD
            self.fc1 = nn.Linear(in_features = self.backend.out_dim,
                             out_features = args['bio_out'],bias=True)
            self.fc2 = nn.Linear(in_features = self.backend.out_dim,
                            out_features = args['nb_classes'],bias=True)
        else:
            # Concat
            self.fc2 = nn.Linear(in_features = self.backend.out_dim + args['bio_out'],
                out_features = args['nb_classes'],bias=True)
        

        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def get_Bio(self, X_pad, fs):
        X_pad = X_pad.cpu().numpy()
        bio = []
        bio_length = []
        for i in range(X_pad.shape[0]):
            bio.append(self.Wav2bioCNN.wav2bio(X_pad[i], fs, self.device))
            bio_length.append(len(bio[-1]))
        
        # stack to tensor
        bio = torch.IntTensor(bio)
        bio_length = torch.IntTensor(bio_length)
        bio = bio.to(self.device)
        bio_length = bio_length.to(self.device)
        return bio, bio_length
    
    def forward(self, x, bio = None, bio_lengths = None, y = None):
        if (bio is None):
            bio, bio_lengths = self.get_Bio(x, 16000)
        out , x, ssl_feat = self.backend(x)
        
        # #PHUCDT
        if (bio is not None):
            bio_scoring = self.bioScoring(bio, bio_lengths)
            if (self.is_add):
                x = self.fc1(x)
                x = x + bio_scoring # add the conditioning bio scoring
            else:
                x = torch.cat((x, bio_scoring), 1)

        b=x
        x = self.fc2(x)
        
        output=self.logsoftmax(x)
        if (self.is_train):
            return output, ssl_feat, b
        else:
            return output
        # return out, x
    
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