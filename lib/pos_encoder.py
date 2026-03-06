"""VSE modules"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel
from lib.modules.resnet import ResnetFeatureExtractor
from lib.modules.mlp import MLP
from lib.vocab import deserialize_vocab
import logging

import json
import os

logger = logging.getLogger(__name__)

class SharedConv1d(nn.Module):
    def __init__(self, kernel_size):
        super(SharedConv1d, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size)  
    
    def forward(self, x):

        batch_size, in_channels, length = x.shape
        x = x.reshape(batch_size * in_channels, 1, length)
        x = self.conv(x)
        _, _, new_length = x.shape
        x = x.reshape(batch_size, in_channels, new_length)
        
        return x
    
def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


def get_text_encoder(opt, use_bi_gru=True, no_txtnorm=False):
    if use_bi_gru==True:
        return EncoderText(opt, use_bi_gru=use_bi_gru,no_txtnorm=no_txtnorm)
    else:
        print('Using Bert!')
        return EncoderText_Bert(opt.embed_size, no_txtnorm=no_txtnorm)

def get_image_encoder(img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    img_enc = EncoderImageAggr(img_dim, embed_size, precomp_enc_type, no_imgnorm)
    return img_enc

def get_positional_embeddings(positions, embed_size,gamma=1024):
    """
    positions: [batch_size, grid_num, 2]
    embed_size: dimension of the embeddings
    """
    batch_size, grid_num, _ = positions.shape
    position_embeddings = torch.zeros((batch_size, grid_num, embed_size), device=positions.device)
    
    div_term = torch.exp(torch.arange(0, embed_size, 2, device=positions.device).float() * 
                         (-np.log(10000.0) / embed_size)).unsqueeze(0).unsqueeze(0)  # Shape [1, 1, embed_size//2]
    
    # Expand positions to match the shape of div_term
    x_pos = positions[:, :, 0].unsqueeze(-1)*gamma  # Shape [batchsize, grid_num, 1]
    y_pos = positions[:, :, 1].unsqueeze(-1)*gamma  # Shape [batchsize, grid_num, 1]
    
    x_pos = x_pos.repeat(1, 1, embed_size // 2)  # Shape [batchsize, grid_num, embed_size//2]
    y_pos = y_pos.repeat(1, 1, embed_size // 2)  # Shape [batchsize, grid_num, embed_size//2]

    x_pos_emb = torch.zeros((batch_size, grid_num, embed_size), device=positions.device)
    y_pos_emb = torch.zeros((batch_size, grid_num, embed_size), device=positions.device)
    
    x_pos_emb[:, :, 0::2] = torch.sin(x_pos * div_term)  # Shape [batchsize, grid_num, embed_size]
    x_pos_emb[:, :, 1::2] = torch.cos(x_pos * div_term)  # Shape [batchsize, grid_num, embed_size]
    y_pos_emb[:, :, 0::2] = torch.sin(y_pos * div_term)  # Shape [batchsize, grid_num, embed_size]
    y_pos_emb[:, :, 1::2] = torch.cos(y_pos * div_term)  # Shape [batchsize, grid_num, embed_size]
    
    position_embeddings = (x_pos_emb + y_pos_emb) / 2   # (x_pos_emb + y_pos_emb) / 2
    
    return position_embeddings


class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.img_dim = img_dim
        self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        if precomp_enc_type == 'basic':
            self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        
        self.theta = 0.05
        self.linear1 = SharedConv1d(kernel_size=1024)
        self.linear2 = nn.Linear(embed_size, embed_size)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image, image_lengths, pos):
        """Extract image feature vectors."""
        features = self.fc(image)

        if self.precomp_enc_type == 'basic':
            # When using pre-extracted region features, add an extra MLP for embedding transformation
            xy_pos = pos[:, :, :2]  # [batchsize, grid_num, 2]
            pos_embeddings = get_positional_embeddings(xy_pos, self.embed_size)  # [batchsize, grid_num, embed_size]

            features = self.mlp(image) + features
            pos_feat = features # + pos_embeddings

        if self.training:
            features_external = self.linear1(pos_feat)
            features_external = torch.div(features_external,1024)
            features_external_ = self.linear2(features)
            features_external1 = features_external
            features_k_softmax1 = nn.Softmax(dim=1)(features_external1)
            features_k_softmax1_ = nn.Softmax(dim=1)(features_external_)

            feature_img_1 = torch.sum( features_k_softmax1 *features_k_softmax1_ * features,dim=1)
            feature_img_2 = torch.sum( features_k_softmax1 * features_k_softmax1_ * features,dim=1)
        else:
            img_emb= features

            features_in = self.linear1(pos_feat)
            features_in = torch.div(features_in,1024)
            attn = nn.Softmax(dim=1)(features_in)

            features_in_ = self.linear2(features)
            attn_ = nn.Softmax(dim=1)(features_in_)

            feature_img_1 = torch.sum(attn* attn_ * img_emb,dim=1)
            feature_img_2 = torch.sum(attn*  attn_ * img_emb,dim=1)

        if not self.no_imgnorm:
            feature_img_1 = l2norm(feature_img_1, dim=-1)
            feature_img_2 = l2norm(feature_img_2, dim=-1)

        return feature_img_1, feature_img_2

def load_glove_embeddings(glove_path, vocab, embedding_dim=300):

    embeddings_index = {}

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    vocab_size = len(vocab)
    embedding_matrix = np.random.uniform(-0.05, 0.05, (vocab_size, embedding_dim)) 

    for word, idx in vocab.word2idx.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]

    return torch.tensor(embedding_matrix, dtype=torch.float32)

# Language Model with BiGRU
class EncoderText(nn.Module):
    def __init__(self, opt, use_bi_gru=True, no_txtnorm=False,is_init=True):
        super(EncoderText, self).__init__()
        self.embed_size = opt.embed_size
        self.no_txtnorm = no_txtnorm

        self.theta = 0.05

        vocab_file = 'all_vocab.json'
        vocab = deserialize_vocab(os.path.join(opt.vocab_path, vocab_file))
        vocab.add_word('<mask>')  # add the mask, for testing cloze

        embedding_matrix = load_glove_embeddings('./lib/vocab/glove.6B/glove.6B.300d.txt', vocab, opt.word_dim)
        self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)

        # caption embedding
        self.rnn = nn.GRU(opt.word_dim, opt.embed_size, opt.num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.linear1 = SharedConv1d(kernel_size=1024)
        self.linear2 = nn.Linear(opt.embed_size, opt.embed_size)
        
        if is_init:
            self.ts = 500. * torch.ones(1)
        else:
            self.ts = 500. * torch.ones(1)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, lengths, pos):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x_emb = self.embed(x)

        lengths = lengths.clamp(min=1) #
        
        self.rnn.flatten_parameters()
        #x_emb_rnn = x_emb[indices]

        packed = pack_padded_sequence(x_emb, self.ts, batch_first=True, enforce_sorted=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        cap_emb_rnn, cap_len = pad_packed_sequence(out, batch_first=True)
        cap_emb = cap_emb_rnn
        
        cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] + cap_emb[:, :, cap_emb.size(2) // 2:]) / 2


        max_len = 500
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        
        cap_emb = cap_emb[:, :500, :] 

        if self.training:
            cap_external = self.linear1(cap_emb)
            cap_external = torch.div(cap_external,1024)
            cap_external_ = self.linear2(cap_emb)
            
            cap_external = cap_external.masked_fill(mask == 0,-10000)
            cap_external_ = cap_external_.masked_fill(mask == 0,-10000)

            #attn
            attn = nn.Softmax(dim=1)(cap_external)
            attn_ = nn.Softmax(dim=1)(cap_external_)

            attn = attn.masked_fill(mask == 0,0)
            attn_ = attn_.masked_fill(mask == 0,0)

            feature_cap = attn * attn_ * cap_emb
            feature_cap = torch.sum(feature_cap,dim=1)

            feature_cap_2 = attn_ * cap_emb
            feature_cap_2 = torch.sum(feature_cap_2,dim=1)

        else:
            cap_external = self.linear1(cap_emb)
            cap_external = torch.div(cap_external,1024)
            cap_external = cap_external.masked_fill(mask == 0,-10000)
            attn = nn.Softmax(dim=1)(cap_external)
            attn = attn.masked_fill(mask == 0,0)
            
            
            cap_external_ = self.linear2(cap_emb)
            cap_external_ = cap_external_.masked_fill(mask == 0,-10000)
            attn_ = nn.Softmax(dim=1)(cap_external_)
            attn_ = attn_.masked_fill(mask == 0,0)

            feature_cap = attn_ * cap_emb
            feature_cap = torch.sum(feature_cap,dim=1)
            feature_cap_2 = attn * attn_ * cap_emb
            feature_cap_2 = torch.sum(feature_cap_2,dim=1)
        

        if not self.no_txtnorm:
            feature_cap = l2norm(feature_cap, dim=-1) 
            feature_cap_2 = l2norm(feature_cap_2, dim=-1) 

        return feature_cap, feature_cap_2 

class EncoderText_Bert(nn.Module):
    def __init__(self, embed_size, no_txtnorm=False):
        super(EncoderText_Bert, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        bert_path = '../vsepp-python3/'
        self.bert = BertModel.from_pretrained(bert_path)
        self.linear = nn.Linear(768, embed_size)
        self.linear1 = SharedConv1d(kernel_size=1024)
        self.linear2 = nn.Linear(embed_size, embed_size)

    def forward(self, x, lengths, pos):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        with torch.no_grad():
            bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D
    
        cap_emb = self.linear(bert_emb)
        # cap_emb = self.dropout(cap_emb)

        max_len = 500
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        
        cap_emb = cap_emb[:, :500, :] 

        if self.training:
            cap_external = self.linear1(cap_emb)
            cap_external = torch.div(cap_external, 1024)
            cap_external_ = self.linear2(cap_emb)
            
            cap_external = cap_external.masked_fill(mask == 0,-10000)
            cap_external_ = cap_external_.masked_fill(mask == 0,-10000)

            attn = nn.Softmax(dim=1)(cap_external)
            attn_ = nn.Softmax(dim=1)(cap_external_)
            
            attn = attn.masked_fill(mask == 0,0)
            attn_ = attn_.masked_fill(mask == 0,0)

            feature_cap = attn *  attn_ * cap_emb
            feature_cap = torch.sum(feature_cap,dim=1)
            # feature_cap = torch.sum(attn * cap_emb,dim=1)  # [bs*2, 512]

            feature_cap_2 = attn * attn_ * cap_emb
            feature_cap_2 = torch.sum(feature_cap_2,dim=1)
            
        else:
            cap_external = self.linear1(cap_emb)
            cap_external = torch.div(cap_external, 1024)
            cap_external = cap_external.masked_fill(mask == 0,-10000)
            attn = nn.Softmax(dim=1)(cap_external)
            attn = attn.masked_fill(mask == 0,0)
            
            cap_external_ = self.linear2(cap_emb)
            cap_external_ = cap_external_.masked_fill(mask == 0,-10000)
            attn_ = nn.Softmax(dim=1)(cap_external_)
            attn_ = attn_.masked_fill(mask == 0,0)
            feature_cap = attn * attn_ * cap_emb
            feature_cap = torch.sum(feature_cap,dim=1)
            feature_cap_2 = attn * attn_ * cap_emb
            feature_cap_2 = torch.sum(feature_cap_2,dim=1)
            
        if not self.no_txtnorm:
            feature_cap = l2norm(feature_cap, dim=-1) 
            feature_cap_2 = l2norm(feature_cap_2, dim=-1) 

        return feature_cap, feature_cap_2 

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()