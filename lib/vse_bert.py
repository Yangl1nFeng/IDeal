"""VSE model"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_

# from lib.my_encoder import get_image_encoder, get_text_encoder
from lib.pos_encoder import get_image_encoder, get_text_encoder
from lib.loss import ContrastiveLoss, CCL, Transfer_loss

import logging

logger = logging.getLogger(__name__)


class VSEModel(object):
    """
        The standard VSE model
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = get_image_encoder(opt.img_dim, opt.embed_size,
                                         no_imgnorm=opt.no_imgnorm)
        self.txt_enc = get_text_encoder(opt, use_bi_gru=False, no_txtnorm=opt.no_txtnorm)    # if use the bert set "use_bi_gru=False"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.img_enc.to(self.device)
            self.txt_enc.to(self.device)
            cudnn.benchmark = True

            print("Image Encoder is on:", next(self.img_enc.parameters()).device)
            print("Text Encoder is on:", next(self.txt_enc.parameters()).device)
 
        # Loss and Optimizer
        # self.criterion = ContrastiveLoss(opt=opt,
        #                                  margin=opt.margin,
        #                                  max_violation=opt.max_violation)
        
        if opt.is_transfering:
            print("Using transfering loss for interaction adaptation tuning.")
            self.criterion = Transfer_loss(opt=opt)
        else:
            print("Using CCL loss for vanilla training.")
            self.criterion = CCL(opt=opt)
        
        # self.criterion = UCCH_ContrastiveLoss()

        # 获取 txt_enc.bert 的参数
        bert_params = list(self.txt_enc.bert.parameters())

        # 获取 txt_enc 和 img_enc 的其他参数
        all_params = list(self.txt_enc.parameters()) + list(self.img_enc.parameters())

        # 过滤掉 bert_params，确保不重复
        bert_params_set = set(bert_params)
        other_params = [p for p in all_params if p not in bert_params_set]

        # 记录所有优化参数
        self.params = bert_params + other_params  # 确保和 optimizer 对齐

        # 设置优化器
        self.optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': 0.01 * opt.learning_rate, 'weight_decay': 0},  # BERT 层通常不适用 weight decay
            {'params': other_params, 'lr': opt.learning_rate, 'weight_decay': 0.01}
        ])

        self.Eiters = 0
        self.data_parallel = False

    def set_max_violation(self, max_violation):
        # if max_violation:
        #     self.criterion.max_violation_on()
        # else:
        #     self.criterion.max_violation_off()
        return

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1], strict=False)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.data_parallel = True
        logger.info('Image encoder is data paralleled now.')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    def forward_emb(self, images, captions, lengths, image_lengths=None, pos=None):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = images.to(self.device)
        captions = captions.to(self.device)
        image_lengths = image_lengths.to(self.device)
        pos = pos.to(self.device)
        img_emb1,img_emb2 = self.img_enc(images, image_lengths, pos)

        lengths = torch.Tensor(lengths).to(self.device)
        cap_emb1, cap_emb2 = self.txt_enc(captions, lengths, pos)
        return img_emb1, img_emb2, cap_emb1, cap_emb2

    def forward_loss(self, img_emb1,img_emb2, cap_emb1,cap_emb2):
        """Compute the loss given pairs of image and caption embeddings
        """
        # 扩展的文本接近原来的文本中心【加上一个memory bank？】 直接接近对应图像，因为这个就相当于文本的中心
        # 扩展的文本远离扩展的负样本 和 原来的负样本 【现在的方法】可以把距离换成ICLR论文里面的距离然后再计算

        # raw
        # loss = self.criterion(img_emb, cap_emb)

        # RCL
        sim = cap_emb2.mm(cap_emb2.t())
        sim2 = img_emb2.mm(cap_emb2.t())
        loss = self.criterion(sim,sim2)

        # UCCH
        # loss = self.criterion(img_emb, cap_emb)

        self.logger.update('Le', loss.item())
        return loss

    def train_emb(self, images, captions, caption_lengths, image_lengths=None, warmup_alpha=None, pos=None):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        images_all = images  #.reshape(images.size(0)*images.size(1),images.size(2),images.size(3))
        image_lens = image_lengths.reshape(-1)
        captions_all = captions.reshape(captions.size(0),captions.size(-1))
        caption_lens = caption_lengths.reshape(-1)
       
        # compute the embeddings
        img_emb1,img_emb2, cap_emb1,cap_emb2 = self.forward_emb(images_all, captions_all, caption_lens, image_lengths=image_lens, pos=pos)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb1,img_emb2, cap_emb1,cap_emb2)

        if warmup_alpha is not None:
            loss = loss * warmup_alpha #linear lr warmup

        # compute gradient and update
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

