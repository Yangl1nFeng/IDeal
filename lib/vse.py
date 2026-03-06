"""VSE model"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_

# from lib.my_encoder import get_image_encoder, get_text_encoder
from lib.pos_encoder import get_image_encoder, get_text_encoder
from lib.loss import ContrastiveLoss, CCL, UCCH_ContrastiveLoss

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
        self.txt_enc = get_text_encoder(opt, use_bi_gru=True, no_txtnorm=opt.no_txtnorm)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True
 
        # Loss and Optimizer
        # self.criterion = ContrastiveLoss(opt=opt,
        #                                  margin=opt.margin,
        #                                  max_violation=opt.max_violation)
        self.criterion = CCL(opt=opt)
        # self.criterion = UCCH_ContrastiveLoss()

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params
        self.opt = opt

        self.optimizer = torch.optim.AdamW(self.params, lr=opt.learning_rate)

        logger.info('Use {} as the optimizer, with init lr {}'.format(self.opt.optim, opt.learning_rate))

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
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            image_lengths = image_lengths.cuda()
            pos = pos.cuda()
        img_emb1,img_emb2 = self.img_enc(images, image_lengths, pos)

        lengths = torch.Tensor(lengths).cuda()
        cap_emb1, cap_emb2 = self.txt_enc(captions, lengths, pos)
        return img_emb1, img_emb2, cap_emb1, cap_emb2

    def forward_loss(self, img_emb1,img_emb2, cap_emb1,cap_emb2):
        """Compute the loss given pairs of image and caption embeddings
        """

        # raw
        # loss = self.criterion(img_emb, cap_emb)

        # RCL
        sim = cap_emb1.mm(cap_emb1.t())
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

