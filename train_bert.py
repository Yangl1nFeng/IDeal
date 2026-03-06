"""Training script"""
import os
import time
import numpy as np
import torch

from lib.vocab import deserialize_vocab
from lib.datasets import i_c_bert
from lib.vse_bert import VSEModel
from lib.evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, compute_sim

import logging
# import tensorboard_logger as tb_logger
from transformers import BertTokenizer
import arguments


def main():
    # Hyper Parameters
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()

    if not os.path.exists(opt.model_name):
        os.makedirs(opt.model_name)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info(opt)

    # Load Vocabulary
    bert_path = opt.bert_path 
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    train_loader, val_loader = i_c_bert.get_loaders(
        opt.data_path, opt.data_name, tokenizer, opt.batch_size, opt.workers, opt)
    
    
    if opt.is_transfering:
        ''' raw training '''
        model_path = opt.original_ckpt # "/home/fengyanglin/ESA-main/ESA_BIGRU/runs/8_9/source.pth" # "/home/fengyanglin/ESA-main/ESA_BIGRU/runs/nr3d_bert/model_best_bert.pth" # "/home/fengyanglin/ESA-main/ESA_BIGRU/runs/model_best_bert.pth"
        checkpoint = torch.load(model_path)
        model = VSEModel(opt)
        model.load_state_dict(checkpoint['model'])
    else:
        ''' domain adaptation '''
        model = VSEModel(opt)
    
    lr_schedules = [opt.lr_update, ]

    start_epoch = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            model.Eiters = checkpoint['Eiters']
            logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                        .format(opt.resume, start_epoch, best_rsum))
            if opt.reset_start_epoch:
                start_epoch = 0
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0
    rsum, rsum2 = validate(opt, val_loader, model)
    for epoch in range(start_epoch, opt.num_epochs):
        logger.info(opt.logger_name)
        logger.info(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch, lr_schedules)

        if epoch >= opt.vse_mean_warmup_epochs:
            opt.max_violation = True
            model.set_max_violation(opt.max_violation)
        
        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        rsum, rsum2 = validate(opt, val_loader, model)

        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_{}.pth'.format(epoch), prefix=opt.model_name + '/')
    # f.close()
    # g.close()

def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    logger.info('image encoder trainable parameters: {}'.format(count_params(model.img_enc)))
    logger.info('txt encoder trainable parameters: {}'.format(count_params(model.txt_enc)))

    num_loader_iter = len(train_loader.dataset) // train_loader.batch_size + 1

    end = time.time()
    # opt.viz = True
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        images, img_lengths, captions, lengths, _, pos,_,_,_ = train_data

        model.train_emb(images, captions, lengths, image_lengths=img_lengths,pos=pos)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger.info log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader.dataset) // train_loader.batch_size + 1, batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))


def validate(opt, val_loader, model):
    logger = logging.getLogger(__name__)
    model.val_start()
    start = time.time()
    with torch.no_grad():
        # compute the encoding for all the validation images and captions
        img_embs, cap_embs,_, _,_ = encode_data(
            model, val_loader, opt.log_step, logging.info)
    
    ''' If evaluate 3DLLM dataset, please set data_div at 2, otherwise 10. '''
    data_div = 10
    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), data_div)])

    sims = compute_sim(img_embs[:,1,:], cap_embs[:,1,:])
    end = time.time()
    logger.info("calculate similarity time:".format(end - start))


    # caption retrieval
    npts = img_embs.shape[0]
    # (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
    (r1, r5, r10, medr, meanr) = i2t(npts, sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    # (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, cap_lens, sims)
    (r1i, r5i, r10i, medri, meanr) = t2i(npts, sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    logger.info('Current rsum is {}'.format(currscore))

    # record metrics in tensorboard 

    return r1 + r5 + r10, r1i + r5i + r10i


def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):
    logger = logging.getLogger(__name__)
    tries = 15

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix + 'model_best.pth')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch, lr_schedules):
    logger = logging.getLogger(__name__)
    """Sets the learning rate to the initial LR
       decayed by 10 every opt.lr_update epochs"""
    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * 0.1
            param_group['lr'] = new_lr
            logger.info('new lr {}'.format(new_lr))


def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


if __name__ == '__main__':
    main()
