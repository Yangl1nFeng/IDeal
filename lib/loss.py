import torch
import torch.nn as nn
from torch.autograd import Variable

class UCCH_ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.2, shift=2., measure=False, max_violation=False):
        super(UCCH_ContrastiveLoss, self).__init__()
        self.margin = margin
        self.shift = shift
        
        self.sim = lambda x, y: x.mm(y.t())

        self.max_violation = max_violation
        self.count = 1

    def set_margin(self, margin):
        self.margin = margin

    def loss_func(self, cost, tau):
        cost = (cost - cost.diag().reshape([-1, 1])).exp()
        I = (cost.diag().diag() == 0)
        return cost[I].sum() / (cost.shape[0] * (cost.shape[0] - 1))

    def forward(self, im, s=None, tau=0.2, lab=None):
        if s is None:
            scores = im
            diagonal = im[:, 0].view(im.size(0), 1)
            d1 = diagonal.expand_as(scores)

            # compare every diagonal score to scores in its column
            # caption retrieval
            cost = (self.margin + scores - d1).clamp(min=0)
            # keep the maximum violating negative for each query
            if self.max_violation:
                cost = cost.max(1)[0]

            return cost.sum()

        else:
            # compute image-sentence score matrix
            scores = self.sim(im, s)
            self.count += 1
            
            diagonal = scores.diag().view(im.size(0), 1)
            d1 = diagonal.expand_as(scores)
            d2 = diagonal.t().expand_as(scores)
            mask_s = (scores >= (d1 - self.margin)).float().detach()
            cost_s = scores * mask_s + (1. - mask_s) * (scores - self.shift)
            mask_im = (scores >= (d2 - self.margin)).float().detach()
            cost_im = scores * mask_im + (1. - mask_im) * (scores - self.shift)
            loss = (-cost_s.diag() + tau * (cost_s / tau).exp().sum(1).log() + self.margin).mean() + (-cost_im.diag() + tau * (cost_im / tau).exp().sum(0).log() + self.margin).mean()
            return loss

class CCL(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, tau=0.15, method='log', q=0.8, ratio=0):
        super(CCL, self).__init__()
        self.opt = opt
        self.tau = tau 
        self.method = method
        self.q = q
        self.ratio = ratio
        self.mask_2 = [
                        [0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
                        [0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
                        [1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
                        [0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
                        [1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
                        [0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
                        [1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
                        [0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1.],
                        [1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0.],
                        [0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1.],
                        [1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0.],
                        [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1.],
                        [1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0.],
                        [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1.],
                        [1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0.],
                        [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0.]
                        ]
        self.mask_3 = [
                        [0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [1., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [1., 1., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [1., 1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [1., 1., 1., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [1., 1., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 1., 0., 1., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 1., 1., 1.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 1., 1., 1.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 1., 1.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 1.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.]
                        ]
        self.mask_2 = torch.tensor(self.mask_2).cuda()

    def forward(self, scores1, scores2):
        
        eps = 1e-10
        scores1 = (scores1 / self.tau).exp()
        i2t = scores1 / (scores1.sum(1, keepdim=True) + eps)
        t2i = scores1.t() / (scores1.t().sum(1, keepdim=True) + eps)

        scores2 = (scores2 / self.tau).exp()
        i2t_ = scores2 / (scores2.sum(1, keepdim=True) + eps)
        t2i_ = scores2.t() / (scores2.t().sum(1, keepdim=True) + eps)
    
        randn, eye = torch.rand_like(scores2), torch.eye(scores2.shape[0]).cuda()
        randn[eye > 0] = randn.min(dim=1)[0] - 1  # 对角线接近-1
        n = scores2.shape[0]
        num = n - 1 if self.ratio <= 0 or self.ratio >= 1 else int(self.ratio * n)
        V, K = randn.topk(num, dim=1)
        mask = torch.zeros_like(scores2)    # 和raw loss的形状是一样的
        mask[torch.arange(n).reshape([-1, 1]).cuda(), K] = 1.

        # mm_a = (torch.arange(scores.size(0))//self.opt.hardnum +1) * self.opt.hardnum
        # mask_a = torch.arange(n).view(n,1).expand_as(scores)
        # mask1 = (mask_a<mm_a.long())
        # # mask = torch.mul((mask1 * mask1.t()).long().float(), 0.8)
        # mask =(mask1 * mask1.t()).long().float()
        # mask = torch.sub(1,mask).cuda()

        # mask_2 = torch.sub(1.,mask).cuda()
        # mask_2 = mask_2 - torch.eye(mask.shape[0]).cuda()
        
        # if torch.cuda.is_available():
        #     I = mask.cuda()
        if self.method == 'log':
            alpha = 5
            # criterion = lambda x: -((1. - x + eps).log() * mask).sum(1).sum(0) / 12 # - 0.5 * (((x + eps).log() * mask_2).sum(1).sum(0) / 2)  #  - ((x + eps).log() * mask_2).sum(1).mean()
            # criterion = lambda x: (-alpha * torch.mul(torch.sub(1,x)**(1/alpha), torch.log(1-x)) * mask).sum(1).sum(0) / (scores.size(0)-self.opt.hardnum) # - 0.2 * (5 * torch.mul(x**0.2, torch.log(x)) * mask_2).sum(1).sum(0) / (self.opt.hardnum)
            # criterion = lambda x: (-1 * torch.log(1-x) * mask).sum(1).sum(0) / (scores.size(0)-self.opt.hardnum)
            # criterion = lambda x: (-alpha * torch.mul(torch.sub(1,x)**(1/alpha), torch.log(1-x)) * self.mask_2).sum(1).sum(0) # / ((scores.size(0)-self.opt.hardnum)/2) # / (scores.size(0)-self.opt.hardnum)
            criterion = lambda x: (-alpha * torch.mul(torch.sub(1,x)**(1/alpha), torch.log(1-x)) * torch.sub(1,torch.eye(5).cuda())).sum(1).sum(0) / (scores1.size(0)-1)
            # criterion = lambda x: (-torch.mul(torch.sub(1,x)**(1), torch.log(1-x)) * torch.sub(1,torch.eye(8).cuda())).sum(1).sum(0) / (scores1.size(0)-1)
        return  criterion(i2t_) + criterion(t2i_)

class Transfer_loss(nn.Module):
    def __init__(self, opt, tau=0.15, method='log', q=0.5, ratio=0, t=0.05, alpha=5):
        super(Transfer_loss, self).__init__()
        self.opt = opt
        self.tau = tau  # 温度参数
        self.method = method
        self.q = q
        self.ratio = ratio
        self.t = 0.05  # 相似度阈值
        self.alpha = alpha  # 衰减速率参数

    def forward(self, scores, scores1):
        eps = 1e-10

        scores = (scores / self.tau).exp()
        i2t_ = scores / (scores.sum(1, keepdim=True) + eps)
        t2i_ = scores.t() / (scores.t().sum(1, keepdim=True) + eps)

        scores1 = (scores1 / self.tau).exp()
        i2t = scores1 / (scores1.sum(1, keepdim=True) + eps)
        t2i = scores1.t() / (scores1.t().sum(1, keepdim=True) + eps)

        randn, eye = torch.rand_like(scores), torch.eye(scores.shape[0]).cuda()
        randn[eye > 0] = randn.min(dim=1)[0] - 1  # 将对角线的值接近 -1
        n = scores.shape[0]
        num = n - 1 if self.ratio <= 0 or self.ratio >= 1 else int(self.ratio * n)
        V, K = randn.topk(num, dim=1)
        mask = torch.zeros_like(scores)  # 创建一个形状与损失相同的mask
        mask[torch.arange(n).reshape([-1, 1]).cuda(), K] = 1.

        pos_mask = torch.eye(n).cuda()  # 仅正样本位置为1

        if self.method == 'log':
            def weight_function(similarity):
                return torch.where(
                    similarity <= self.t,
                    torch.tensor(1.0).cuda(),  # similarity <= t, 权重为 1
                    torch.exp(-self.alpha * (similarity - self.t))  # similarity > t, 使用指数衰减
                )

            # 计算每个相似度的加权值
            similarity_weight_i2t = weight_function(i2t_)
            similarity_weight_t2i = weight_function(t2i_)

            criterion = lambda x, y, similarity_weight: -(((1. - y + eps).log() * mask * similarity_weight).sum(1).sum()) - 0.2 * (x * pos_mask).sum(1).log().sum()
            # criterion = lambda x, similarity_weight: ((1. - (1. - x + eps) ** self.q * similarity_weight) / self.q * mask).sum(1).mean() + 1 * (((1. - x + eps) ** self.q * pos_mask) / self.q * mask).sum(1).mean()

        return criterion(i2t, i2t_, similarity_weight_i2t) + criterion(t2i, t2i_, similarity_weight_t2i)  # criterion(i2t, similarity_weight_i2t) + criterion(t2i, similarity_weight_t2i) # criterion(i2t, i2t_, similarity_weight_i2t) + criterion(t2i, t2i_, similarity_weight_t2i)

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = get_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X