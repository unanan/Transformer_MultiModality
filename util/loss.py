import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self,generator, reduction='mean', ignore_index=553):
        super(CrossEntropyLoss, self).__init__(reduction=reduction,ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.generator = generator

    def forward(self, out, target, mask, norm=1):
        assert target.shape == mask.shape
        out = self.generator(out)
        
        target = target.masked_fill(mask=mask==0, value=self.ignore_index)

        # print(target)
        if len(target.shape)==2:  #TODO
            # out = torch.reshape(out,(out.size(0)*out.size(1), out.size(-1))).squeeze()
            out = torch.reshape(out,(-1, out.size(-1))).squeeze()
            # target = torch.reshape(target,(target.size(0)*target.size(1), )).squeeze()
            target = torch.reshape(target,(-1,)).squeeze()

        # print(F.log_softmax(x, 1).shape, target.shape)
        # print(x[:2], target[:2])
        # loss = self.criterion(x, target)

        loss = super(CrossEntropyLoss, self).forward(out, target)/norm
        return loss


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.criterion = nn.KLDivLoss()
        
    def forward(self, x, target, mask):
        pass


class FocalLoss_BCE_2d(nn.Module):
    def __init__(self, gamma=1.2, alpha=0.25, scale_factor=500.0, size_average=False):
        super(FocalLoss_BCE_2d, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.scale_factor = scale_factor
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
    
    def forward(self, input, target, mask):
        # if input.dim()>2:
        #     # input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        #     input = input.view(-1,input.size(2),input.size(3))
        #     # input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        #     # input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        # target = target.view(-1,target.size(2),target.size(3))
        # print(target.shape)
        samples_num, _, _ = target.shape
        error = 1 - torch.abs(input - target) + 1e-20  # 防止logp变得无限小
        log_error = torch.log(error)
        loss = -self.scale_factor * (1 - error) ** self.gamma * log_error
        loss *= mask
        # print(f"input:{input}, target:{target}, error:{error}, log_error:{log_error}, loss:{loss}, sum:{loss.sum()}")
        if self.size_average:
            return loss.mean()
        else:
            # print(loss.sum(),error,log_error,np.sum(loss.detach().numpy()>0.1)+1)
            # print(np.maximum(loss.detach().numpy(), 0.01))
            return loss.sum() / samples_num  # /(np.sum(loss.detach().numpy()>0.01)+1)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, generator, smoothing=0.01, ignore_index=-100, size_average=False):
        super(LabelSmoothingLoss, self).__init__()
        assert 0 <= smoothing <= 1
        self.smoothing = smoothing
        self.criterion = nn.KLDivLoss(size_average=size_average)
        self.generator = generator
        self.ignore_index = ignore_index
        self.confidence = 1 - smoothing
        self.true_dist = None
    
    def forward(self, out, target, mask, norm=1):
        out = self.generator(out)
        
        target = target.masked_fill(mask=mask == 0, value=self.ignore_index)

        out = out.contiguous().view(-1, out.size(-1))
        target = target.contiguous().view(-1)
        
        # print(out, target)
        
        true_dist = out.data.clone()
        true_dist.fill_(self.smoothing / (out.size(1) - 2))  # fill 0
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # true_dist[:,  self.ignore_index] = 0  #:,
        true_dist = true_dist.masked_fill(mask=true_dist == self.ignore_index, value=0)
        self.true_dist = true_dist
        
        # loss = self.criterion(out.contiguous().view(-1, out.size(-1)), target.contiguous().view(-1)) / norm
        # print(out.dtype,target.dtype)
        loss = self.criterion(out, true_dist) / norm
        # print(loss.item())
        return loss

#====================================================== CritWrapper ====================================================
#
# class CritWrapper:
#     "Loss wrapper."
#
#     def __init__(self, generator, crit="crossentropy"):
#         self.generator = generator
#         self.ignore_index = 553
#         # self.criterion = CrossEntropyLoss(ignore_index=self.ignore_index) #TODO
#         self.criterion = LabelSmoothingLoss(padding_idx=self.ignore_index, start_smoothing=0.1)
#
#     def __call__(self, out, y, loss_mask, norm=1, loss_calc = True):
#         assert y.shape == loss_mask.shape
#
#         out = self.generator(out)
#         # y = y.masked_fill(mask=loss_mask == 0, value=553)
#         # crit = nn.CrossEntropyLoss()#ignore_index=self.ignore_index)
#         # if len(y.shape)==2:  #TODO
#         #     out = torch.reshape(out,(out.size(0)*out.size(1), out.size(-1))).squeeze()
#         #     y = torch.reshape(y,(y.size(0)*y.size(1), )).squeeze()
#         # criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.ignore_index)
#         # y = y.masked_fill(mask=loss_mask == 0, value=self.ignore_index)
#
#         # if len(y.shape) == 2:
#         #     out = torch.reshape(out, (out.size(0) * out.size(1), out.size(-1))).squeeze()
#         #     y = torch.reshape(y, (y.size(0) * y.size(1),)).squeeze()
#         # loss = criterion(out, y)
#         # loss = crit(out, y)
#
#         loss = self.criterion(out, y, loss_mask)/ 1#norm
#
#         print(loss.item())
#         # loss = self.crossentropy(out, y, loss_mask) / norm
#
#
#         return loss