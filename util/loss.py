import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = -100
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
    
    def forward(self, x, target, mask):
        target.masked_fill(mask=(mask == 0), value=self.ignore_index)
        loss = self.criterion(x, target)
        
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
    "Implement label smoothing."
    
    def __init__(self, padding_idx, start_smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        # self.criterion = nn.KLDivLoss(size_average=False)
        self.criterion = FocalLoss_BCE_2d(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - start_smoothing
        self.smoothing = start_smoothing
        
        self.true_dist = None
        # self.loss = 10000.0
    
    def forward(self, x, target, mask):
        # assert x.size(1) == size, print(x.size(1), size)
        
        # target = target.unsqueeze(-1)
        # print(target)
        true_dist = x.data.clone()
        size = target.shape[-1]
        true_dist.fill_(self.smoothing / (size - 2))  # fill 0
        # logging.info(f"smoothing:{self.smoothing}\t\tfill:{self.smoothing / (size - 2)}")
        # true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # print(true_dist.shape, target.shape)
        true_dist.scatter_(2, target.unsqueeze(-1), self.confidence)
        
        true_dist[:, :, self.padding_idx] = 0
        
        # print(true_dist.shape, mask.shape)
        mask = mask.unsqueeze(-1).expand(true_dist.shape)
        # print(true_dist.shape,mask.shape)
        # true_dist[:,self.padding_idx] = 0
        # mask = torch.nonzero(target == self.padding_idx)
        # if mask.dim() > 0:
        #     # print(mask.shape)
        #     # true_dist.index_fill_(2, mask.squeeze(), 0.0) # orginal
        #     # true_dist[torch.arange(true_dist.size(1)).unsqueeze(1),mask] = 0.0 #test failed
        #     # true_dist[torch.arange(true_dist.size(1)).unsqueeze(1),mask.unsqueeze(-1)] = 0.0 #test failed
        #     for ind in mask:
        #         true_dist[ind[0],ind[1],:] = 0.0
        #
        #     # true_dist[target == self.padding_idx] = 0.0
        #     # for i, ind in enumerate(mask):
        #     #     true_dist[i].index_fill_(0,ind,1)
        
        # if mask.dim() > 0:
        #     true_dist.index_fill_(0, mask.squeeze(), 0.0)
        
        self.true_dist = true_dist
        # print(x,true_dist)
        return self.criterion(x, true_dist, mask)  # Default:, requires_grad=False


#====================================================== CritWrapper ====================================================

class CritWrapper:
    "Loss wrapper."
    
    def __init__(self, generator, crit=CrossEntropyLoss()):
        self.generator = generator
        self.criterion = crit
    
    def __call__(self, out, y, loss_mask, norm=1):
        assert y.shape == loss_mask.shape
        
        out = self.generator(out)
        loss = self.criterion(out, y, loss_mask) / norm
        
        return loss