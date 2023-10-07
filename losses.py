import torch 
from torch import nn 
class CustomizedLoss(torch.nn.Module):
    def __init__(self):
        super(CustomizedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction = 'none')
   
    
    def forward(self, preds, gts, gamma, epoch, train= True):
        if not train:
            return torch.mean(self.ce_loss(preds, gts))
        else:
            ind_loss = self.ce_loss(preds, gts)
            sub_loss = ind_loss-max(ind_loss)
            p = torch.exp(sub_loss/gamma)
            p = p/sum(p)
            total_loss = sum(p*ind_loss)
            return total_loss 
