import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd 
from collections import Counter
import math
#lots of loss functions implementation

def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    
    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights_orig=weights.clone() # a tensor of size [1, number of classes]
    

    # parts similar to original implementation but still little different 
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot # generate for each batch element the corresponding class weight where the class index=1
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)
    
    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)  ## focal loss part fix it 
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        #pred = logits.softmax(dim = 1)
        #cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
        criterion=nn.CrossEntropyLoss(weight=weights_orig)
        cb_loss=criterion(logits,labels)

    return cb_loss

def cross_entropy_weighted(samples_per_cls,no_of_classes,beta,weight_option="square_root"):

      if(weight_option=="square_root"):
        weights=[0]*no_of_classes
        for id,sample in enumerate(samples_per_cls):
          weights[id]=(1.0/math.sqrt(samples_per_cls[id]))*100
        weights=np.array(weights)

      elif(weight_option=="effective_num_samples"):
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes

      weights = torch.tensor(weights).float() #a tensor of [1,num_classes]
      print(weights)

      criterion=nn.CrossEntropyLoss(weight=weights)

      return(criterion)

#label smoothing for ground truth in the cross entropy loss function
#reference issue: https://github.com/pytorch/pytorch/issues/7455
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):

        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim
    
    def forward(self, pred, target):
      pred = pred.log_softmax(dim=self.dim)
      with torch.no_grad():
          true_dist = torch.zeros_like(pred)
          true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) #(1-smoothing)*GT
          true_dist += self.smoothing / pred.size(self.dim) #(1-smoothing)*GT + smoothing * predicted 
      return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
      
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)

def binary_cross_entropy_loss(device,pos_weights=None,reduction='mean'):
  loss=nn.BCEWithLogitsLoss(reduction='mean',pos_weight=pos_weights).to(device)
  return(loss)
