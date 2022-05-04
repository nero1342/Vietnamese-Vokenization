from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses

from torch import nn 
import torch 

class LossComputer(nn.Module):
    def __init__(self, cfg):
        self.loss_func = losses.TripletMarginLoss(distance = CosineSimilarity(), 
				     reducer = ThresholdReducer(high=0.3), 
			 	     embedding_regularizer = LpRegularizer())

    def forward(self, embeddings):
        v_e, l_e = embeddings
        labels = torch.arange(v_e.size(0))
        data = torch.cat([v_e, l_e], dim=0)
        labels = torch.cat([labels, labels], dim=0)

        return self.loss_func(data, labels)

