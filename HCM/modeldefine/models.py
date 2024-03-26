import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, NodeClassification


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass,adj,dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        # self.madj = madj
        self.attentions = [GraphAttentionLayer(nfeat, nhid,adj,dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass,adj,dropout=dropout, alpha=alpha, concat=False)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # x = x+self.addi
        return F.log_softmax(x, dim=1)

