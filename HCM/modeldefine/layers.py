import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing.preprocessing import randomwalk_fea

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features,adj,dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        # self.adj2 = adj2
        # self.W = nn.Linear(in_features, out_features)
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # walk length 10-6, Get Hybrid Node Feature
        identity, ranadj = randomwalk_fea(adj,walk_length=2, num_walks=10)
        beta = 0.01
        window = 8
        ranadj = ranadj.cuda()
        # Hyper parameter
        self.randomadj = randomset(adj, ranadj, beta, 0.8, window)

    def forward(self, h, adj):
        # h = torch.cat([h,adj],dim=1)
        # A = self.adj2.cuda()
        # h = torch.cat([h,A],dim=1)
        # randomadj = randomset(adj, self.randomadj, self.beta, 1-self.beta, self.window)
        # A = self.randomadj
        # h = torch.cat([h,adj],dim=1)
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(self.randomadj > 0, e, zero_vec)
        # attention = e
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, 0.5, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class NodeClassification(nn.Module):
    def __init__(self, in_features, out_features, adj2,dropout, alpha, concat=True):
        super(NodeClassification, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.adj2 = adj2

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h = torch.cat([h,adj],dim=1)
        # A = self.adj2.cuda()
        # h = torch.cat([h,A],dim=1)
        # A = randomset(adj, A, 0, 0.6, 10)
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        # attention = e
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# Retain Information
import random
import numpy as np
def randomset(adj, adj_2, prob1,prob2,window):
    if prob1 <0. or prob1 >=1:
        raise ValueError("prob must be in [0,1)")
    if prob2 <0. or prob2 >=1:
        raise ValueError("prob must be in [0,1)")
        
    for i in range(adj_2.size(0)):
        if len(torch.nonzero(adj_2[i])) > window:
            choce = random.choices(torch.nonzero(adj_2[i]),k=len(torch.nonzero(adj_2[i])) - window)
            for j in choce:
                adj_2[i][j] = 0.

    retain_p1 = 1 - prob1
    retain_p2 = 1 - prob2
    random_tensor1 = np.random.binomial(n=1, p=retain_p1, size=adj.shape)
    random_tensor2 = np.random.binomial(n=1, p=retain_p2, size=adj_2.shape)
    adj = torch.mul(adj,torch.tensor(random_tensor1).cuda())
    adj_2 = torch.mul(adj_2, torch.tensor(random_tensor2).cuda())
    # print(f"adj shape:{adj.shape}, adj_2 shape:{adj_2.shape}")
    adj = adj + adj_2
    return adj
