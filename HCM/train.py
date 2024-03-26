import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
# from utils import load_data, accuracy, randomwalk_fea
from preprocessing.preprocessing import load_data, accuracy
from modeldefine.models import GAT

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# writer = SummaryWriter(log_dir='/home/sjf/recoda-experiment/pyGAT-origin/to/RP-gat')
train_record = []
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=76, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8,help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--version',type=int,help='training version log')
parser.add_argument('--ex',type=str,help='experiment control')
parser.add_argument('--dataset',type=str,help='experiment datasets')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

record = str(args.ex)+": COLLAB: Seed: "+str(args.seed)+" Epochs: "+str(args.epochs) \
            +" lr: "+str(args.lr)+" weight_decay: "+str(args.weight_decay)+" hidden: "+str(args.hidden) \
            +" nb_head: "+str(args.nb_heads)+" drrop:0.5 walk_length: 8 walknum: 10 window: 8" \
            +" beta: 0.8"
train_record.append(record)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# def randomset(adj, adj_2, prob1,prob2,window):
#     if prob1 <0. or prob1 >=1:
#         raise ValueError("prob must be in [0,1)")
#     if prob2 <0. or prob2 >=1:
#         raise ValueError("prob must be in [0,1)")
        
#     for i in range(adj_2.size(0)):
#         if len(torch.nonzero(adj_2[i])) > window:
#             choce = random.choices(torch.nonzero(adj_2[i]),k=len(torch.nonzero(adj_2[i])) - window)
#             for j in choce:
#                 adj_2[i][j] = 0.

#     retain_p1 = 1 - prob1
#     retain_p2 = 1 - prob2
#     random_tensor1 = np.random.binomial(n=1, p=retain_p1, size=adj.shape)
#     random_tensor2 = np.random.binomial(n=1, p=retain_p2, size=adj_2.shape)
#     adj = torch.mul(adj,torch.tensor(random_tensor1).cuda())
#     adj_2 = torch.mul(adj_2, torch.tensor(random_tensor2).cuda())
#     # print(f"adj shape:{adj.shape}, adj_2 shape:{adj_2.shape}")
#     adj = adj + adj_2
#     return adj


# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)

if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    # adj2 = adj2.cuda()
    # randomadj = randomadj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)

# identity, ranadj = randomwalk_fea(walk_length=10,num_walks=30)
# beta = 0.01
# window = 8
# ranadj = ranadj.cuda()
# 0.9 - 0.5
# randomadj = randomset(adj, ranadj, beta, 0.8, window)

# Model and optimizer
model = GAT(nfeat=features.shape[1], 
            nhid=args.hidden, 
            nclass=int(labels.max()) + 1,
            adj=adj,
            dropout=args.dropout, 
            nheads=args.nb_heads, 
            alpha=args.alpha)

optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

model.cuda()
def train(epoch,adj):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    # print('labels shape origin gat',labels[idx_train])
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    # writer.add_scalar('accuracy/train',acc_train,epoch)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    # writer.add_scalar('accuracy/val',acc_val,epoch)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    record = 'Epoch: '+str(epoch+1)+' loss_train: '+str(loss_train)+' acc_train: '+str(acc_train)+' loss_val: '+str(loss_val)+' acc_val: '+str(acc_val)+' time '+str(time.time()-t)
    train_record.append(record)
    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))
    record = 'loss_test: '+str(loss_test)+' accuracy: '+str(acc_test)
    train_record.append(record)

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch,adj))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()
file = '/root/experiment/citeseerex/records/records_{}'.format(args.version)+'_{}.txt'.format(args.dataset)
try:
    with open(file, 'x') as f:
        for item in train_record:
            f.write('{}\n'.format(item))
    f.close()
    print('Successfully saved to {}.'.format(file))
# except FileExistsError:
#     print('File already exists, skipping.')
except Exception as e:
    print('An error occurred: {}'.format(str(e)))
