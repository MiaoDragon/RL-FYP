import torch
from torch.autograd import Variable
import time
import math
import random
# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
# calculate the probability for each transition
# calculate distance matrix
state_feature_batch = FloatTensor([1,0,1,0]).unsqueeze(1)
inner_product = state_feature_batch.matmul(state_feature_batch.transpose(1,0))
state_feature_batch_l2 = (state_feature_batch ** 2).sum(dim=1,keepdim=True).expand_as(inner_product)
distance_matrix = state_feature_batch_l2 + state_feature_batch_l2.transpose(1,0) - 2 * inner_product
print(distance_matrix)

# calculate Q value ditance matrix
# Here use target value to calculate
target_batch = FloatTensor([1,0,1,4]).unsqueeze(1)
Q_dist_matrix = target_batch.expand_as(distance_matrix)
Q_dist_matrix = Q_dist_matrix - Q_dist_matrix.transpose(1,0) # not absolute value
Q_dist_matrix = Q_dist_matrix.abs()
action_batch = LongTensor([1,1,1,1]).unsqueeze(1)

# Number[i,j] = Number[i,j] + (D_f[i,j] <= sample_S^2 AND D_Q[i,j] <= sample_Q AND action[i]=action[j])
# only consider same actions
Action_Mask = (action_batch.expand_as(distance_matrix)) == (action_batch.transpose(1,0).expand_as(distance_matrix))

Mask = (distance_matrix <= (2)) & (Q_dist_matrix <= 2) & Action_Mask
print(Mask)
Cluster = []
while True:
    # clustering by VC, always find largest degree
    Number = Mask.sum(dim=1)
    value, indx = Number.max(dim=0)
    if value[0] == 0:
        # already empty
        break
    v = Mask[indx]
    Cluster.append(v)
    # delete vertices
    Delete = 1-v.transpose(1,0).matmul(v)
    Mask = Mask & Delete
k = len(Cluster)
Cluster = torch.cat(Cluster)
print(Cluster)

Number = Cluster.sum(dim=1).type(LongTensor)
probability_batch = torch.ones(k) / float(k)
print(probability_batch)
cluster_is = torch.multinomial(probability_batch,2,replacement=True)
# convert the cluster indices to number of items in each cluster
print(cluster_is)
Sample_num = torch.eye(k).index_select(0,cluster_is).sum(dim=0).type(LongTensor)
print(Sample_num)
#N = Cluster[0].size()[0] # number of vertices
print(Number)

state_sample = []
action_sample = []
target_sample = []
for i in range(k):
    n = Sample_num[i]
    N = Number[i]
    if n == 0:
        continue
    cluster = Cluster[i]
    # get nonzero indices
    v_indices = cluster.nonzero().squeeze(1)
    if n == N:
        #print('true')
        # pick up all
        #state_sample.append(state_batch.index_select(0, v_indices))
        #action_sample.append(action_batch.index_select(0, v_indices))
        #target_sample.append(target_batch.index_select(0, v_indices))
        continue
    prob = torch.ones(v_indices.size()) / n
    if n < N:
        # uniformly pick
        v_indices_is = torch.multinomial(prob, n)
        v_indices = v_indices.index_select(0, v_indices_is)
        print(v_indices)
        #state_sample.append(state_batch.index_select(0, v_indices))
        #action_sample.append(action_batch.index_select(0, v_indices))
        #target_sample.append(target_batch.index_select(0, v_indices))
        continue
    # uniformly pick with replacement
    v_indices_is = torch.multinomial(prob, n, replacement=True)
    v_indices = v_indices.index_select(0, v_indices_is)
    #state_sample.append(state_batch.index_select(0, v_indices))
    #action_sample.append(action_batch.index_select(0, v_indices))
    #target_sample.append(target_batch.index_select(0, v_indices))
#state_batch = torch.stack(state_sample)
#action_batch = torch.stack(action_sample)
#target_batch = torch.stack(target_sample)
