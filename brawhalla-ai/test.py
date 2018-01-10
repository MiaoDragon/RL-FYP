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
state_feature_batch = FloatTensor([2,0,1,0,0,0,0,0,0,0]).unsqueeze(1)
inner_product = state_feature_batch.matmul(state_feature_batch.transpose(1,0))
state_feature_batch_l2 = (state_feature_batch ** 2).sum(dim=1,keepdim=True).expand_as(inner_product)
distance_matrix = state_feature_batch_l2 + state_feature_batch_l2.transpose(1,0) - 2 * inner_product
print(distance_matrix)

# calculate Q value ditance matrix
# Here use target value to calculate
target_batch = FloatTensor([1,0,1,0,1,1,1,1,1,1]).unsqueeze(1)
Q_dist_matrix = target_batch.expand_as(distance_matrix)
Q_dist_matrix = Q_dist_matrix - Q_dist_matrix.transpose(1,0) # not absolute value
Q_dist_matrix = Q_dist_matrix.abs()
action_batch = LongTensor([1,1,1,1,1,1,1,1,1,1]).unsqueeze(1)

# Number[i,j] = Number[i,j] + (D_f[i,j] <= sample_S^2 AND D_Q[i,j] <= sample_Q AND action[i]=action[j])
# only consider same actions
Action_Mask = (action_batch.expand_as(distance_matrix)) == (action_batch.transpose(1,0).expand_as(distance_matrix))

Mask = (distance_matrix <= (2)) & (Q_dist_matrix <= 2) & Action_Mask
Mask = Mask.type(FloatTensor)
print(Mask)

Number = Mask.sum(dim=1,keepdim=True)
print(Number)

# using the mask to calculate the number used for each transition
probability_batch = Mask.matmul(1. / Number) / Number
print(probability_batch)

probability_batch.squeeze(1)
