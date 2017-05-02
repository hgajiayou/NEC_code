import random
import torch
import numpy as np
from sklearn.neighbors import KDTree
Num_action=5
Mem_size=5
actchosen_count=np.zeros(Num_action,dtype=int)
learning_rate=0.8


class DND(object):
    """Numpy implementation of Differentiable Neural Dictionary. Use Numpy structured array"""
    def __init__(self, mem_size, num_action):
        # construct dnd for different actions
        self.mem_size=mem_size
        self.num_action=num_action
        self.sigma=0.001
        self.size=str(self.num_action)+'float32'
        self.keys=np.zeros((self.num_action,self.mem_size),dtype=[('key',self.size),('value','float32')])
        # kd-tree with key to perform knn search
        self.trees = [KDTree(self.keys[i]['key'], leaf_size=40) for i in range(num_action)]

    def write(self, action_index,key_value,Q_value):
        """
            Append to memory if no key matched, update memory if key matches.
        """
        global actchosen_count,learning_rate
        dist, _ind = self.trees[action_index].query(key_value.reshape(1, -1), k=1)
        if dist[0][0] == 0:
        # if key_value in self.keys[action_index]['key']:
            Qi=self.keys[action_index]['value'][_ind[0][0]]
            Qi=Qi+learning_rate*(Q_value-Qi)
            self.keys[action_index]['value'][_ind[0][0]]=Qi
        else:
            self.keys[action_index]['key'][actchosen_count[action_index]]=key_value
            self.keys[action_index]['value'][actchosen_count[action_index]]=Q_value
            # self.trees = [KDTree(self.keys[i]['key'], leaf_size=40) for i in range(self.num_action)]
            self.trees[action_index] = KDTree(self.keys[action_index]['key'], leaf_size=40)
            actchosen_count[action_index]+=1
#calcurate weight wi, output Q-values of each action as in (1) in the article
    def cal_Qvalues(self,query_key,p):
        Q_values=np.zeros(self.num_action)
        for i in range(self.num_action):
            dist, _ind=self.trees[i].query(query_key.reshape(1,-1), k=p)
            dist_temp=1/np.matrix(dist+self.sigma)
            dist_sum=dist_temp.sum()
            weight=dist_temp/dist_sum
            values=self.keys[i]['value'][_ind].T
            Q_values[i]=float(weight*values)
        return torch.Tensor(Q_values)

#test

dnd=DND(mem_size=Mem_size,num_action=Num_action)
bb=np.ones((1,5))
aa=bb+3
cc=bb+1
kk=bb+2
actchosen_count=np.zeros(Num_action,dtype=int)
action_set = [0, 1, 0, 0, 1]
# print("dnd_before",np.shape(dnd.keys))
dnd.write(0, aa, 5)
dnd.write(0,cc,3)
chosen_action=dnd.cal_Qvalues(cc,2)
action_ind=chosen_action.max(0)[1].cpu()
action_val=chosen_action.max(0)[0].cpu()
# actchosen_count[chosen_action]+=1
print(action_val)
dnd.write(0,cc,5)
chosen_action=dnd.cal_Qvalues(cc,2)
action_ind=chosen_action.max(0)[1].cpu()
action_val=chosen_action.max(0)[0].cpu()
# actchosen_count[chosen_action]+=1
print(action_val)
dnd.write(1,cc,5)
print(dnd.keys[1]['key'])
dnd.write(1,bb,5)
print(dnd.keys[1]['key'])
dnd.write(1,kk,5)
print(dnd.keys[1]['key'],dnd.keys[1]['value'])
dnd.write(1,bb,10)
print(dnd.keys[1]['key'],dnd.keys[1]['value'])

