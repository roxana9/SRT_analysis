#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 22:22:18 2022

@author: roxana
"""
import pandas as pd
import numpy as np
import os
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
import anndata
#from scipy.sparse import issparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors 
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
def augmentation(adata,edge_index,aug_type,ratio=None):
    """
    graph augmentation methods.
    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    
    edge_index : Torch tensor
        a torch tensor contains the index of edges
    aug_type : string, optional
        Graph augmentation type of input spatial data. Available methods are: random_shuffling, node_dropping,
        EdgePerturbation, Diffusion,RWSample, NodeAttrMask.
    ratio: ratio (float, optional):
        use this parameter for edgeperterbution
        Percentage of edges to add or drop. (default: :obj:`0.1`)
        The number of input ST data. 'single' means single ST sample; 'multiple' means multiple ST samples (i.e., ST data integration task)      
    Returns
    -------
    augmented features.
    """
    feat=get_feature(adata)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(feat, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.contiguous())
    if aug_type== 'random_shuffling':
        feat_aug=random_shuffling(feat)
        adata.obsm['feat_aug_random_shuffling'] = feat_aug
    elif aug_type== 'node_dropping':
        feat_aug=node_dropping(data,adata)
    elif aug_type== 'EdgePerturbation':
        feat_aug=EdgePerturbation(data,adata)
        #adata.obsm['feat_aug_NodeAttrMask'] = feat_aug
    elif aug_type== 'Diffusion':
        feat_aug=Diffusion(adata,data)
    elif aug_type== 'RWSample':
        feat_aug=RWSample(adata,data,ratio=0.2)
    elif aug_type== 'NodeAttrMask':
        feat_aug=NodeAttrMask(feat)
        adata.obsm['feat_aug_NodeAttrMask'] = feat_aug
    return feat_aug
def get_feature(adata):
      
    adata_Vars =  adata[:, adata.var['highly_variable']]
       
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    adata.obsm['feat'] = feat
    return feat
     
def random_shuffling(feature):
    # fix_seed(FLAGS.random_seed) 
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    
    return feature_permutated    
def NodeAttrMask(data,mode='onehot',mask_ratio=0.1, mask_mean=0.5, mask_std=0.5):
    """
    Node attribute masking on the given graph. 
    
    Args:
        mode (string, optinal): Masking mode with three options:
            :obj:`"whole"`: mask all feature dimensions of the selected node with a Gaussian distribution;
            :obj:`"partial"`: mask only selected feature dimensions with a Gaussian distribution;
            :obj:`"onehot"`: mask all feature dimensions of the selected node with a one-hot vector.
            (default: :obj:`"whole"`)
        mask_ratio (float, optinal): The ratio of node attributes to be masked. (default: :obj:`0.1`)
        mask_mean (float, optional): Mean of the Gaussian distribution to generate masking values.
            (default: :obj:`0.5`)
        mask_std (float, optional): Standard deviation of the distribution to generate masking values. 
            Must be non-negative. (default: :obj:`0.5`)
    """
    # fix_seed(FLAGS.random_seed) 
    node_num, feat_dim = np.shape(data)
    x = data

    if mode == "whole":
        print('whole')
        mask = torch.zeros(node_num)
        mask_num = int(node_num * mask_ratio)
        idx_mask = np.random.choice(node_num, mask_num, replace=False)
        x[idx_mask] = torch.tensor(np.random.normal(loc=mask_mean, scale=mask_std, 
                                                    size=(mask_num, feat_dim)), dtype=torch.float32)
        mask[idx_mask] = 1
        #print(mask)

    elif mode == "partial":
        print('partial')
        mask = torch.zeros((node_num, feat_dim))
        for i in range(node_num):
            for j in range(feat_dim):
                if random.random() < mask_ratio:
                    x[i][j] = torch.tensor(np.random.normal(loc=mask_mean, 
                                                            scale=mask_std), dtype=torch.float32)
                    mask[i][j] = 1

    elif mode == "onehot":
        print('onehot')
        mask = torch.zeros(node_num)
        mask_num = int(node_num * mask_ratio)
        idx_mask = np.random.choice(node_num, mask_num, replace=False)
        x[idx_mask] = torch.tensor(np.eye(feat_dim)[np.random.randint(0, feat_dim, size=(mask_num))], dtype=torch.float32)
        mask[idx_mask] = 1

    else:
        raise Exception("Masking mode option '{0:s}' is not available!".format(mode))

    
    return x  
        
def EdgePerturbation(data,adata,add=False, drop=True, ratio=0.1):
    """
    Edge perturbation on the given graph
    
    Args:
        add (bool, optional): Set :obj:`True` if randomly add edges in a given graph.
            (default: :obj:`True`)
        drop (bool, optional): Set :obj:`True` if randomly drop edges in a given graph.
            (default: :obj:`False`)
        ratio (float, optional): Percentage of edges to add or drop. (default: :obj:`0.1`)
    """
    print('EdgePerturbation: drop')
    node_num, _ = data.x.size()
    _,edge_num = data.edge_index.size()
    perturb_num = int(edge_num *ratio)
    
    edge_index = data.edge_index.detach().clone()
    idx_remain = edge_index
    idx_add = torch.tensor([]).reshape(2, -1).long()
    
    if drop:
        idx_remain = edge_index[:, np.random.choice(edge_num, edge_num-perturb_num, replace=False)]
    
    if add:
        idx_add = torch.randint(node_num, (2, perturb_num))
        
    new_edge_index = torch.cat((idx_remain, idx_add), dim=1)
    new_edge_index = torch.unique(new_edge_index, dim=1)
    idx=np.array(new_edge_index)
    adj=np.zeros((node_num,node_num)) 
    adj[idx[0],idx[1]]=1
    adata.obsm['adj_EdgePerturbation'] = adj
    return adj

def node_dropping(data,adata,ratio=0.1):
    """
    Uniformly node dropping on the given graph
    
    Args:
        data: a torch tensor contains the graph
        adata: an Anndata object
        ratio (float, optinal): Ratio of nodes to be dropped. (default: :obj:`0.1`)
    return a new anndata object and torch tensor of the new graph 
    """
    
        
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    
    keep_num = int(node_num * (1-ratio))
    idx_nondrop = torch.randperm(node_num)[:keep_num]
    mask_nondrop = torch.zeros_like(data.x[:,0]).scatter_(0, idx_nondrop, 1.0).bool()
    
    edge_index, _ = subgraph(mask_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)
    idx=np.array(edge_index)
    adj=np.zeros((keep_num,keep_num)) 
    adj[idx[0],idx[1]]=1
    x=np.array(data.x[mask_nondrop]) 
    adata2=anndata.AnnData(x)
    adata2.obsm['adj_node_dropping'] = adj
    return adata2,Data(x=data.x[mask_nondrop], edge_index=edge_index)       
        
def Diffusion(adata,data,mode='heat',alpha=0.2,t=5,add_self_loop=False):
    """
    Diffusion on the given graph, used in 
    `MVGRL <https://arxiv.org/pdf/2006.05582v1.pdf>`_. 
    Args:
        mode (string, optional): Diffusion instantiation mode with two options:
            :obj:`"ppr"`: Personalized PageRank; :obj:`"heat"`: heat kernel.
            (default: :obj:`"ppr"`)
        alpha (float, optinal): Teleport probability in a random walk. (default: :obj:`0.2`)
        t (float, optinal): Diffusion time. (default: :obj:`5`)
        add_self_loop (bool, optional): Set True to add self-loop to edge_index.
            (default: :obj:`False`)
    """

    
    node_num, _ = data.x.size()
    
    if add_self_loop:
        sl = torch.tensor([[n, n] for n in range(node_num)]).t()
        edge_index = torch.cat((data.edge_index, sl), dim=1)
    else:
        edge_index = data.edge_index.detach().clone()
    
    orig_adj = to_dense_adj(edge_index)[0]
    orig_adj = torch.where(orig_adj>1, torch.ones_like(orig_adj), orig_adj)
    d = torch.diag(torch.sum(orig_adj, 1))

    if mode == "ppr":
        print('ppr')
        dinv = torch.inverse(torch.sqrt(d))
        at = torch.matmul(torch.matmul(dinv, orig_adj), dinv)
        diff_adj = alpha * torch.inverse((torch.eye(orig_adj.shape[0]) - (1 - alpha) * at))

    elif mode == "heat":
        print('heat')
        diff_adj = torch.exp(t * (torch.matmul(orig_adj, torch.inverse(d)) - 1))

    else:
        raise Exception("Must choose one diffusion instantiation mode from 'ppr' and 'heat'!")

    edge_ind, edge_attr = dense_to_sparse(diff_adj)
    idx=np.array(edge_ind)
    adj=np.zeros((node_num,node_num)) 
    adj[idx[0],idx[1]]=1
    adata.obsm['adj_diffusion'] = adj
    return  Data(x=data.x, edge_index=edge_ind, edge_attr=edge_attr)
        
def RWSample(adata,data,ratio=0.1):
    """
    Subgraph sampling based on random walk on the given graph.

    
    Args:
        ratio (float, optional): Percentage of nodes to sample from the graph.
            (default: :obj:`0.1`)
    """

    
    node_num, _ = data.x.size()
    sub_num = int(node_num * ratio)

    
    edge_index = data.edge_index.detach().clone()

    # edge_index = edge_index.numpy()
    idx_sub = [np.random.randint(node_num, size=1)[0]]
    # idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])
    idx_neigh = set([n.item() for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        # idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))
        idx_neigh.union(set([n.item() for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_sub = torch.LongTensor(idx_sub).to(data.x.device)
    mask_nondrop = torch.zeros_like(data.x[:,0]).scatter_(0, idx_sub, 1.0).bool()
    edge_index, _= subgraph(mask_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)
    node_num, _ = data.x[mask_nondrop].size()
    idx=np.array(edge_index)
    adj=np.zeros((node_num,node_num)) 
    adj[idx[0],idx[1]]=1
    sub=np.array(data.x[mask_nondrop])
    sub_graph=anndata.AnnData(sub)
    sub_graph.obsm['adj_node_dropping'] = adj
    return sub_graph,Data(x=data.x[mask_nondrop], edge_index=edge_index)

        
        
        
        
        
        