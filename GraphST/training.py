#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 06:44:15 2022

@author: roxana
"""

import torch
from preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, add_contrastive_label, get_feature, fix_seed
import time

from torch.utils.data import DataLoader
import random
import numpy as np
from model_test import Encoder, Encoder_sparse, Encoder_map, Encoder_sc
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
from pytorch_metric_learning import losses
import graph_aug as gu
class my_model():
    def __init__(self, adata, random_seed=50, add_regularization=True, device='cpu'):
       self.adata = adata.copy()
       self.random_seed = random_seed
       self.add_regularization = True
       self.device = device
       
       fix_seed(self.random_seed)
       #preprocess(self.adata)
       construct_interaction(self.adata)
       add_contrastive_label(self.adata)
       
       self.adata_output = self.adata.copy()
    
    def train_model(self):
       if self.add_regularization:
          adata = self.adata_output.copy()
          #preprocess(adata)
          get_feature(adata)
          model = Train(adata, device=self.device)
          emb = model.train()
          self.adata_output.obsm['emb'] = emb
          
          fix_seed(self.random_seed)
          adata = self.adata_output.copy()
          #preprocess(adata)
          get_feature(adata)
          model = Train(adata, add_regularization=True, device=self.device)
          emb_regularization = model.train()
          self.adata_output.obsm['emb_reg'] = emb_regularization
          
       else:
          model = Train(self.adata.copy())
          emb= model.train()
          self.adata_output.obsm['emb'] = emb
          
       return self.adata_output ,losses  
    
class Train():
    def __init__(self, 
            adata,
            adata_sc = None,
            device='cuda:0',
            learning_rate=0.01,
            weight_decay=0.00,
            epochs=600, 
            batch_size=128,
            dim_input=3000,
            dim_output=64,
            random_seed = 50,
            alpha = 10,
            beta = 1,
            theta = 0.1,
            lamda1 = 10,
            lamda2 = 1,
            add_regularization = False,
            deconvolution = False,
            datatype = '10X'
            ):
        '''\
        Parameters
        ----------
        adata : anndata
            AnnData object of spatial data.
        adata_sc : anndata, optional
            AnnData object of scRNA-seq data. adata_sc is needed for deconvolution. The default is None.
        device : string, optional
            Using GPU or CPU? The default is 'cuda:0'.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight factor to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 600.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 50.
        alpha : float, optional
            Weight factor to control the influence of reconstruction loss in representation learning. 
            The default is 10.
        beta : float, optional
            Weight factor to control the influence of contrastive loss in representation learning. 
            The default is 1.
        theta : float, optional
            Weight factor to control the influence of penalty term in representation learning. 
            The default is 0.1.
        lamda1 : float, optional
            Weight factor to control the influence of reconstruction loss in mapping matrix learning. 
            The default is 10.
        lamda2 : float, optional
            Weight factor to control the influence of contrastive loss in mapping matrix learning. 
            The default is 1.
        add_regularization : bool, optional
            Add penalty term in representation learning?. The default is False.
        deconvolution : bool, optional
            Deconvolution task? The default is False.
        datatype : string, optional    
            Data type of input. Our model supports 10X Visium ('10X'), Stereo-seq ('Stereo'), and Slide-seq/Slide-seqV2 ('Slide') data. 
        Returns
        -------
        The learned representation 'self.emb_rec'.
        '''
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.batch_size=batch_size
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.add_regularization = add_regularization
        self.deconvolution = deconvolution
        self.datatype = datatype
        self.edge_index=construct_interaction(self.adata)
        self.edge_index=torch.FloatTensor(self.edge_index).to(self.device)
        self.features = torch.FloatTensor(adata.obsm['feat'].copy()).to(self.device)
       # self.features_a = gu.augmentation(self.adata,self.edge_index,aug_type='NodeAttrMask')
        self.features_a =self.features
        #self.features_a = torch.FloatTensor(self.features_a).to(self.device)
        
        self.label_CSL=add_contrastive_label(self.adata)
        self.label_CSL = torch.FloatTensor(self.label_CSL).to(self.device)
        self.adj = adata.obsm['adj_EdgePerturbation']
       # self.adj=adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
        
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        
        if self.datatype in ['Stereo', 'Slide']:
           #using sparse
           print('Building sparse matrix ...')
           self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else: 
           # standard version
           self.adj = preprocess_adj(self.adj)
           self.adj = torch.FloatTensor(self.adj).to(self.device)
        
    def train(self):
        if self.datatype in ['Stereo', 'Slide']:
           self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
           self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        self.loss_func = losses.TripletMarginLoss()
        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        
        if not self.add_regularization:
           print('Begin to train ST data...')
        self.model.train()
        self.epoch_losses = []
        for epoch in range(self.epochs): 
            self.model.train()
             
            # epoch_loss = 0
            self.hiden_feat, self.emb, ret,ret_a = self.model(self.features, self.features_a, self.adj)
    #         self.labels = torch.ones(self.emb.size(0))
    #         self.labels2 = torch.zeros(self.emb.size(0))
    # #         self.labels = torch.cat([self.labels, self.labels2], dim=0)
    #         self.embeddings = torch.cat([ret,ret_a], dim=0)
    # #         dataset =torch.utils.data.TensorDataset(self.embeddings, self.labels)
    # #         data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # #         for i, (data, labels) in enumerate(data_loader):
    #         #loss = self.loss_CSL(self.embeddings ,self.label_CSL)
    #         loss.backward()
    #         self.optimizer.step()
               
    # #         epoch_loss += loss.detach().item()
    #         self.optimizer.zero_grad()
        #     loss.backward(retain_graph=True)
        # epoch_loss /= (i + 1)
        # print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        # epoch_losses.append(epoch_loss)
            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
           
            self.loss_feat = F.mse_loss(self.features, self.emb)
            
            if self.add_regularization:
               self.loss_norm = 0
               for name, parameters in self.model.named_parameters():
                   if name in ['weight1', 'weight2']:
                      self.loss_norm = self.loss_norm + torch.norm(parameters, p=2) 
               loss =  self.alpha*self.loss_feat + self.beta*(self.loss_sl_1 + self.loss_sl_2) + self.theta*self.loss_norm 
            else: 
               loss =  self.alpha*self.loss_feat + self.beta*(self.loss_sl_1 + self.loss_sl_2)
            print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if not self.add_regularization:
           print("Optimization finished for ST data!")
        
        with torch.no_grad():
             self.model.eval()
             if self.deconvolution:
                self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
             else:  
                if self.datatype in ['Stereo', 'Slide']:
                   self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                   self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy() 
                else:
                   self.emb_rec = self.model(self.features, self.features_a, self.adj)[1].detach().cpu().numpy()
             
             return self.emb_rec
         
    
    def cosine_similarity(self, pred_sp, emb_sp):  #pres_sp: spot x gene; emb_sp: spot x gene
        '''\
        Calculate cosine similarity based on predicted and reconstructed gene expression matrix.    
        '''
        
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)
        
        if torch.any(torch.isnan(M)):
           M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

        return M        