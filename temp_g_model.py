#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 02:07:09 2022

@author: roxana
"""

import torch
from preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, add_contrastive_label, get_feature, permutation, fix_seed
import time
import random
import numpy as np
from model import Encoder, Encoder_sparse, Encoder_map, Encoder_sc
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
class temp_g_model():
    def __init__(self, adata, random_seed=50, add_regularization=True, device='cpu'):
       self.adata = adata.copy()
       self.random_seed = random_seed
       self.add_regularization = True
       self.device = device
       
       fix_seed(self.random_seed)
       preprocess(self.adata)
       construct_interaction(self.adata)
       add_contrastive_label(self.adata)
       
       self.adata_output = self.adata.copy()
    
    def train_D(self):
       if self.add_regularization:
          adata = self.adata_output.copy()
          #preprocess(adata)
          get_feature(adata)
          model = Train(adata, device=self.device)
          emb = model.train()
          self.adata_output.obsm['emb'] = emb
          
          fix_seed(self.random_seed)
          adata = self.adata_output.copy()
          preprocess(adata)
          get_feature(adata)
          model = Train(adata, add_regularization=True, device=self.device)
          emb_regularization = model.train()
          self.adata_output.obsm['emb_reg'] = emb_regularization
          
       else:
          model = Train(self.adata.copy())
          emb = model.train()
          self.adata_output.obsm['emb'] = emb
          
       return self.adata_output   
    
class Train():
    def __init__(self, 
            adata,
            adata_sc = None,
            device='cuda:0',
            learning_rate=0.001,
            learning_rate_sc = 0.01,
            weight_decay=0.00,
            epochs=300, 
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
        learning_rate_sc : float, optional
            Learning rate for scRNA representation learning. The default is 0.01.
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
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.add_regularization = add_regularization
        self.deconvolution = deconvolution
        self.datatype = datatype
        
        self.features = torch.FloatTensor(adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(adata.obsm['label_CSL']).to(self.device)
        self.adj = adata.obsm['adj']
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
        
        if self.deconvolution:
           self.adata_sc = adata_sc.copy() 
            
           if isinstance(self.adata.X, csc_matrix) or isinstance(self.adata.X, csr_matrix):
              self.feat_sp = adata.X.toarray()[:, ]
           else:
              self.feat_sp = adata.X[:, ]
           if isinstance(self.adata_sc.X, csc_matrix) or isinstance(self.adata_sc.X, csr_matrix):
              self.feat_sc = self.adata_sc.X.toarray()[:, ]
           else:
              self.feat_sc = self.adata_sc.X[:, ]
            
           # fill nan as 0
           self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
           self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values
          
           self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
           self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
        
           if self.adata_sc is not None:
              self.dim_input = self.feat_sc.shape[1] 

           self.n_cell = adata_sc.n_obs
           self.n_spot = adata.n_obs
            
    def train(self):
        if self.datatype in ['Stereo', 'Slide']:
           self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
           self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        
        if not self.add_regularization:
           print('Begin to train ST data...')
        self.model.train()
        
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()
              
            self.features_a = permutation(self.features)
            self.hiden_feat, self.emb, ret, ret_a = self.model(self.features, self.features_a, self.adj)
            
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
         
    
    def loss(self, emb_sp, emb_sc):
        '''\
        Calculate loss
        Parameters
        ----------
        emb_sp : torch tensor
            Spatial spot representation matrix.
        emb_sc : torch tensor
            scRNA cell representation matrix.
        Returns
        -------
        Loss values.
        '''
        # cell-to-spot
        map_probs = F.softmax(self.map_matrix, dim=1)   # dim=0: normalization by cell
        self.pred_sp = torch.matmul(map_probs.t(), emb_sc)
           
        loss_recon = F.mse_loss(self.pred_sp, emb_sp, reduction='mean')
        loss_NCE = self.Noise_Cross_Entropy(self.pred_sp, emb_sp)
           
        return loss_recon, loss_NCE
        
    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
        '''\
        Calculate noise cross entropy. Considering spatial neighbors as positive pairs for each spot
            
        Parameters
        ----------
        pred_sp : torch tensor
            Predicted spatial gene expression matrix.
        emb_sp : torch tensor
            Reconstructed spatial gene expression matrix.
        Returns
        -------
        loss : float
            Loss value.
        '''
        
        mat = self.cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        
        # positive pairs
        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(axis=1)
        
        ave = torch.div(p, k)
        loss = - torch.log(ave).mean()
        
        return loss
    
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