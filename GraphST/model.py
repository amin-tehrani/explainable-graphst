import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from . import similarity
from typing import Union, Literal, Callable

from .preprocess import permutation

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        ### Calculate the sum of neighbors' embeddings
        vsum = torch.mm(mask, emb)

        ### Calculate the num of neighbors for each node (denominator of the average)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        
        ### Calculate the average of neighbors' embeddings
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1) 

class BaseEncoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act='relu', is_sparse=False):
        super(BaseEncoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act_str = act
        self.is_sparse = is_sparse

        self.act = getattr(F, self.act_str)
        
        # Replace manual weight matrices with GCNConv layers
        self.gcn1 = GCNConv(self.in_features, self.out_features, bias=False)
        self.gcn2 = GCNConv(self.out_features, self.in_features, bias=False)
        
        
        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def get_args_dict(self): 
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "graph_neigh": self.graph_neigh,
            "dropout": self.dropout,
            "act": self.act_str,
            "is_sparse": self.is_sparse
        }

        
    def forward(self, feat, feat_a, adj):
        # Convert adjacency matrix to edge_index format for PyG
        edge_index, edge_weight = dense_to_sparse(adj)
        
        # Forward pass for original features
        z = F.dropout(feat, self.dropout, self.training)
        z = self.gcn1(z, edge_index, edge_weight)
        emb = self.act(z)
        
        # Forward pass for augmented features
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = self.gcn1(z_a, edge_index, edge_weight)
        emb_a = self.act(z_a)
        
        # Graph-level representations
        g = self.read(emb, self.graph_neigh) 
        g = self.sigm(g)  

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)  
        
        # Discriminator outputs
        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb)

        
        
        # Reconstruction path
        h = self.gcn2(z, edge_index, edge_weight) 
        
        return z, h, ret, ret_a, emb, emb_a
    
class GraphStEncoder(Module):
    def __init__(self, base_encoder: Module):
        super(GraphStEncoder, self).__init__()
        self.base_encoder = base_encoder
    
    def forward(self, feat, feat_a, adj):
        hiden_emb, h, ret, ret_a, _, _ = self.base_encoder(feat, feat_a, adj)
        return hiden_emb, h, ret, ret_a


class ExplainableEncoder(Module):
    def __init__(self, base_encoder: Module, feat_a, sim_func = None):
        super(ExplainableEncoder, self).__init__()
        self.base_encoder = base_encoder
        self.feat_a = feat_a

        if sim_func is not None:
            self.similarity_fn = sim_func
        else:
            self.similarity_fn = self.default_similarity_fn

    def default_similarity_fn(self, hiden_emb, h, ret, ret_a, emb, emb_a):
        return torch.exp(-torch.norm(hiden_emb - h, p=2))

    def forward(self, x, edge_index, edge_weight=None):
        # Convert edge_index to dense adjacency matrix
        adj = torch.zeros((x.size(0), x.size(0)), device=x.device)
        if edge_weight is not None:
            adj[edge_index[0], edge_index[1]] = edge_weight
        else:
            adj[edge_index[0], edge_index[1]] = 1.0
        
        # print(x.shape)
        feat = x
        feat_a = self.feat_a
        hiden_emb, h, ret, ret_a, emb, emb_a = self.base_encoder(feat, feat_a, adj)
        # print(ret.shape, ret_a.shape)

        similarity = self.similarity_fn(hiden_emb, h, ret, ret_a, emb, emb_a)

        return similarity        

class Encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)
        
        hiden_emb = z
        
        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)
        
        emb = self.act(z)
        
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        emb_a = self.act(z_a)
        
        g = self.read(emb, self.graph_neigh) 
        g = self.sigm(g)  

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)  

        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb) 
        
        return hiden_emb, h, ret, ret_a
    
# class GCNConv(nn.Module):
#     def __init__(self, in_features, out_features, dropout=0.0, act=F.relu, is_sparse=False):
#         super(GCNConv, self).__init__()
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         self.dropout = dropout
#         self.is_sparse = is_sparse

#     def forward(self, x, adj):
#         if self.dropout is not None:
#             x = F.dropout(x, self.dropout, self.training)
#         z = torch.mm(x, self.weight)

#         if self.is_sparse:
#             z = torch.spmm(adj, z)
#         else:
#             z = torch.mm(adj, z)



#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.weight)

class Encoder_sparse(Module):
    """
    Sparse version of Encoder
    """
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder_sparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.spmm(adj, z)
        
        hiden_emb = z
        
        h = torch.mm(z, self.weight2)
        h = torch.spmm(adj, h)
        
        emb = self.act(z)
        
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.spmm(adj, z_a)
        emb_a = self.act(z_a)
         
        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)
        
        g_a = self.read(emb_a, self.graph_neigh)
        g_a =self.sigm(g_a)       
       
        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb)
        
        return hiden_emb, h, ret, ret_a     

class Encoder_sc(torch.nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0.0, act=F.relu):
        super(Encoder_sc, self).__init__()
        self.dim_input = dim_input
        self.dim1 = 256
        self.dim2 = 64
        self.dim3 = 32
        self.act = act
        self.dropout = dropout
        
        #self.linear1 = torch.nn.Linear(self.dim_input, self.dim_output)
        #self.linear2 = torch.nn.Linear(self.dim_output, self.dim_input)
        
        #self.weight1_en = Parameter(torch.FloatTensor(self.dim_input, self.dim_output))
        #self.weight1_de = Parameter(torch.FloatTensor(self.dim_output, self.dim_input))
        
        self.weight1_en = Parameter(torch.FloatTensor(self.dim_input, self.dim1))
        self.weight2_en = Parameter(torch.FloatTensor(self.dim1, self.dim2))
        self.weight3_en = Parameter(torch.FloatTensor(self.dim2, self.dim3))
        
        self.weight1_de = Parameter(torch.FloatTensor(self.dim3, self.dim2))
        self.weight2_de = Parameter(torch.FloatTensor(self.dim2, self.dim1))
        self.weight3_de = Parameter(torch.FloatTensor(self.dim1, self.dim_input))
      
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1_en)
        torch.nn.init.xavier_uniform_(self.weight1_de)
        
        torch.nn.init.xavier_uniform_(self.weight2_en)
        torch.nn.init.xavier_uniform_(self.weight2_de)
        
        torch.nn.init.xavier_uniform_(self.weight3_en)
        torch.nn.init.xavier_uniform_(self.weight3_de)
        
    def forward(self, x):
        x = F.dropout(x, self.dropout, self.training)
        
        #x = self.linear1(x)
        #x = self.linear2(x)
        
        #x = torch.mm(x, self.weight1_en)
        #x = torch.mm(x, self.weight1_de)
        
        x = torch.mm(x, self.weight1_en)
        x = torch.mm(x, self.weight2_en)
        x = torch.mm(x, self.weight3_en)
        
        x = torch.mm(x, self.weight1_de)
        x = torch.mm(x, self.weight2_de)
        x = torch.mm(x, self.weight3_de)
        
        return x
    
class Encoder_map(torch.nn.Module):
    def __init__(self, n_cell, n_spot):
        super(Encoder_map, self).__init__()
        self.n_cell = n_cell
        self.n_spot = n_spot
          
        self.M = Parameter(torch.FloatTensor(self.n_cell, self.n_spot))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.M)
        
    def forward(self):
        x = self.M
        
        return x 
