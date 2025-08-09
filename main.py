from anndata import AnnData
from GraphST.graphst import GraphST
import squidpy as sq
from GraphST.utils import clustering
from GraphST.model import BaseEncoder, ExplainableEncoder
import numpy as np
import pandas as pd
import torch
from torch_geometric.explain import Explainer, GNNExplainer
import time

device = torch.device('cpu')

def load_dataset_V1_Human_Lymph_Node() -> AnnData:
    return sq.read.visium("dataset/V1_Human_Lymph_Node")

def annotate_graphclusters_V1_Human_Lymph_Node(adata: AnnData):
    graphclust = pd.read_csv("dataset/V1_Human_Lymph_Node/analysis/clustering/graphclust/clusters.csv")
    graphclust.set_index("Barcode", inplace=True)
    adata.obs['graphclust'] = graphclust['Cluster'] - 1
    

def get_graphst(adata: AnnData) -> GraphST :
    return GraphST(adata, device=device)

def train_graphst(graphst: GraphST):
    return graphst.train()

def save_graphst_base_encoder(graphst: GraphST, path="graphst_base_encoder.pt"):
    bencoder = graphst.model.base_encoder
    return torch.save(bencoder.state_dict(), path)

def load_graphst_base_encoder(graphst: GraphST,path="graphst_base_encoder.pt"):
    state_dict = torch.load(path)
    base_encoder = BaseEncoder(graphst.dim_input, graphst.dim_output, graphst.graph_neigh, is_sparse=False)
    base_encoder.load_state_dict(state_dict)
    return base_encoder


def apply_clustering(gadata: AnnData, method="leiden", **kwargs):
    clustering(gadata, method, **kwargs)


def explain_gnnexplainer():
    pass



if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("fork", force=True)  # Only once per program
    


    adata = load_dataset_V1_Human_Lymph_Node()
    annotate_graphclusters_V1_Human_Lymph_Node(adata)

    gst = get_graphst(adata)

    # path=input("Load graphST? Enter the path or N for training. Leave empty to use default path: graphst_base_encoder.pt\n")
    path=""
    if path == "N":
        train_graphst(gst)
        if save_path:=input("Save graphST? Enter the path or N for no saving. Leave empty to use default path: graphst_base_encoder.pt\n") != "N":
            if save_path:
                save_graphst_base_encoder(gst, save_path)
            else:
                save_graphst_base_encoder(gst)

        base_encoder = gst.model.base_encoder
    else:
        if path:
            base_encoder = load_graphst_base_encoder(gst,path)
        else:
            base_encoder = load_graphst_base_encoder(gst)
        
    # apply_clustering(graphst.gadata, method="leiden", n_clusters=7)

    node_barcodes = adata.obs.index.tolist()
    node_ids = [adata.obs.index.get_loc(barcode) for barcode in node_barcodes]
    

    from explain_gnnexplainer import explain

    limit = 100
    epochs = 10


    for i in explain(base_encoder, gst.features, gst.adj, node_ids[:limit], epochs=epochs):
        print(i)
