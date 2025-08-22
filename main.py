from anndata import AnnData
from GraphST.graphst import GraphST
import squidpy as sq
from GraphST.utils import clustering
from GraphST.model import BaseEncoder, ExplainableEncoder
import numpy as np
import pandas as pd
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import dense_to_sparse
import time
import pickle
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset_V1_Human_Lymph_Node() -> AnnData:
    return sq.read.visium("dataset/V1_Human_Lymph_Node")

def annotate_graphclusters_V1_Human_Lymph_Node(adata: AnnData):
    graphclust = pd.read_csv("dataset/V1_Human_Lymph_Node/analysis/clustering/graphclust/clusters.csv")
    graphclust.set_index("Barcode", inplace=True)
    adata.obs['graphclust'] = graphclust['Cluster'] - 1
    

def get_graphst(adata: AnnData, device=device) -> GraphST :
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


def save_explanations(explainer_spec, explainer_title, explainer_epochs, explanations):
    if not os.path.exists("explanations"):
        os.mkdir("explanations")

    output_dir = f"explanations/{explainer_title}_epochs_{explainer_epochs}"
    os.mkdir(output_dir)

    with open(os.path.join(output_dir, "explanations.pkl"), "wb") as f:
        pickle.dump(explanations, f)

    with open(os.path.join(output_dir, "spec.json"), "wt", encoding="utf-8") as f:
        json.dump(explainer_spec, f)

    return output_dir

def load_explanations(dir_path):
    with open(os.path.join("explanations", dir_path, "explanations.pkl"), "rb") as f:
        explanations = pickle.load(f)

    with open(os.path.join("explanations", dir_path, "spec.json"), "rt", encoding="utf-8") as f:
        spec = json.load(f)

    return explanations, spec   

if __name__ == "__main__":
    path="graphst_base_encoder.pt"


    torch.cuda.empty_cache()
    import multiprocessing as mp
    mp.set_start_method("fork", force=True)  # Only once per program
    
    print(device)

    adata = load_dataset_V1_Human_Lymph_Node()
    annotate_graphclusters_V1_Human_Lymph_Node(adata)

    gst = get_graphst(adata)

    if path is None:
        path=input("Load graphST? Enter the path or N for training. Leave empty to use default path: graphst_base_encoder.pt\n")
    
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
            base_encoder = load_graphst_base_encoder(gst,path).to(device)
        else:
            base_encoder = load_graphst_base_encoder(gst).to(device)
        
    # apply_clustering(graphst.gadata, method="leiden", n_clusters=7)

    from distilled_decoder import DistilledDecoder, ClusterPredictor


    decoder = DistilledDecoder(gst.dim_output, gst.adata.obs['graphclust'].nunique()).to(device)
    # decoder.fit(gst, base_encoder, epochs=1000, lr=0.01, val_ratio=0.2, eval_frequency=10)
    decoder.load_state_dict(torch.load("best_distilled_decoder.pt"))


    cluster_predictor = ClusterPredictor(base_encoder, decoder).to(device)

    edge_index, _ = dense_to_sparse(gst.adj)
    cluster_explainer = Explainer(
        model=cluster_predictor,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='phenomenon',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    clusters = torch.argmax(cluster_predictor(gst.features, edge_index), dim=1)
    res = cluster_explainer(gst.features, edge_index, target=clusters)

    # Save explanation object
    with open("cluster_explanation.pkl", "wb") as f:
        pickle.dump(res, f)

    # To reuse/load explanation later:
    # with open("cluster_explanation.pkl", "rb") as f:
    #     res = pickle.load(f)


    # node_barcodes = adata.obs.index.tolist()
    # node_ids = [adata.obs.index.get_loc(barcode) for barcode in node_barcodes]
    

    # from explain_gnnexplainer import explain, explain_batch

    # limit = 2
    # epochs = 250
    
    # ########
    # from torch_geometric.utils import dense_to_sparse
    # edge_index, _ = dense_to_sparse(gst.adj)

    # _xmodel = ExplainableEncoder(base_encoder).to(device)
    # explainer = Explainer(
    #     model=_xmodel,
    #     algorithm=GNNExplainer(epochs=epochs),
    #     explanation_type='model',
    #     node_mask_type='attributes',
    #     edge_mask_type=None,
    #     model_config=dict(
    #         mode='multiclass_classification',
    #         task_level='node',
    #         return_type='log_probs',
    #     ),
    # )

    # explainer2 = Explainer(
    #     model=_xmodel,
    #     algorithm=GNNExplainer(epochs=epochs),
    #     explanation_type='model',
    #     node_mask_type='attributes',
    #     edge_mask_type=None,
    #     model_config=dict(
    #         mode='regression',
    #         task_level='node',
    #         # return_type='log_probs',
    #     ),
    # )

    # explainer4 = Explainer(
    #     model=_xmodel,
    #     algorithm=GNNExplainer(epochs=1000),
    #     explanation_type='model',
    #     node_mask_type='attributes',
    #     edge_mask_type=None,
    #     model_config=dict(
    #         mode='regression',
    #         task_level='node',
    #         # return_type='log_probs',
    #     ),
    # )


    # res = explainer4(gst.features, edge_index)


    # res_ = explainer(gst.features, edge_index, index=1)

    # res3 = explainer3(gst.features, edge_index)

    # print("res: ", )
    # for res in explain(device, base_encoder, gst.features, gst.adj, node_ids[:limit], epochs=epochs, parallel_num_proc=1):
    #     print(res)
