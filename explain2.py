# %%
from torchviz import make_dot
import squidpy as sq
import scanpy as sc
import matplotlib.pyplot as plt
from GraphST.graphst import GraphST
from GraphST.utils import clustering
import networkx as nx
import torch
import numpy as np
import pandas as pd
import time
from torch_geometric.explain import Explainer, GNNExplainer

from GraphST.graphst import Encoder, ExplainableEncoder, BaseEncoder, GraphStEncoder
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import dense_to_sparse

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
device

# %%
adata = sq.read.visium("dataset/V1_Human_Lymph_Node")
adata

# %%
adata.obs.index.unique()

# %%
graphclust = pd.read_csv("dataset/V1_Human_Lymph_Node/analysis/clustering/graphclust/clusters.csv")
graphclust.set_index("Barcode", inplace=True)
adata.obs['graphclust'] = graphclust['Cluster'] - 1

# %%
gst = GraphST(adata, device=device)

# %%
# from GraphST.model import AvgReadout

# ar = AvgReadout()
# ar.forward(torch.rand(10, 5), torch.rand((10, 10)))

# %%
gadata = gst.train()

# %%
# clustering(gadata, method="leiden")

# %%
G = nx.from_numpy_array(gst.adj.numpy())

# %%
hires_scale = adata.uns['spatial']['V1_Human_Lymph_Node']['scalefactors']['tissue_hires_scalef']
coords_hires = adata.obsm['spatial'] * hires_scale
pos = coords_hires

# %%
# ax = plt.subplot(1, 2, 1)
# ax.imshow(adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires'])
# ax.invert_yaxis()
# ax.axis('off')
# nx.draw(G, pos=pos, node_size=3, node_color=gadata.obs['graphclust'].astype(int).tolist(), edge_color='gray', with_labels=False, ax=ax)
# ax.set_title("True Clustering")

# ax = plt.subplot(1, 2, 2)
# ax.imshow(adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires'])
# ax.invert_yaxis()
# ax.axis('off')
# nx.draw(G, pos=pos, node_size=3, node_color=gadata.obs['leiden'].astype(int).tolist(), edge_color='gray', with_labels=False, ax=ax)
# ax.set_title("GraphST using Leiden clustering")


# plt.show()

# %%
# save the gst.model weights
# torch.save(gst.model.base_encoder.state_dict(), "base_encoder_weights.pth")

# %%
base_encoder = gst.model.base_encoder
# base_encoder = BaseEncoder(gst.dim_input, gst.dim_output, gst.graph_neigh, is_sparse=False)
# base_encoder.load_state_dict(torch.load("base_encoder_weights.pth", weights_only=True), )
# base_encoder = base_encoder.to(device)

xmodel = ExplainableEncoder(base_encoder).to(device)

# %%
import multiprocessing as mp

# Move this outside to make it picklable
def explain_single_node(args):
    xmodel, feat, adj, index, epochs = args
    edge_index, _ = dense_to_sparse(adj)
    explainer = Explainer(
        model=xmodel,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type=None,
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )
    print(f"Explaining node {index} at {time.ctime()}")
    return explainer(feat, edge_index, index=index)

# Main explain function
def explain(xmodel, feat, adj, node_ids: list, epochs=100, parallel_num_proc=mp.cpu_count()):
    if parallel_num_proc is None or parallel_num_proc <= 1:
        # Sequential
        for node_index in node_ids:
            yield explain_single_node((xmodel, feat, adj, node_index, epochs))
    else:
        args_list = [(xmodel, feat, adj, node_index, epochs) for node_index in node_ids]
        with mp.get_context("spawn").Pool(processes=parallel_num_proc) as pool:
            for explanation in pool.imap(explain_single_node, args_list):
                yield explanation


# %%
import importlib
import explain_module
importlib.reload(explain_module)

from explain_module import explain, explain_single_node

# %%
# cluster_num = 0
# node_barcodes = adata.obs[adata.obs['graphclust'] == cluster_num].index.tolist()
# # convert node_ids (barcodes) to numerical indices
# node_ids = [adata.obs.index.get_loc(barcode) for barcode in node_barcodes]
# explanations = list(explain(base_encoder, gst.features, gst.adj, node_ids[:10]))

# %%
def calc_cluster_explanations(cluster_num, limit=10, epochs=100, parallel_num_proc=mp.cpu_count()):
    node_barcodes = adata.obs[adata.obs['graphclust'] == cluster_num].index.tolist()
    # convert node_ids (barcodes) to numerical indices
    node_ids = [adata.obs.index.get_loc(barcode) for barcode in node_barcodes]
    if limit is None:
        limit = len(node_ids)
    explanations = list(explain(xmodel, gst.features, gst.adj, node_ids[:limit], epochs=epochs, parallel_num_proc=parallel_num_proc))
    return node_ids, explanations

def load_explanations(path):
    import pickle
    with open(path, 'rb') as f:
        node_ids, explanations = pickle.load(f)
    return node_ids, explanations

def save_explanations(path, node_ids, explanations):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump((node_ids, explanations), f)

def calc_save_cluster_explanations(cluster_num, limit=10, epochs=100, parallel_num_proc=mp.cpu_count()):
    node_ids, explanations = calc_cluster_explanations(cluster_num, limit, epochs, parallel_num_proc)
    path = f"cluster_{cluster_num}_{limit}_{epochs}.pkl"
    save_explanations(path, node_ids, explanations)
    return node_ids, explanations, path


# %%
node_ids0, explanations0, path0 = calc_save_cluster_explanations(0, limit=20, epochs=100)

# %%
node_ids1, explanations1, path1 = calc_save_cluster_explanations(1, limit=10, epochs=200)

# %%
node_hvg_df.iloc[0]

# %%
# hvg = gst.adata.var[gst.adata.var['highly_variable']]
# node_hvg_df = pd.DataFrame(columns=['Spot_ID','Cluster']+list(gst.adata.var.index), index=gst.adata.obs.index, dtype=float)
# node_hvg_df['Cluster'] = gst.adata.obs['graphclust']

# def add_explanations_to_df(node_ids, explanations, threshold=0.7):
#    for i, node_index, explanation in zip(range(len(node_ids)), node_ids, explanations):
#         # print(f"Node ID: {node_index}")
#         feature_mask = explanation.node_mask[node_index]

#         # plt.figure(figsize=(10, 5))
#         # plt.hist(feature_mask.cpu().numpy(), bins=50, color='blue', alpha=0.7)
#         # plt.title(f'Feature Importance Histogram (Node {node_index})')
#         # plt.xlabel('Feature Importance')
#         # plt.ylabel('Frequency')
#         # plt.grid(True)
#         # plt.show()


#         # important features are above threshold
#         important_features_ids = torch.where(feature_mask > threshold)[0]
        
#         # TODO: Implement also top-k
#         important_features_mask = feature_mask[important_features_ids]

#         # print("Num of important features:", len(important_features_ids))

#         # list gene_id from hvg of important features
#         important_features_gene_ids = hvg.index[important_features_ids].tolist()

#         # print("Important Gene Ids:", important_features_gene_ids)
#         # print(important_features_mask.cpu().numpy())
#         node_hvg_df.loc[node_index, important_features_gene_ids] = important_features_mask.cpu().numpy() 

# def analyze_explanations(node_ids, explanations, threshold=0.7, top_k=None):

#     hvg = gst.adata.var[gst.adata.var['highly_variable']]

#     # df is a dataframe with gene_id as index and feature importance as values
#     df = pd.DataFrame(columns=hvg.index, index=node_ids, dtype=float).fillna(0.0)
#     for i, node_index, explanation in zip(range(len(node_ids)), node_ids, explanations):
#         # print(f"Node ID: {node_index}")
#         feature_mask = explanation.node_mask[node_index]

#         # plt.figure(figsize=(10, 5))
#         # plt.hist(feature_mask.cpu().numpy(), bins=50, color='blue', alpha=0.7)
#         # plt.title(f'Feature Importance Histogram (Node {node_index})')
#         # plt.xlabel('Feature Importance')
#         # plt.ylabel('Frequency')
#         # plt.grid(True)
#         # plt.show()


#         # important features are above threshold
#         important_features_ids = torch.where(feature_mask > threshold)[0]
        
#         # TODO: Implement also top-k
#         important_features_mask = feature_mask[important_features_ids]

#         # print("Num of important features:", len(important_features_ids))

#         # list gene_id from hvg of important features
#         important_features_gene_ids = hvg.index[important_features_ids].tolist()

#         # print("Important Gene Ids:", important_features_gene_ids)
#         # print(important_features_mask.cpu().numpy())
#         df.loc[node_index, important_features_gene_ids] = important_features_mask.cpu().numpy()
# 1
#     return df
    

# %%
# df1 = analyze_explanations(node_ids, explanations, threshold=0.1, top_k=None)

# %%
# # Plot a diagram whose x-axis is the df columns and the y-axis is the sum of the df rows
# plt.figure(figsize=(10, 5))
# df1.sum(axis=0).plot(kind='bar', color='blue', alpha=0.7)
# # Hide labels on x-axis
# plt.xticks([])
# plt.title('Sum of Feature Importance for Each Gene')
# plt.xlabel('Gene')
# plt.ylabel('Sum of Feature Importance')
# plt.show()

# %%
# node_ids2, explanations2 = get_cluster_explanations(1, limit=10)

# %%
# df2 = analyze_explanations(node_ids2, explanations2, threshold=0.7, top_k=None)
# df2

# %%
# # list the most important features
# feature_mask = explanation.node_mask[node_index]


# # count non-zero elements in the feature mask
# num_nonzero_features = feature_mask.nonzero().size(0)


# # Draw a histogram of the feature mask
# plt.figure(figsize=(10, 5))
# plt.hist(feature_mask.cpu().numpy(), bins=50, color='blue', alpha=0.7)
# plt.title(f'Feature Importance Histogram (Node {node_index})')
# plt.xlabel('Feature Importance')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

# # important features are above 0.8
# important_features_ids = torch.where(feature_mask > 0.8)[0]
# important_features_mask = feature_mask[important_features_ids]
# important_features = gst.features[node_index][important_features_ids]


# hvg = gadata.var[gadata.var['highly_variable']]


# # list gene_id from hvg of important features
# important_features_gene_ids = hvg.index[important_features_ids].tolist()

# print(*zip(important_features_ids, important_features_gene_ids, important_features_mask), sep='\n')



# %%
# path = 'feature_importance100.png'

# explanation.visualize_feature_importance(path, top_k=100)
# print(f"Feature importance plot has been saved to '{path}'")

# path = 'subgraph.png'
# explanation.visualize_graph(path)
# print(f"Subgraph visualization plot has been saved to '{path}'")


