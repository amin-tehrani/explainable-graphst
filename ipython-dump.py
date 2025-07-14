# coding: utf-8
gst
gst.train()
gst
gst.adata
adata
gst.adata
gst.adata.emb
gst.adata.obsm
gst.adata.obsm['emb']
type(gst.adata.obsm['emb'])
gst.adata.obsm['emb'].shape
adata
adata.obsm['spatial']
adata.uns['spatial']
adata.X
adata.n_obs
adata.obs
adata.obsm
adata.obsm['spatial'].shape
type(adata.n_obs)
type(adata.obs)
adata.obs.unique
adata.obs.unique()
adata.obs
adata.obs['in_tissue'].unique
adata.obs['in_tissue'].unique()
adata.obs
adata.obs.index
adata.obs.index.apply()
adata.obs.index
type(adata.obs.index)
list(adata.obs.index)
list(map(lambda x:x, adata.obs.index))
list(map(lambda x:x.split('-')[1], adata.obs.index))
set(map(lambda x:x.split('-')[1], adata.obs.index))
adata.obs
adata.obs['array_row'].unique()
sorted(adata.obs['array_row'].unique())
adata.obs['array_col'].unique()
sorted(adata.obs['array_col'].unique())
adata.obs
adata
adata.var
adata
adata.obsm
adata.obsm['spatial'].shape
adata.obsm['spatial']
adata
adata.obs
adata.obs.iloc[10]
adata.obsm['spatial'].iloc[10]
adata.obsm['spatial'][10]
adata.obs['spatial']
adata
adata.uns['spatial']
type(adata.uns['spatial'])
adata.uns['spatial'].keys()
adata.uns['spatial']['V1_Human_Lymph_Node']
adata.uns['spatial']['V1_Human_Lymph_Node'].keys()
adata.uns['spatial']['V1_Human_Lymph_Node']['metadata']
adata.uns['spatial']['V1_Human_Lymph_Node']['images']
adata.uns['spatial']['V1_Human_Lymph_Node']['images'].keys()
adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires']
type(adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires'])
adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires'].shape
adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires'].shape()
adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires'].shape
plt.imshow(adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires'])
plt.show()
plt.imshow(adata.uns['spatial']['V1_Human_Lymph_Node']['images']['lowers'])
plt.imshow(adata.uns['spatial']['V1_Human_Lymph_Node']['images']['lowres'])
plt.show()
plt.subplot(1,2,1)
plt.imshow(adata.uns['spatial']['V1_Human_Lymph_Node']['images']['lowers'])
plt.imshow(adata.uns['spatial']['V1_Human_Lymph_Node']['images']['lowrs'])
plt.imshow(adata.uns['spatial']['V1_Human_Lymph_Node']['images']['lowres'])
plt.subplot(1,2,2)
plt.imshow(adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires'])
plt.show()
adata.uns['spatial']['V1_Human_Lymph_Node']
adata.uns['spatial']['V1_Human_Lymph_Node'].keys()
adata
gst.adata
gst.adata.var
gst.adata
gst.adata.uns['hvg']
gst.adata.uns['log1p']
gst.adata.obsm
gst.adata.obsm['adj']
import networkx as nx
nx.from_numpy_array(gst.adata.obsm['adj'])
G = nx.from_numpy_array(gst.adata.obsm['adj'])
G
plt.figure()
plt.plot(G)
plt.show()
plt.figure()
nx.draw(G)
plt.show()
gst.adata.obsm
gst.adata.obsm['graph_neigh']
gst.adata.obsm['feat']
gst.adata.obsm['feat_a']
gst.adata.obsm['emb']
gst.adata.obsm
gst.adata.obsm['label_CLS']
gst.adata.obsm['label_CSL']
nx.draw(G, node_size=0.1)
plt.show()
plt.show()
plt.show()
G
G.nodes
G.nodes[0]
G.nodes[1]
type(G.nodes)
G.nodes
G.nodes.keys()
type(G.nodes.keys())
list(G.nodes.keys())
import random
subG = G.subgraph(random.sample(list(G.nodes), 100)).copy()
subG
nx.draw(subG, node_size=0.1)
plt.show()
subG = G.subgraph(random.sample(list(G.nodes), 500)).copy()
nx.draw(subG, node_size=0.1)
plt.show()
nx.draw(subG, node_size=0.5, edge_color='gray', with_labels=True)
plt.show()
adata
adata.obs['spatial']
gst.adata.obs['spatial']
adata.obs
adata.obsm['spatial'][10]
adata.obsm['spatial']
adata
adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires'].shape
adata.uns['spatial']['V1_Human_Lymph_Node']['scalefactors']
hires_scale = adata.uns['spatial']['V1_Human_Lymph_Node']['scalefactors']['tissue_hires_scalef']
coords_hires = adata.obsm['spatial'] * hires_scale
hires_scale
coords_hires
adata.obsm['spatial']
coords_hires
coords_hires.shape
nx.draw(subG, coords_hires, node_size=0.5, edge_color='gray', with_labels=True)
plt.show()
nx.draw(subG, coords_hires, node_size=1, edge_color='gray', with_labels=False)
plt.show()
subG = G.subgraph(random.sample(list(G.nodes), 1000)).copy()
nx.draw(subG, coords_hires, node_size=1, edge_color='gray', with_labels=False)
plt.show()
subG = G.subgraph(random.sample(list(G.nodes), 2000)).copy()
nx.draw(subG, coords_hires, node_size=1, edge_color='gray', with_labels=False)
plt.show()
plt.imshow(adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires'])
plt.show()
nx.draw(subG, coords_hires, node_size=1, edge_color='gray', with_labels=False)
plt.show()
plt.imshow(adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires'])
plt.show()
nx.draw(subG, coords_hires.T, node_size=1, edge_color='gray', with_labels=False)
coords_hires
coords_hires.shape
coords_hires.reshape(1,0)
coords_hires.unique()
coords_hires[0]
coords_hires[:]
coords_hires[1:]
coords_hires[:1]
coords_hires[:2]
coords_hires[::1]
coords_hires[::0]
coords_hires[:][1]
np.unique(coords_hires[:1])
import numpy as np
np.unique(coords_hires[:1])
np.unique(coords_hires)
np.unique(coords_hires[:,0])
np.unique(coords_hires[:,1])
nx.draw(subG, coords_hires, node_size=1, edge_color='gray', with_labels=False)
plt.show()
pos = coords_hires[:,[1,0]]
pos
nx.draw(subG, pos, node_size=1, edge_color='gray', with_labels=False)
plt.show()
fig, ax = plt.subplots(figsize=(8,8))
plt.imshow(adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires'])
plt.show()
plt.show()
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires'])
ax.invert_yaxis()
ax.axis('off')
nx.draw(G, pos=pos, node_size=1, edge_color='gray', with_labels=False, ax=ax)
plt.show()
