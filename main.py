from GraphST import GraphST
import squidpy as sq
import scanpy as sc
import matplotlib.pyplot as plt
import networkx as nx


adata = sq.read.visium("dataset/V1_Human_Lymph_Node")
print(adata)

# sc.pl.spatial(adata, spot_size=1.3)
# sc.pl.spatial(adata, color=["CD3D", "MS4A1"], spot_size=1.3)



print(adata)
gst = GraphST(adata)

print(gst)
nx.draw_networkx(gm, )
