import multiprocessing as mp
from torch_geometric.utils import dense_to_sparse
from torch_geometric.explain import Explainer, GNNExplainer
from GraphST.model import ExplainableEncoder, BaseEncoder
import torch

import time

# Globals set by init_worker
_explainer = None
_xmodel = None
_feat = None
_edge_index = None

def init_worker(feat, edge_index, base_encoder_path, base_encoder_args, epochs):
    global _explainer, _xmodel, _feat, _edge_index

    print(f"[{mp.current_process().name}] Initializing worker...", flush=True)

    _base_encoder = BaseEncoder(**base_encoder_args)
    _base_encoder.load_state_dict(torch.load(base_encoder_path))
    # _base_encoder.eval()
    print(f"[{mp.current_process().name}] Base encoder:", _base_encoder, flush=True)

    _xmodel = ExplainableEncoder(_base_encoder)
    _feat = feat
    _edge_index = edge_index

    print(f"...", _xmodel, _feat, _edge_index, flush=True)

    _explainer = Explainer(
        model=_xmodel,
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
    print(f"[{mp.current_process().name}] Worker ready", flush=True)

def explain_batch(args):
    batch_id, node_ids = args
    print(f"[{mp.current_process().name}] Batch {batch_id}: explaining {len(node_ids)} nodes", flush=True)

    res = []
    for node_index in node_ids:
        res.append(_explainer(_feat, _edge_index, index=node_index))
    return res

def explain(base_encoder, feat, adj, node_ids, epochs=100, parallel_num_proc=mp.cpu_count()):
    edge_index, _ = dense_to_sparse(adj)

    base_encoder_path = "graphst_base_encoder.pt"
    base_encoder_args = base_encoder.get_args_dict()

    batches = []
    for i in range(parallel_num_proc):
        start = i * len(node_ids) // parallel_num_proc
        end = (i + 1) * len(node_ids) // parallel_num_proc
        batches.append(node_ids[start:end])

    args_list = list(enumerate(batches))

    with mp.get_context("fork").Pool(
        processes=parallel_num_proc,
        initializer=init_worker,
        initargs=(feat, edge_index, base_encoder_path, base_encoder_args, epochs)
    ) as pool:
        for explanation in pool.imap(explain_batch, args_list):
            yield explanation
