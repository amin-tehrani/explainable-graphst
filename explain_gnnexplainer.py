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

def init_worker(device, feat, edge_index, _base_encoder, epochs, outdir):
    global _explainer, _xmodel, _feat, _edge_index, _outdir

    _outdir = outdir

    print(f"[{mp.current_process().name}] Initializing worker...", flush=True)

    print(f"[{mp.current_process().name}] Base encoder:", _base_encoder, flush=True)

    _xmodel = ExplainableEncoder(_base_encoder).to(device)
    _feat = feat
    _edge_index = edge_index

    # print(f"...", _xmodel, _feat, _edge_index, flush=True)

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

    res = {}
    for node_index in node_ids:
        stime = time.time()
        print(f"[{mp.current_process().name}] Batch {batch_id}: Node {node_index}, start @ {time.ctime(stime)} ", flush=True)
        x = _explainer(_feat, _edge_index, index=node_index)
        
        
        etime = time.time()
        print(f"[{mp.current_process().name}] Batch {batch_id}: Node {node_index}, end @ {time.ctime(etime)}, duration: {etime-stime} ", flush=True)

    return res

def explain(device,base_encoder, feat, adj, node_ids, outdir="explanations", epochs=100, parallel_num_proc=mp.cpu_count()):
    edge_index, _ = dense_to_sparse(adj)

    if parallel_num_proc is None or parallel_num_proc <= 1:
        init_worker(device, feat, edge_index, base_encoder, epochs)
        return (i for i in explain_batch((0, node_ids)).items())
        

    batches = []
    for i in range(parallel_num_proc):
        start = i * len(node_ids) // parallel_num_proc
        end = (i + 1) * len(node_ids) // parallel_num_proc
        batches.append(node_ids[start:end])

    args_list = list(enumerate(batches))

    with mp.get_context("fork").Pool(
        processes=parallel_num_proc,
        initializer=init_worker,
        initargs=(device, feat, edge_index, base_encoder, epochs, outdir)
    ) as pool:
        for results in pool.imap(explain_batch, args_list):
            for i in results.items():
                yield i

