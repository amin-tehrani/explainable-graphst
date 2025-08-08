import multiprocessing as mp
from torch_geometric.utils import dense_to_sparse
from torch_geometric.explain import Explainer, GNNExplainer
from GraphST.model import ExplainableEncoder, BaseEncoder

import time

# Global variables for worker processes
_explainer = None
_xmodel = None
_feat = None
_edge_index = None

def init_worker(feat, edge_index, base_encoder_state_dict, base_encoder_args, epochs):
    """Initialize each worker process with its own model and explainer"""
    global _explainer, _xmodel, _feat, _edge_index
    

    # Recreate model in each process
    _base_encoder = BaseEncoder(**base_encoder_args)
    _base_encoder.load_state_dict(base_encoder_state_dict)
    _base_encoder.eval()

    _xmodel = ExplainableEncoder(_base_encoder)

    _feat = feat
    _edge_index = edge_index
    
    # Create explainer once per process
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

# def explain_single_node(args):
#     feat, edge_index, index = args
#     # edge_index, _ = dense_to_sparse(adj)
    
#     print(f"Explaining node {index} at {time.ctime()}")
#     res = _explainer(feat, edge_index, index=index)
#     print(f"Finished explaining node {index} at {time.ctime()}")
#     return res

def explain_batch(args):
    batch_id, node_ids  = args
    
    print(f"Batch#{batch_id} @ {mp.current_process().name} - Begin explaining {len(node_ids)} nodes at {time.ctime()}")

    res = []
    for node_index in node_ids:
        starttime =time.time()
        print(f"Batch#{batch_id} @ {mp.current_process().name} - Node#{node_index}: explaining node {node_index} at {starttime}")
        res.append(_explainer(_feat, _edge_index, index=node_index))
        endtime = time.time()
        print(f"Batch#{batch_id} @ {mp.current_process().name} - Node#{node_index}: finished {node_index} at {endtime}, duration: {(endtime-starttime):.3f}")

    return res
    

def explain(base_encoder, feat, adj, node_ids: list, epochs=100, parallel_num_proc=mp.cpu_count()):
    edge_index, _ = dense_to_sparse(adj)
    if parallel_num_proc is None or parallel_num_proc <= 1:
        parallel_num_proc = 1
        
    print(f"Explaining Total {len(node_ids)} nodes with {parallel_num_proc} processes.")
    
    # Prepare model for serialization
    base_encoder_state_dict = base_encoder.state_dict()
    base_encoder_args = base_encoder.get_args_dict() # You'll need to pass the model's init arguments
    

    batchs = []
    for i in range(parallel_num_proc):
        start = i * len(node_ids) // parallel_num_proc
        end = (i + 1) * len(node_ids) // parallel_num_proc
        batchs.append(node_ids[start:end])
    
    args_list = list(enumerate(batchs))

    with mp.get_context("spawn").Pool(
        processes=parallel_num_proc,
        initializer=init_worker,
        initargs=(feat, edge_index, base_encoder_state_dict, base_encoder_args, epochs)
    ) as pool:
        for explanation in pool.imap(explain_batch, args_list):
            yield explanation