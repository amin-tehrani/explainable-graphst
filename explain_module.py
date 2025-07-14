import multiprocessing as mp
from torch_geometric.utils import dense_to_sparse
from torch_geometric.explain import Explainer, GNNExplainer

import time


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
    res = explainer(feat, edge_index, index=index)
    print(f"Finished explaining node {index} at {time.ctime()}")
    return res

# Main explain function
def explain(xmodel, feat, adj, node_ids: list, epochs=100, parallel_num_proc=mp.cpu_count()):
    if parallel_num_proc is None or parallel_num_proc <= 1:
        print(f"Explaining Total {len(node_ids)} nodes without parallelization.")
        # Sequential
        for node_index in node_ids:
            yield explain_single_node((xmodel, feat, adj, node_index, epochs))
    else:
        print(f"Explaining Total {len(node_ids)} nodes in parallel with {parallel_num_proc} processes.")
        args_list = [(xmodel, feat, adj, node_index, epochs) for node_index in node_ids]
        with mp.get_context("spawn").Pool(processes=parallel_num_proc) as pool:
            for explanation in pool.imap(explain_single_node, args_list):
                yield explanation
