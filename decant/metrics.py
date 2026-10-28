"""Integration metrics for DECANT embeddings.

Six metrics (AMI, ARI, NMI, batchASW, Graph_connectivity, iLISI) plus the
Overall summary score.
"""

import numpy as np
import scanpy as sc
import sklearn.metrics


METRIC_NAMES = ["AMI", "ARI", "NMI", "batchASW", "Graph_connectivity", "iLISI"]


def evaluate_metrics(adata, emb_key, batch_key, label_key):
    """Compute the six integration metrics on an embedding."""
    import scib

    sc.pp.neighbors(adata, use_rep=emb_key, n_neighbors=15)
    sc.tl.leiden(adata, key_added="leiden_eval")

    true = adata.obs[label_key]
    pred = adata.obs["leiden_eval"]

    return {
        "AMI": float(sklearn.metrics.adjusted_mutual_info_score(true, pred)),
        "ARI": float(sklearn.metrics.adjusted_rand_score(pred, true)),
        "NMI": float(sklearn.metrics.normalized_mutual_info_score(pred, true)),
        "batchASW": float(scib.me.silhouette_batch(
            adata, batch_key=batch_key, label_key=label_key,
            embed=emb_key, verbose=False)),
        "Graph_connectivity": float(scib.me.graph_connectivity(
            adata, label_key=label_key)),
        "iLISI": float(scib.me.ilisi_graph(
            adata, batch_key=batch_key, type_="embed",
            use_rep=emb_key, scale=False)),
    }


def calc_overall(metrics, n_batches):
    """Combine the six metrics into a single Overall score."""
    ilisi_norm = (metrics["iLISI"] - 1.0) / max(n_batches - 1, 1)
    bio = np.mean([metrics["AMI"], metrics["ARI"], metrics["NMI"]])
    batch = np.mean([metrics["batchASW"], metrics["Graph_connectivity"], ilisi_norm])
    return float(0.4 * batch + 0.6 * bio)
