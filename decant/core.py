"""DECANT feature selection with a PCA embedding.

DECANT ranks every feature by biological signal (Fc) and batch effect (Fb)
and keeps the top features by score = Fc_rank / (Fb_rank + 1) ** gamma.
"""

import gc

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import scipy.stats
import sklearn.metrics

from .preprocess import tfidf_for_ranking


def compute_feature_ranks(adata, batch_key, omics, seed=0):
    """Compute the biological-signal rank (Fc) and batch-effect rank (Fb) per feature."""
    X_rank = tfidf_for_ranking(adata.X) if omics == "atac" else adata.X

    batches = list(adata.obs[batch_key].unique())
    batch_labels = adata.obs[batch_key].values
    n_features = adata.shape[1]

    f_cluster_list = []
    batch_stats = []

    for batch in batches:
        mask = batch_labels == batch
        n_cells = int(mask.sum())
        X_batch = X_rank[mask]

        if sp.issparse(X_batch):
            mean = np.asarray(X_batch.mean(axis=0)).ravel().astype(np.float64)
            mean_sq = np.asarray(X_batch.power(2).mean(axis=0)).ravel().astype(np.float64)
            var = np.maximum(mean_sq - mean ** 2, 0)
            X_dense = X_batch.toarray().astype(np.float32)
        else:
            X_dense = np.asarray(X_batch, dtype=np.float32)
            mean = np.mean(X_dense, axis=0).astype(np.float64)
            var = np.var(X_dense, axis=0, ddof=0).astype(np.float64)

        batch_stats.append({"mean": mean, "var": var, "n": n_cells})

        std = np.sqrt(var).astype(np.float32)
        std[std == 0] = 1.0
        X_zscore = (X_dense - mean.astype(np.float32)) / std
        del X_dense, X_batch
        gc.collect()

        nc = min(50, X_zscore.shape[1] - 1, X_zscore.shape[0] - 1)
        ad_batch = sc.AnnData(X_zscore)
        sc.pp.pca(ad_batch, n_comps=nc)
        sc.pp.neighbors(ad_batch, n_pcs=nc)
        sc.tl.leiden(ad_batch, key_added="cluster", resolution=1, random_state=seed)

        clusters = np.unique(ad_batch.obs["cluster"].values)
        if len(clusters) < 2:
            f_cluster_list.append(np.zeros(n_features, dtype=np.float32))
        else:
            groups = [X_zscore[ad_batch.obs["cluster"].values == c] for c in clusters]
            f_cluster, _ = scipy.stats.f_oneway(*groups)
            f_cluster_list.append(np.nan_to_num(f_cluster).astype(np.float32))

        del X_zscore, ad_batch
        gc.collect()

    total_n = sum(s["n"] for s in batch_stats)
    grand_mean = sum(s["mean"] * s["n"] for s in batch_stats) / total_n
    ss_between = sum(s["n"] * (s["mean"] - grand_mean) ** 2 for s in batch_stats)
    ss_within = sum(s["var"] * s["n"] for s in batch_stats)
    ms_between = ss_between / max(len(batches) - 1, 1)
    ms_within = ss_within / max(total_n - len(batches), 1)
    ms_within[ms_within == 0] = 1e-10
    f_batch = np.nan_to_num(ms_between / ms_within)

    fc_rank = np.mean([scipy.stats.rankdata(f) for f in f_cluster_list], axis=0)
    fb_rank = scipy.stats.rankdata(f_batch)

    del batch_stats, f_cluster_list
    gc.collect()
    return fc_rank, fb_rank


def compute_entanglement(fc_rank, fb_rank):
    """Spearman correlation between the two ranks (bio-batch entanglement)."""
    rho, _ = scipy.stats.spearmanr(fc_rank, fb_rank)
    return float(rho)


ALPHA = 1.0


def select_features(fc_rank, fb_rank, gamma, n_features):
    """Return the indices of the top n_features by the DECANT score."""
    n = min(int(n_features), len(fc_rank))
    score = fc_rank ** ALPHA / (fb_rank + 1.0) ** gamma
    return np.argsort(score)[-n:]


def run_pca(adata, omics, seed=0):
    """Run PCA on DECANT-selected features.

    RNA selected genes are scaled before PCA following the standard Scanpy
    workflow. Methylation and ATAC follow the original PCA scripts without
    this scaling step. The embedding is stored in ``obsm["X_pca"]``.
    """
    ad = adata.copy()
    if omics == "rna":
        sc.pp.scale(ad)
    sc.tl.pca(ad, random_state=seed)
    return ad, "X_pca"


def compute_tuning_score(adata, emb_key, batch_key, rho, seed=0):
    """Tuning score for gamma selection: cluster_sil - (1 + max(0, rho)) * batch_sil.

    Silhouettes use a fresh Leiden clustering of the embedding; large
    embeddings are subsampled to 50000 cells.
    """
    sc.pp.neighbors(adata, use_rep=emb_key, n_neighbors=15)
    sc.tl.leiden(adata, key_added="_tuning_leiden", resolution=1.0, random_state=seed)

    emb = adata.obsm[emb_key]
    cluster_labels = adata.obs["_tuning_leiden"].values
    batch_labels = adata.obs[batch_key].values

    if len(np.unique(cluster_labels)) < 2:
        return -1.0, np.nan, np.nan

    if len(cluster_labels) > 50000:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(cluster_labels), 50000, replace=False)
        emb = emb[idx]
        cluster_labels = cluster_labels[idx]
        batch_labels = batch_labels[idx]
        if len(np.unique(cluster_labels)) < 2:
            return -1.0, np.nan, np.nan

    cluster_sil = float(sklearn.metrics.silhouette_score(emb, cluster_labels))
    if len(np.unique(batch_labels)) < 2:
        batch_sil = 0.0
    else:
        batch_sil = float(sklearn.metrics.silhouette_score(emb, batch_labels))

    weight = 1.0 + max(0.0, rho)
    score = cluster_sil - weight * batch_sil
    return score, cluster_sil, batch_sil


GAMMA_VALUES = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00)


def run_decant(adata, omics, batch_key, n_features,
               gamma_values=GAMMA_VALUES, seed=0):
    """Run DECANT feature selection with a PCA embedding.

    DECANT is label-free: it ranks features, scans gamma, and keeps the gamma
    with the highest tuning score. Returns the selected features and the PCA
    embedding for that gamma; downstream evaluation is left to the caller
    (see ``evaluate_metrics``).
    """
    omics = omics.lower()
    if omics == "atac":
        values = adata.X.data if sp.issparse(adata.X) else np.asarray(adata.X)
        if not np.isin(values, (0, 1)).all():
            raise ValueError("ATAC input must be binarized (0/1).")
    n_features = min(int(n_features), adata.shape[1])
    n_batches = int(adata.obs[batch_key].nunique())

    fc_rank, fb_rank = compute_feature_ranks(adata, batch_key, omics, seed=seed)
    rho = compute_entanglement(fc_rank, fb_rank)

    tuning_table = []
    for gamma in gamma_values:
        indices = select_features(fc_rank, fb_rank, gamma, n_features)
        embedding, emb_key = run_pca(adata[:, indices], omics, seed=seed)
        score, cluster_sil, batch_sil = compute_tuning_score(
            embedding, emb_key, batch_key, rho, seed=seed)
        tuning_table.append({
            "gamma": float(gamma),
            "alpha": ALPHA,
            "beta": float(gamma),
            "n_features": n_features,
            "tune_score": float(score),
            "cluster_sil": float(cluster_sil),
            "batch_sil": float(batch_sil),
            "entanglement_rho": rho,
        })
        del embedding
        gc.collect()

    adaptive = max(tuning_table, key=lambda row: row["tune_score"])
    adaptive_gamma = adaptive["gamma"]

    selected_indices = select_features(fc_rank, fb_rank, adaptive_gamma, n_features)
    embedding, emb_key = run_pca(adata[:, selected_indices], omics, seed=seed)

    return {
        "adaptive_gamma": adaptive_gamma,
        "rho": rho,
        "n_features": n_features,
        "n_batches": n_batches,
        "tuning_table": tuning_table,
        "embedding": embedding,
        "selected_indices": selected_indices,
    }
