"""The scATAC TF-IDF transform and the pairwise scDNAm preprocessing."""

import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


def tfidf_for_ranking(count_matrix):
    """TF-IDF transform used only for ATAC ranking."""
    if not sp.issparse(count_matrix):
        count_matrix = sp.coo_matrix(count_matrix)

    n_peaks = count_matrix.shape[1]
    peak_sums = count_matrix.sum(axis=0)
    cell_sums = count_matrix.sum(axis=1)

    term_freq = count_matrix.multiply(1.0 / peak_sums)
    inv_doc_freq = np.log(1 + 1.0 * n_peaks / cell_sums).reshape(-1, 1)
    return term_freq.multiply(inv_doc_freq).tocsr()


def tfidf_for_visualization(count_mat):
    """TF-IDF transform used only for PCA/UMAP visualization."""
    if not sp.issparse(count_mat):
        count_mat = sp.coo_matrix(count_mat)
    count_mat = count_mat.tocoo()
    transposed = count_mat.T
    nfreqs = transposed.multiply(1.0 / transposed.sum(axis=0))
    tfidf_mat = nfreqs.multiply(
        np.log(1 + 1.0 * transposed.shape[1] / transposed.sum(axis=1)).reshape(-1, 1)
    ).tocsr()
    return tfidf_mat.T.tocsr()


def preprocess_pair_scdnam_h5ad(
    input_h5ads,
    output_dir,
    output_filenames=None,
    target_col="MajorType",
    batch_labels=("donor1", "donor2"),
    filter_rule="filtered_intersection_gt10",
    write_combined_h5ad=None,
    rebuild_windows=False,
    overwrite=False,
    col_chunk=2048,
):
    """Preprocess a pair of raw scDNAm h5ad files for DECANT.

    The pair is filtered to shared cell types, genomic bins missing in every
    retained cell are dropped, and remaining NaNs are filled by the pair-wide
    per-bin median. DECANT uses the resulting ``adata.X``.
    """
    if rebuild_windows:
        raise NotImplementedError("rebuild_windows is not implemented; DECANT uses adata.X")
    col_chunk = int(col_chunk)

    input_paths = [Path(p).expanduser().resolve() for p in input_h5ads]
    output_dir = Path(output_dir).expanduser().resolve()
    if output_filenames is None:
        output_filenames = [f"{p.stem}_{target_col}.h5ad" for p in input_paths]
    output_paths = [output_dir / name for name in output_filenames]
    combined_path = None if write_combined_h5ad is None else Path(write_combined_h5ad).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_list, var_list, shapes = _read_pair_obs_var(input_paths)
    if not var_list[0].equals(var_list[1]):
        raise ValueError("input h5ad files must have the same var_names in the same order")

    row_idx, obs_out, kept_types = _filtered_rows(obs_list, target_col, filter_rule, batch_labels)
    keep_mask, medians = _compute_keep_mask_and_medians(
        input_paths, row_idx, shapes[0][1], col_chunk
    )
    var_names = var_list[0][keep_mask].astype(str).tolist()

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and path.exists():
            path.unlink()
    if combined_path is not None:
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and combined_path.exists():
            combined_path.unlink()

    for out_path, obs in zip(output_paths, obs_out):
        _write_placeholder_h5ad(out_path, obs, target_col)
        _replace_var_and_x(out_path, var_names, obs.shape[0])

    if combined_path is not None:
        combined_obs = pd.concat(obs_out, axis=0)
        _write_placeholder_h5ad(combined_path, combined_obs, target_col)
        _replace_var_and_x(combined_path, var_names, combined_obs.shape[0])

    _fill_output_matrices(output_paths, combined_path, input_paths, row_idx, keep_mask, medians, col_chunk)

    manifest = _build_manifest(
        input_paths=input_paths,
        output_paths=output_paths,
        combined_path=combined_path,
        target_col=target_col,
        filter_rule=filter_rule,
        kept_types=kept_types,
        row_idx=row_idx,
        n_vars=len(var_names),
        batch_labels=batch_labels,
        rebuild_windows=rebuild_windows,
    )
    manifest_path = output_dir / "manifest.json"
    if overwrite and manifest_path.exists():
        manifest_path.unlink()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def _read_pair_obs_var(paths):
    obs_list = []
    var_list = []
    shapes = []
    for path in paths:
        a = ad.read_h5ad(path, backed="r")
        try:
            obs_list.append(a.obs.copy())
            var_list.append(pd.Index(a.var_names.astype(str)))
            shapes.append(a.shape)
        finally:
            a.file.close()
    return obs_list, var_list, shapes


def _filtered_rows(obs_list, target_col, filter_rule, batch_labels):
    labels = []
    valid = []
    for obs in obs_list:
        raw = obs[target_col]
        ok = raw.notna().to_numpy()
        labels.append(raw.astype(str))
        valid.append(ok)

    if filter_rule == "filtered_intersection_gt10":
        kept_sets = []
        for lab, ok in zip(labels, valid):
            counts = lab[ok].value_counts()
            kept_sets.append(set(counts[counts > 10].index.astype(str)))
        kept_types = sorted(kept_sets[0] & kept_sets[1])
    else:
        kept_types = sorted(set(labels[0][valid[0]]) & set(labels[1][valid[1]]))

    row_idx = []
    obs_out = []
    for obs, lab, ok, batch_label in zip(obs_list, labels, valid, batch_labels):
        mask = ok & lab.isin(kept_types).to_numpy()
        rows = np.flatnonzero(mask)
        filtered = obs.iloc[rows].copy()
        filtered["batch"] = str(batch_label)
        if "data_batch" not in filtered.columns:
            filtered["data_batch"] = str(batch_label)
        row_idx.append(rows)
        obs_out.append(filtered)
    return row_idx, obs_out, kept_types


def _read_x_block(path, rows, start, stop):
    with h5py.File(path, "r") as handle:
        x = handle["X"]
        if isinstance(x, h5py.Dataset):
            return np.asarray(x[rows, start:stop], dtype=np.float64)

    a = ad.read_h5ad(path, backed="r")
    try:
        block = a.X[rows, start:stop]
        if sp.issparse(block):
            block = block.toarray()
        return np.asarray(block, dtype=np.float64)
    finally:
        a.file.close()


def _compute_keep_mask_and_medians(input_paths, row_idx, n_vars, col_chunk):
    keep_mask = np.zeros(n_vars, dtype=bool)
    medians = np.full(n_vars, np.nan, dtype=np.float64)
    for start in range(0, n_vars, col_chunk):
        stop = min(start + col_chunk, n_vars)
        blocks = [
            _read_x_block(path, rows, start, stop)
            for path, rows in zip(input_paths, row_idx)
        ]
        concat = np.vstack(blocks)
        keep = ~np.all(np.isnan(concat), axis=0)
        if keep.any():
            medians[start:stop][keep] = np.nanmedian(concat[:, keep], axis=0)
            keep_mask[start:stop] = keep
    return keep_mask, medians


def _write_placeholder_h5ad(path, obs, target_col):
    tmp = ad.AnnData(
        X=np.zeros((obs.shape[0], 1), dtype=np.float64),
        obs=obs,
        var=pd.DataFrame(index=["placeholder"]),
    )
    tmp.uns["preprocess_note"] = (
        "scDNAm pair preprocessing: shared cell-type filtering, pair-wide "
        "non-all-NaN genomic-bin retention, and pair-wide nanmedian fill. "
        "DECANT uses adata.X; legacy window matrices in obsm are not required."
    )
    tmp.uns["target_col"] = target_col
    tmp.write_h5ad(path)


def _replace_var_and_x(path, var_names, n_obs):
    with h5py.File(path, "a") as handle:
        if "X" in handle:
            del handle["X"]
        x = handle.create_dataset(
            "X",
            shape=(n_obs, len(var_names)),
            dtype="float64",
            chunks=(min(128, max(n_obs, 1)), min(2048, len(var_names))),
            compression=None,
        )
        x.attrs["encoding-type"] = "array"
        x.attrs["encoding-version"] = "0.2.0"
        if "var" in handle:
            del handle["var"]
        var = handle.create_group("var")
        var.attrs["_index"] = "_index"
        var.attrs["column-order"] = np.array([], dtype=float)
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["encoding-version"] = "0.2.0"
        dt = h5py.string_dtype(encoding="utf-8")
        ds = var.create_dataset("_index", data=np.asarray(var_names, dtype=object), dtype=dt)
        ds.attrs["encoding-type"] = "string-array"
        ds.attrs["encoding-version"] = "0.2.0"


def _fill_output_matrices(output_paths, combined_path, input_paths, row_idx, keep_mask, medians, col_chunk):
    out_handles = [h5py.File(path, "a") for path in output_paths]
    combined_handle = h5py.File(combined_path, "a") if combined_path is not None else None
    try:
        out_x = [handle["X"] for handle in out_handles]
        combined_x = None if combined_handle is None else combined_handle["X"]
        combined_offsets = np.cumsum([0] + [len(rows) for rows in row_idx])
        out_col = 0
        for start in range(0, keep_mask.size, col_chunk):
            stop = min(start + col_chunk, keep_mask.size)
            local_keep = keep_mask[start:stop]
            if not local_keep.any():
                continue
            med = medians[start:stop][local_keep]
            n_keep = int(local_keep.sum())
            for i, (path, rows) in enumerate(zip(input_paths, row_idx)):
                block = _read_x_block(path, rows, start, stop)[:, local_keep]
                nan_mask = np.isnan(block)
                if nan_mask.any():
                    block[nan_mask] = np.take(med, np.where(nan_mask)[1])
                out_x[i][:, out_col:out_col + n_keep] = block
                if combined_x is not None:
                    r0, r1 = combined_offsets[i], combined_offsets[i + 1]
                    combined_x[r0:r1, out_col:out_col + n_keep] = block
            out_col += n_keep
    finally:
        for handle in out_handles:
            handle.close()
        if combined_handle is not None:
            combined_handle.close()


def _build_manifest(
    input_paths,
    output_paths,
    combined_path,
    target_col,
    filter_rule,
    kept_types,
    row_idx,
    n_vars,
    batch_labels,
    rebuild_windows,
):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "function": "preprocess_pair_scdnam_h5ad",
        "input_paths": [str(p) for p in input_paths],
        "inputs": [_file_record(p) for p in input_paths],
        "output_paths": [str(p) for p in output_paths],
        "outputs": [
            {
                "path": str(path),
                "n_obs": int(len(rows)),
                "n_vars": int(n_vars),
                "batch_label": str(batch),
            }
            for path, rows, batch in zip(output_paths, row_idx, batch_labels)
        ],
        "combined_output_path": None if combined_path is None else str(combined_path),
        "target_col": target_col,
        "filter_rule": filter_rule,
        "kept_cell_types": [str(x) for x in kept_types],
        "total_n_obs": int(sum(len(rows) for rows in row_idx)),
        "total_n_vars": int(n_vars),
        "batch_labels": [str(x) for x in batch_labels],
        "rebuild_windows": bool(rebuild_windows),
    }


def _file_record(path):
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
