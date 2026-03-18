"""Data processing pipeline: SMILES → Fingerprints → BitBIRCH Clustering → PCA.

Fixes applied vs original:
  1. Import centroid_from_sum from public API (bblean.fingerprints), not _py_similarity
  2. Batch fps_from_smiles calls instead of one-SMILES-at-a-time
  3. Remove misleading tolerance param when using diameter criterion
  4. Pass n_features explicitly to BitBirch.fit()
  5. Cast get_assignments() to int64 for consistency with parallel path
  6. np.vstack get_centroids() return (list → 2D array)
  7. Fix column name 'Cluster' → 'cluster' in _compute_incremental_pca
  8. Split fingerprints into multiple files for run_multiround_bitbirch
  9. Don't store unfitted BitBirch after parallel clustering
 10. Count clusters from assignments array, not get_cluster_mol_ids()
 11. Build cluster_to_indices once, not redundantly in fallback chains
 12. Filter -1 assignments before computing cluster_sizes
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from tqdm.auto import tqdm
import threading
from multiprocessing import Pool
from functools import partial
import gc
import os
import math

# --- bblean public API imports ---
from bblean import BitBirch
from bblean.fingerprints import (
    fps_from_smiles,
    unpack_fingerprints,
    centroid_from_sum,       # FIX 1: public API, not bblean._py_similarity
)


# ---------------------------------------------------------------------------
# Module-level worker functions (must be picklable for multiprocessing.Pool)
# ---------------------------------------------------------------------------

def _process_smiles_batch(smiles_batch_with_idx, fp_kind, nbits):
    """Worker: generate packed fingerprints for a batch of SMILES.

    FIX 2: Calls fps_from_smiles once per batch (not per-SMILES), so the
    RDKit generator object is created only once.  Invalid SMILES are tracked
    via skip_invalid=True and post-filtered.
    """
    batch_idx, smiles_list = smiles_batch_with_idx

    try:
        # One call for the whole batch — generator created once
        fps = fps_from_smiles(
            smiles_list,
            kind=fp_kind,
            n_features=nbits,
            pack=True,
            skip_invalid=True,      # silently skip bad SMILES
            sanitize='minimal',
        )
    except Exception:
        fps = np.empty((0, (nbits + 7) // 8), dtype=np.uint8)

    # Because skip_invalid=True silently drops rows, we need to figure out
    # which local indices survived.  Re-validate quickly to build the mask.
    valid_local_indices = []
    if len(fps) == len(smiles_list):
        # All valid — fast path
        valid_local_indices = list(range(len(smiles_list)))
    elif len(fps) > 0:
        # Some were dropped — re-check individually (lightweight, no FP regen)
        from rdkit.Chem import MolFromSmiles
        for local_idx, smi in enumerate(smiles_list):
            mol = MolFromSmiles(smi, sanitize=False)
            if mol is not None:
                valid_local_indices.append(local_idx)
                if len(valid_local_indices) == len(fps):
                    break  # found them all

    return fps, batch_idx, valid_local_indices


# --- Shared data for centroid worker processes (set via Pool initializer) ---
_shared_X_packed = None
_shared_cluster_to_indices = None
_shared_nbits = None


def _centroid_pool_init(X_packed, cluster_to_indices, nbits):
    """Pool initializer: store read-only data as globals (COW on fork)."""
    global _shared_X_packed, _shared_cluster_to_indices, _shared_nbits
    _shared_X_packed = X_packed
    _shared_cluster_to_indices = cluster_to_indices
    _shared_nbits = nbits


def _compute_centroids_for_ids(cluster_ids_batch):
    """Worker: compute majority-vote centroids for a batch of cluster IDs.

    Returns a list of (cluster_id, centroid_array) tuples so that the caller
    can verify alignment.  FIX 7-partial: returning IDs prevents the silent
    length-mismatch bug when a cluster ID has no molecules.
    """
    results = []
    for cluster_id in cluster_ids_batch:
        indices = _shared_cluster_to_indices.get(cluster_id)
        if indices is None or len(indices) == 0:
            continue
        cluster_packed = _shared_X_packed[indices]
        cluster_unpacked = unpack_fingerprints(cluster_packed, n_features=_shared_nbits)
        linear_sum = np.sum(cluster_unpacked, axis=0, dtype=np.uint64)
        centroid = centroid_from_sum(linear_sum, len(cluster_unpacked), pack=False)
        results.append((cluster_id, centroid))
        del cluster_unpacked
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _available_cores():
    """Number of CPU cores available to this process (respects cgroups/SLURM)."""
    return max(1, len(os.sched_getaffinity(0)))


def _build_cluster_to_indices(assignments):
    """Single O(n) pass: cluster_id → np.array of molecule indices."""
    mapping = {}
    for idx, cid in enumerate(assignments):
        cid = int(cid)
        if cid < 0:
            continue  # FIX 12: skip unassigned (-1)
        mapping.setdefault(cid, []).append(idx)
    return {cid: np.array(v, dtype=np.int64) for cid, v in mapping.items()}


def _count_clusters(assignments):
    """Count valid (>= 0) unique cluster IDs from an assignments array."""
    return len(np.unique(assignments[assignments >= 0]))


def _cluster_sizes_series(assignments):
    """Cluster size Series indexed by cluster ID, excluding -1.  FIX 12."""
    valid = assignments[assignments >= 0]
    return pd.Series(valid).value_counts().sort_index()


def _parallel_compute_centroids(X_packed, cluster_ids, cluster_to_indices, nbits):
    """Compute centroids for *cluster_ids* using a multiprocessing pool.

    Returns (centroid_array, actual_cluster_ids) where actual_cluster_ids may
    be shorter than the input if some IDs had no molecules.
    """
    n_cores = max(1, _available_cores() - 1)
    batch_size = max(10, len(cluster_ids) // (n_cores * 20))
    batches = [cluster_ids[i:i + batch_size]
               for i in range(0, len(cluster_ids), batch_size)]
    chunksize = max(1, len(batches) // (n_cores * 4))

    with Pool(n_cores, initializer=_centroid_pool_init,
              initargs=(X_packed, cluster_to_indices, nbits)) as pool:
        batch_results = pool.map(_compute_centroids_for_ids, batches,
                                 chunksize=chunksize)

    # Flatten and separate IDs from arrays
    ids_out, centroids_out = [], []
    for batch in batch_results:
        for cid, centroid in batch:
            ids_out.append(cid)
            centroids_out.append(centroid)

    if not centroids_out:
        return np.empty((0, nbits), dtype=np.uint8), []

    return np.array(centroids_out, dtype=np.uint8), ids_out


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Data_Processing:
    def __init__(self, parent_app):
        self.app = parent_app

    # --- Cluster size helpers (used by GUI) --------------------------------

    def get_min_cluster_size(self):
        try:
            return int(self.app.min_cluster_size_var.get())
        except ValueError:
            return 10

    def get_max_cluster_size(self):
        try:
            value = int(self.app.max_cluster_size_var.get())
            return float('inf') if value <= 0 else value
        except ValueError:
            return 1000

    def is_cluster_in_overview_range(self, size):
        return self.get_min_cluster_size() <= size <= self.get_max_cluster_size()

    # --- File I/O ----------------------------------------------------------

    def load_smiles_file(self):
        file_path = filedialog.askopenfilename(
            title="Select SMILES CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not file_path:
            return

        self.app.gui.progress.start()
        self.app.gui.file_label.config(text="Loading file...")
        thread = threading.Thread(target=self._load_file_bg, args=(file_path,),
                                  daemon=True)
        thread.start()

    def _load_file_bg(self, file_path):
        """Background thread: load CSV and auto-detect format."""
        import time
        try:
            t0 = time.time()
            sample = pd.read_csv(file_path, nrows=5, header=None)

            if len(sample.columns) == 1:
                df = pd.read_csv(file_path, sep=" ", names=["SMILES", "Name"],
                                 engine='c', low_memory=False)
            elif len(sample.columns) == 2:
                df = pd.read_csv(file_path, names=["SMILES", "Name"],
                                 engine='c', low_memory=False)
            else:
                df = pd.read_csv(file_path, engine='c', low_memory=False)
                if 'smiles' in df.columns:
                    df.rename(columns={'smiles': 'SMILES'}, inplace=True)
                elif 'SMILES' not in df.columns:
                    cols = list(df.columns)
                    df.rename(columns={cols[0]: 'SMILES'}, inplace=True)
                    if len(cols) > 1:
                        df.rename(columns={cols[1]: 'Name'}, inplace=True)

            self.app.data = df
            elapsed = time.time() - t0
            self.app.root.after(0, lambda: self._finish_file_load(file_path, elapsed))
        except Exception as e:
            self.app.root.after(0, lambda: self._file_load_error(str(e)))

    def _finish_file_load(self, file_path, elapsed):
        self.app.gui.progress.stop()
        self.app.gui.file_label.config(text=f"Loaded: {os.path.basename(file_path)}")
        self.app.gui.process_btn.config(state="normal")
        self.app.display_data_info()
        n = len(self.app.data)
        messagebox.showinfo("Success",
                            f"Loaded {n:,} rows in {elapsed:.2f}s "
                            f"({n / elapsed:,.0f} rows/s)")

    def _file_load_error(self, msg):
        self.app.gui.progress.stop()
        self.app.gui.file_label.config(text="Load failed")
        messagebox.showerror("Error", f"Failed to load file:\n{msg}")

    # --- Save / Load results -----------------------------------------------

    def save_results(self):
        if self.app.data is None or self.app.cluster_assignments is None:
            messagebox.showwarning("Warning", "No clustering results to save!")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Clustering Results", defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
        )
        if not save_path:
            return

        try:
            save_data = {
                'smiles': self.app.data['SMILES'].tolist(),
                'cluster_assignments': self.app.cluster_assignments.tolist(),
                'parameters': {
                    'threshold': self.app.threshold_var.get(),
                    'branching_factor': self.app.branching_var.get(),
                    'radius': self.app.radius_var.get(),
                    'nbits': self.app.nbits_var.get(),
                    'min_cluster_size': self.app.min_cluster_size_var.get(),
                    'max_cluster_size': self.app.max_cluster_size_var.get(),
                    'hide_singletons': self.app.hide_singletons_var.get(),
                    'use_parallel_clustering':
                        self.app.use_parallel_clustering_var.get(),
                    'parallel_num_processes':
                        self.app.parallel_num_processes_var.get(),
                    'use_incremental_pca':
                        self.app.use_incremental_pca_var.get(),
                },
            }

            # Molecule PCA (column is always lowercase 'cluster' — FIX 7)
            if (hasattr(self.app, 'molecule_pca')
                    and self.app.molecule_pca is not None):
                save_data['molecule_pca'] = {
                    'PC1': self.app.molecule_pca['PC1'].tolist(),
                    'PC2': self.app.molecule_pca['PC2'].tolist(),
                    'cluster': self.app.molecule_pca['cluster'].tolist(),
                }

            # Centroid PCA
            if (hasattr(self.app, 'centroid_pca')
                    and self.app.centroid_pca is not None
                    and not self.app.centroid_pca.empty):
                save_data['centroid_pca'] = {
                    'PC1': self.app.centroid_pca['PC1'].tolist(),
                    'PC2': self.app.centroid_pca['PC2'].tolist(),
                    'cluster': self.app.centroid_pca['cluster'].tolist(),
                    'size': self.app.centroid_pca['size'].tolist(),
                    'hover': self.app.centroid_pca['hover'].tolist(),
                }

            if 'Name' in self.app.data.columns:
                save_data['names'] = self.app.data['Name'].tolist()

            np.save(save_path, save_data, allow_pickle=True)

            saved = ["SMILES", "cluster assignments", "parameters"]
            if 'molecule_pca' in save_data:
                saved.append("molecule PCA")
            if 'centroid_pca' in save_data:
                saved.append("centroid PCA")

            # FIX 14-partial: warn user about lost on-demand PCA capability
            note = ""
            if self.app.X is not None and 'molecule_pca' not in save_data:
                note = ("\n\nNote: Fingerprints are NOT saved. "
                        "Per-cluster detail PCA will be unavailable after reload.")

            messagebox.showinfo("Success",
                                f"Saved: {', '.join(saved)}{note}")
        except Exception as e:
            import traceback
            messagebox.showerror("Error",
                                 f"Save failed:\n{e}\n\n{traceback.format_exc()}")

    def load_results(self):
        load_path = filedialog.askopenfilename(
            title="Load Clustering Results",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
        )
        if not load_path:
            return

        try:
            loaded = np.load(load_path, allow_pickle=True).item()

            if not all(k in loaded for k in ('smiles', 'cluster_assignments')):
                messagebox.showerror("Error", "Invalid save file format!")
                return

            self.app.data = pd.DataFrame({'SMILES': loaded['smiles']})
            if 'names' in loaded:
                self.app.data['Name'] = loaded['names']

            self.app.cluster_assignments = np.array(
                loaded['cluster_assignments'], dtype=np.int64)
            self.app.data['cluster'] = self.app.cluster_assignments

            # Restore molecule PCA
            if 'molecule_pca' in loaded:
                d = loaded['molecule_pca']
                self.app.molecule_pca = pd.DataFrame({
                    'PC1': d['PC1'], 'PC2': d['PC2'], 'cluster': d['cluster'],
                })
            else:
                self.app.molecule_pca = None

            # Restore centroid PCA
            if 'centroid_pca' in loaded:
                d = loaded['centroid_pca']
                self.app.centroid_pca = pd.DataFrame({
                    'PC1': d['PC1'], 'PC2': d['PC2'],
                    'cluster': d['cluster'], 'size': d['size'],
                    'hover': d['hover'],
                })
            else:
                self.app.centroid_pca = pd.DataFrame()

            self.app.X = None
            self.app.centroids = None
            self.app.brc = None

            if 'parameters' in loaded:
                p = loaded['parameters']
                self.app.threshold_var.set(str(p.get('threshold', 0.65)))
                self.app.branching_var.set(str(p.get('branching_factor', 50)))
                self.app.radius_var.set(str(p.get('radius', 2)))
                self.app.nbits_var.set(str(p.get('nbits', 2048)))
                for key, var in [
                    ('min_cluster_size', self.app.min_cluster_size_var),
                    ('max_cluster_size', self.app.max_cluster_size_var),
                    ('parallel_num_processes', self.app.parallel_num_processes_var),
                ]:
                    if key in p:
                        var.set(str(p[key]))
                for key, var in [
                    ('hide_singletons', self.app.hide_singletons_var),
                    ('use_parallel_clustering',
                     self.app.use_parallel_clustering_var),
                    ('use_incremental_pca', self.app.use_incremental_pca_var),
                ]:
                    if key in p:
                        var.set(bool(p[key]))

            self.app.gui.file_label.config(
                text=f"Loaded: {os.path.basename(load_path)}")
            self.app.gui.save_btn.config(state="normal")
            self.app.gui.refresh_btn.config(state="normal")
            self.app.display_data_info()
            self.app.show_overview()
            self.app.visualizer.display_clustering_results()

            n_clusters = _count_clusters(self.app.cluster_assignments)
            messagebox.showinfo(
                "Success",
                f"Loaded {len(self.app.data):,} molecules, "
                f"{n_clusters:,} clusters")

        except Exception as e:
            messagebox.showerror("Error", f"Load failed:\n{e}")

    # --- Main processing pipeline ------------------------------------------

    def start_processing(self):
        self.app.gui.progress.config(mode='determinate', value=0, maximum=100)
        self.app.gui.process_btn.config(state="disabled")
        thread = threading.Thread(target=self._process_data, daemon=True)
        thread.start()

    def _log(self, msg):
        """Thread-safe log to the results text widget."""
        self.app.root.after(
            0, lambda: self.app.gui.results_text.insert(tk.END, msg + "\n"))

    def _set_progress(self, value):
        self.app.root.after(
            0, lambda v=value: self.app.gui.progress.config(value=v))

    def _process_data(self):
        import time
        try:
            # ---- Read GUI parameters ----
            nbits = int(self.app.nbits_var.get())
            radius = int(self.app.radius_var.get())
            threshold = float(self.app.threshold_var.get())
            branching = int(self.app.branching_var.get())
            fp_kind = {2: 'ecfp4', 3: 'ecfp6'}.get(radius, 'rdkit')

            # ==============================================================
            # STAGE 1: Fingerprint generation  (0 – 60 %)
            # ==============================================================
            self._log("[Stage 1/3] Generating fingerprints (parallel)...")
            t0 = time.time()

            smiles_list = self.app.data['SMILES'].tolist()
            n_samples = len(smiles_list)
            n_cores = max(1, _available_cores() - 1)

            # Adaptive batch size: ≥4 batches/core, capped at 1000
            batch_size = min(1000, max(100,
                             n_samples // (n_cores * 4)))

            # FIX 2: each batch is processed as a whole by fps_from_smiles
            batches = [
                (i, smiles_list[i * batch_size:(i + 1) * batch_size])
                for i in range((n_samples + batch_size - 1) // batch_size)
            ]
            self._log(f"  {n_cores} cores, {len(batches)} batches "
                       f"(size={batch_size}), kind={fp_kind}")

            worker = partial(_process_smiles_batch,
                             fp_kind=fp_kind, nbits=nbits)
            chunksize = max(1, len(batches) // (n_cores * 2))

            fps_list = []
            valid_indices = []

            with Pool(n_cores) as pool:
                for fps, b_idx, local_valid in pool.imap(
                        worker, batches, chunksize=chunksize):
                    if len(fps) > 0:
                        fps_list.append(fps)
                        start = b_idx * batch_size
                        valid_indices.extend(start + li for li in local_valid)
                    pct = int((b_idx + 1) / len(batches) * 60)
                    self._set_progress(pct)

            if not fps_list:
                raise ValueError("No valid fingerprints generated")

            self.app.X = np.vstack(fps_list)
            del fps_list

            # Filter DataFrame to valid-only rows
            if len(valid_indices) < n_samples:
                dropped = n_samples - len(valid_indices)
                self._log(f"  Dropped {dropped:,} invalid SMILES")
                self.app.data = (self.app.data.iloc[valid_indices]
                                 .reset_index(drop=True))

            fp_time = time.time() - t0
            mb = self.app.X.nbytes / 1024**2
            self._log(f"  Packed FP matrix: {self.app.X.shape} "
                       f"({mb:.1f} MB) in {fp_time:.1f}s")
            self._set_progress(60)

            # ==============================================================
            # STAGE 2: BitBIRCH clustering  (60 – 80 %)
            # ==============================================================
            self._log("[Stage 2/3] BitBIRCH clustering...")
            t1 = time.time()
            n_mols = len(self.app.X)
            use_parallel = (self.app.use_parallel_clustering_var.get()
                            and n_mols > 100_000)

            if use_parallel:
                self._log("  Using parallel multiround clustering...")
                assignments, num_clusters = self._parallel_clustering(nbits)
            else:
                assignments, num_clusters = self._serial_clustering(
                    nbits, threshold, branching)

            # FIX 5: always store as int64 for consistency
            self.app.cluster_assignments = assignments.astype(np.int64)
            self.app.data['cluster'] = self.app.cluster_assignments

            cluster_time = time.time() - t1
            self._log(f"  {num_clusters:,} clusters in {cluster_time:.1f}s")
            self._set_progress(80)

            # ==============================================================
            # STAGE 3: PCA  (80 – 100 %)
            # ==============================================================
            self._log("[Stage 3/3] Computing PCA...")

            # FIX 12: exclude -1 from sizes
            cluster_sizes = _cluster_sizes_series(self.app.cluster_assignments)

            # FIX 11: build index map once
            cluster_to_indices = _build_cluster_to_indices(
                self.app.cluster_assignments)

            self._compute_centroid_pca(
                cluster_sizes, cluster_to_indices, num_clusters, nbits)
            self._set_progress(90)

            self._compute_molecule_pca(nbits)
            self._set_progress(100)

            gc.collect()
            self.app.root.after(0, self._finish_processing)

        except Exception as e:
            import traceback
            self._log(f"ERROR: {e}\n{traceback.format_exc()}")
            self.app.root.after(
                0, lambda: messagebox.showerror("Error", str(e)))
            self._set_progress(0)
            self.app.root.after(
                0, lambda: self.app.gui.process_btn.config(state="normal"))

    # --- Stage 2 helpers ---------------------------------------------------

    def _serial_clustering(self, nbits, threshold, branching):
        """Standard single-process BitBIRCH clustering.

        FIX 3: don't pass tolerance with diameter (it's ignored).
        FIX 4: pass n_features explicitly.
        FIX 5: cast to int64.
        FIX 10: count clusters from assignments, not get_cluster_mol_ids().
        """
        brc = BitBirch(
            branching_factor=branching,
            threshold=threshold,
            merge_criterion='diameter',
            # tolerance omitted — not used by DiameterMerge  (FIX 3)
        )
        brc.fit(self.app.X, input_is_packed=True, n_features=nbits)  # FIX 4
        self.app.brc = brc

        # get_assignments returns uint64 starting at 1; convert to 0-indexed int64
        assignments = brc.get_assignments(sort=True).astype(np.int64) - 1  # FIX 5
        num_clusters = _count_clusters(assignments)  # FIX 10
        return assignments, num_clusters

    def _parallel_clustering(self, nbits):
        """Multiround parallel clustering for large datasets.

        FIX 8: split fingerprints into multiple .npy files so that
        run_multiround_bitbirch can actually distribute work.
        FIX 9: don't store an unfitted BitBirch as self.app.brc.
        """
        import tempfile
        import shutil
        import pickle
        from pathlib import Path
        from bblean.multiround import run_multiround_bitbirch

        num_processes = int(self.app.parallel_num_processes_var.get())
        threshold = float(self.app.threshold_var.get())
        branching = int(self.app.branching_var.get())
        n_mols = len(self.app.X)

        temp_dir = Path(tempfile.mkdtemp(prefix="nami_multiround_"))
        try:
            # --- FIX 8: split into multiple files for real parallelism ---
            # Target: one file per initial-round process so each gets its own
            n_files = max(1, min(num_processes, n_mols // 10_000))
            chunk = math.ceil(n_mols / n_files)
            fp_files = []
            for i in range(n_files):
                start, end = i * chunk, min((i + 1) * chunk, n_mols)
                fp_path = temp_dir / f"fps_{i:04d}.npy"
                np.save(fp_path, self.app.X[start:end])
                fp_files.append(fp_path)

            self._log(f"  Split into {n_files} files for {num_processes} "
                       f"processes")

            run_multiround_bitbirch(
                input_files=fp_files,
                out_dir=temp_dir,
                n_features=nbits,
                input_is_packed=True,
                num_initial_processes=num_processes,
                num_midsection_processes=max(2, num_processes // 2),
                branching_factor=branching,
                threshold=threshold,
                tolerance=0.05,
                num_midsection_rounds=1,
                verbose=False,
                cleanup=False,
            )

            # --- Load cluster results ---
            clusters_file = temp_dir / "clusters.pkl"
            if not clusters_file.exists():
                found = [f.name for f in temp_dir.glob("*")]
                raise FileNotFoundError(
                    f"clusters.pkl not found. Files: {found}")

            with open(clusters_file, 'rb') as f:
                mol_ids = pickle.load(f)

            # mol_ids is list-of-lists: mol_ids[cluster_idx] = [mol_indices]
            assignments = np.full(n_mols, -1, dtype=np.int64)
            for cluster_id, indices in enumerate(mol_ids):
                for mol_idx in indices:
                    assignments[int(mol_idx)] = cluster_id

            n_assigned = np.sum(assignments >= 0)
            if n_assigned < n_mols:
                self._log(f"  WARNING: {n_mols - n_assigned:,} molecules "
                           f"unassigned")

            num_clusters = _count_clusters(assignments)
            self._log(f"  {num_clusters:,} clusters from multiround")

            # FIX 9: Don't store an unfitted BitBirch — store None instead.
            # Downstream code checks `self.app.brc is not None` before calling
            # tree methods.
            self.app.brc = None

            # Load pre-computed centroids if available
            centroids_file = temp_dir / "cluster-centroids-packed.pkl"
            if centroids_file.exists():
                with open(centroids_file, 'rb') as f:
                    packed_list = pickle.load(f)
                self.app.centroids = unpack_fingerprints(
                    np.array(packed_list, dtype=np.uint8), n_features=nbits)
                self._log(f"  Loaded {len(self.app.centroids)} pre-computed "
                           f"centroids")
            else:
                self.app.centroids = None

            return assignments, num_clusters

        finally:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

    # --- Stage 3 helpers ---------------------------------------------------

    def _compute_centroid_pca(self, cluster_sizes, cluster_to_indices,
                              num_clusters, nbits):
        """Compute PCA over cluster centroids for the overview plot."""
        hide_singletons = self.app.hide_singletons_var.get()

        if hide_singletons:
            target_ids = [cid for cid, sz in cluster_sizes.items() if sz > 1]
        else:
            target_ids = list(cluster_sizes.index)

        if not target_ids:
            self.app.centroid_pca = pd.DataFrame(
                columns=['PC1', 'PC2', 'cluster', 'size', 'hover'])
            self._log("  No clusters to visualize in PCA")
            return

        # Try to use pre-computed centroids (from parallel path)
        centroids_array = None
        actual_ids = target_ids

        if (hasattr(self.app, 'centroids')
                and self.app.centroids is not None
                and len(self.app.centroids) == num_clusters
                and not hide_singletons):
            # FIX 6: centroids from brc.get_centroids() is a list — vstack it
            if isinstance(self.app.centroids, list):
                centroids_array = np.vstack(self.app.centroids)
            else:
                centroids_array = self.app.centroids
        elif self.app.brc is not None and not hide_singletons:
            # Serial path: get centroids directly from tree
            # FIX 6: vstack the list
            try:
                raw = self.app.brc.get_centroids(packed=False, sort=True)
                centroids_array = np.vstack(raw)
            except Exception as e:
                self._log(f"  get_centroids failed: {e}")

        # Fallback: compute centroids from fingerprints
        if centroids_array is None:
            self._log(f"  Computing centroids for {len(target_ids):,} "
                       f"clusters (parallel)...")
            centroids_array, actual_ids = _parallel_compute_centroids(
                self.app.X, target_ids, cluster_to_indices, nbits)

        if len(centroids_array) < 2:
            self.app.centroid_pca = pd.DataFrame(
                columns=['PC1', 'PC2', 'cluster', 'size', 'hover'])
            return

        pca_df = self._run_pca(centroids_array)
        pca_df['cluster'] = actual_ids[:len(pca_df)]
        pca_df['size'] = [int(cluster_sizes.get(cid, 0))
                          for cid in pca_df['cluster']]
        pca_df['hover'] = [f"Cluster {int(cid)}" for cid in pca_df['cluster']]
        self.app.centroid_pca = pca_df
        self._log(f"  Centroid PCA: {len(pca_df)} points")

    def _compute_molecule_pca(self, nbits):
        """Optionally compute PCA over all molecules."""
        if self.app.use_incremental_pca_var.get():
            self._log("  Computing full molecule PCA (IncrementalPCA)...")
            self.app.molecule_pca = self._incremental_pca(nbits)
        else:
            self.app.molecule_pca = None
            self._log("  Skipping full molecule PCA (on-demand per cluster)")

    def _incremental_pca(self, nbits):
        """Memory-efficient PCA over all molecules.

        FIX 7: column name is 'cluster' (lowercase), matching save_results.
        """
        n_mols = len(self.app.X)
        batch_size = 10_000
        n_batches = (n_mols + batch_size - 1) // batch_size
        ipca = IncrementalPCA(n_components=2)

        # Fit pass
        for bi in range(n_batches):
            s, e = bi * batch_size, min((bi + 1) * batch_size, n_mols)
            ipca.partial_fit(unpack_fingerprints(self.app.X[s:e], nbits))

        # Transform pass
        coords = []
        for bi in range(n_batches):
            s, e = bi * batch_size, min((bi + 1) * batch_size, n_mols)
            coords.append(
                ipca.transform(unpack_fingerprints(self.app.X[s:e], nbits)))

        all_coords = np.vstack(coords)
        var = ipca.explained_variance_ratio_
        self._log(f"  Variance explained: PC1={var[0]:.1%}, PC2={var[1]:.1%}")

        return pd.DataFrame({
            'PC1': all_coords[:, 0],
            'PC2': all_coords[:, 1],
            'cluster': self.app.cluster_assignments,  # FIX 7: lowercase
        })

    def _run_pca(self, data):
        """PCA with automatic standard/incremental selection."""
        if len(data) < 2:
            return pd.DataFrame(columns=['PC1', 'PC2'])

        if len(data) > 100_000:
            ipca = IncrementalPCA(n_components=2, batch_size=10_000)
            for i in range(0, len(data), 10_000):
                ipca.partial_fit(data[i:i + 10_000])
            # FIX 17: always batch the transform for large data
            parts = []
            for i in range(0, len(data), 10_000):
                parts.append(ipca.transform(data[i:i + 10_000]))
            result = np.vstack(parts)
        else:
            result = PCA(n_components=2).fit_transform(data)

        return pd.DataFrame(result, columns=['PC1', 'PC2'])

    # --- Finalize ----------------------------------------------------------

    def _finish_processing(self):
        self.app.root.after(
            1000, lambda: self.app.gui.progress.config(value=0))
        self.app.gui.process_btn.config(state="normal")
        self.app.gui.refresh_btn.config(state="normal")
        self.app.gui.save_btn.config(state="normal")
        self.app.display_data_info()
        self.app.show_overview()
        self.app.visualizer.display_clustering_results()
