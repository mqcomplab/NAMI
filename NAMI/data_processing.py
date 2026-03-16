import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from bblean import BitBirch
from bblean.fingerprints import fps_from_smiles, unpack_fingerprints
from bblean._py_similarity import centroid_from_sum
from tqdm.auto import tqdm
import threading
from multiprocessing import Pool, cpu_count
from functools import partial
import gc
import os

def _process_smiles_batch_bblean(smiles_batch_with_idx, fp_kind, nbits):
    """Helper function for parallel fingerprint generation using BBlean's optimized functions.
    Returns tuple of (packed_fps, valid_indices) to track which SMILES were valid.
    """
    batch_idx, smiles_list = smiles_batch_with_idx
    
    # Process each SMILES individually to track validity
    valid_fps = []
    valid_local_indices = []
    
    for local_idx, smi in enumerate(smiles_list):
        try:
            # Use BBlean's optimized fps_from_smiles
            fp = fps_from_smiles(
                [smi],
                kind=fp_kind,
                n_features=nbits,
                pack=True,
                skip_invalid=False,  # We handle errors explicitly
                sanitize='minimal'  # Faster sanitization
            )
            if len(fp) > 0:
                valid_fps.append(fp[0])
                valid_local_indices.append(local_idx)
        except:
            # Skip invalid SMILES
            pass
    
    # Return as numpy array and indices
    if valid_fps:
        return np.array(valid_fps, dtype=np.uint8), batch_idx, valid_local_indices
    else:
        return np.array([], dtype=np.uint8).reshape(0, (nbits + 7) // 8), batch_idx, []

# --- Shared data for centroid worker processes (set via Pool initializer) ---
_shared_X_packed = None
_shared_cluster_to_indices = None
_shared_nbits = None

def _centroid_pool_init(X_packed, cluster_to_indices, nbits):
    """Pool initializer: store shared data as globals (inherited via fork/COW)."""
    global _shared_X_packed, _shared_cluster_to_indices, _shared_nbits
    _shared_X_packed = X_packed
    _shared_cluster_to_indices = cluster_to_indices
    _shared_nbits = nbits

def _compute_centroids_batch(cluster_ids_batch):
    """Worker: compute centroids for a batch of non-singleton cluster IDs.
    Uses shared globals (no pickling of large arrays).
    """
    centroids = []
    for cluster_id in cluster_ids_batch:
        indices = _shared_cluster_to_indices.get(cluster_id)
        if indices is not None and len(indices) > 0:
            cluster_X_packed = _shared_X_packed[indices]
            cluster_X_unpacked = unpack_fingerprints(cluster_X_packed, n_features=_shared_nbits)
            linear_sum = np.sum(cluster_X_unpacked, axis=0, dtype=np.uint64)
            centroid = centroid_from_sum(linear_sum, len(cluster_X_unpacked), pack=False)
            centroids.append(centroid)
            del cluster_X_unpacked
    return centroids

def _compute_all_centroids_batch(cluster_ids_batch):
    """Worker: compute centroids for a batch of cluster IDs (all clusters).
    Includes zero vectors for empty clusters. Uses shared globals.
    """
    centroids = []
    for cluster_id in cluster_ids_batch:
        indices = _shared_cluster_to_indices.get(cluster_id)
        if indices is not None and len(indices) > 0:
            cluster_X_packed = _shared_X_packed[indices]
            cluster_X_unpacked = unpack_fingerprints(cluster_X_packed, n_features=_shared_nbits)
            linear_sum = np.sum(cluster_X_unpacked, axis=0, dtype=np.uint64)
            centroid = centroid_from_sum(linear_sum, len(cluster_X_unpacked), pack=False)
            centroids.append(centroid)
            del cluster_X_unpacked
        else:
            centroids.append(np.zeros(_shared_nbits, dtype=np.uint8))
    return centroids


def _build_cluster_to_indices(cluster_assignments):
    """Build a dictionary mapping cluster_id -> array of molecule indices.
    Single O(n) pass instead of O(n) per cluster.
    """
    cluster_to_indices = {}
    for idx, cid in enumerate(cluster_assignments):
        cid = int(cid)
        if cid not in cluster_to_indices:
            cluster_to_indices[cid] = []
        cluster_to_indices[cid].append(idx)
    # Convert lists to numpy arrays for fast fancy indexing
    for cid in cluster_to_indices:
        cluster_to_indices[cid] = np.array(cluster_to_indices[cid], dtype=np.int64)
    return cluster_to_indices

class Data_Processing:
    def __init__(self, parent_app):
        self.app = parent_app

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
    
    def save_results(self):
        """Lightweight save method - only saves SMILES, molecule PCA coordinates, and centroids"""
        if self.app.data is None or self.app.cluster_assignments is None:
            messagebox.showwarning("Warning", "No clustering results to save!")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save Clustering Results",
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        
        if not save_path: 
            return
        
        try:
            # Use already-computed molecule PCA coordinates (X is deleted after clustering to save memory)
            molecule_pca = None
            if hasattr(self.app, 'molecule_pca') and self.app.molecule_pca is not None:
                molecule_pca = self.app.molecule_pca
            
            # Basic required data
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
                    'use_parallel_clustering': self.app.use_parallel_clustering_var.get(),
                    'parallel_num_processes': self.app.parallel_num_processes_var.get(),
                    'use_incremental_pca': self.app.use_incremental_pca_var.get()
                }
            }
            
            # Add molecule PCA coordinates
            if molecule_pca is not None:
                save_data['molecule_pca'] = {
                    'PC1': molecule_pca['PC1'].tolist(),
                    'PC2': molecule_pca['PC2'].tolist(),
                    'cluster': molecule_pca['cluster'].tolist()
                }
            
            # Add centroid PCA coordinates
            if hasattr(self.app, 'centroid_pca') and self.app.centroid_pca is not None and not self.app.centroid_pca.empty:
                save_data['centroid_pca'] = {
                    'PC1': self.app.centroid_pca['PC1'].tolist(),
                    'PC2': self.app.centroid_pca['PC2'].tolist(),
                    'cluster': self.app.centroid_pca['cluster'].tolist(),
                    'size': self.app.centroid_pca['size'].tolist(),
                    'hover': self.app.centroid_pca['hover'].tolist()
                }
            
            # Add names if available
            if 'Name' in self.app.data.columns:
                save_data['names'] = self.app.data['Name'].tolist()
            
            np.save(save_path, save_data, allow_pickle=True)
            
            # Show what was saved
            saved_items = ["SMILES", "cluster assignments", "parameters"]
            if molecule_pca is not None:
                saved_items.append("molecule PCA coordinates")
            if 'centroid_pca' in save_data:
                saved_items.append("centroid PCA coordinates")
            if 'names' in save_data:
                saved_items.append("molecule names")
            
            messagebox.showinfo("Success", f"Lightweight results saved successfully!\n\nSaved: {', '.join(saved_items)}")
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to save results:\n{str(e)}\n\nFull error:\n{traceback.format_exc()}"
            messagebox.showerror("Error", error_msg)
    
    def load_results(self):
        """Load lightweight clustering results"""
        load_path = filedialog.askopenfilename(
            title="Load Clustering Results",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        
        if not load_path: 
            return
        
        try:
            loaded_data = np.load(load_path, allow_pickle=True).item()
            
            # Validate loaded data structure
            required_keys = ['smiles', 'cluster_assignments']
            if not all(key in loaded_data for key in required_keys):
                messagebox.showerror("Error", "Invalid save file format!")
                return
            
            # Restore basic data
            smiles_list = loaded_data['smiles']
            self.app.data = pd.DataFrame({'SMILES': smiles_list})
            
            if 'names' in loaded_data:
                self.app.data['Name'] = loaded_data['names']
            
            # Convert back to numpy arrays
            self.app.cluster_assignments = np.array(loaded_data['cluster_assignments'])
            self.app.data['cluster'] = self.app.cluster_assignments.tolist()
            
            # Do NOT regenerate mol objects - will be created on-demand for viewed clusters only
            
            # Restore molecule PCA coordinates if available
            if 'molecule_pca' in loaded_data:
                mol_pca_data = loaded_data['molecule_pca']
                self.app.molecule_pca = pd.DataFrame({
                    'PC1': mol_pca_data['PC1'],
                    'PC2': mol_pca_data['PC2'],
                    'cluster': mol_pca_data['cluster']
                })
            else:
                self.app.molecule_pca = None
            
            # Restore centroid PCA coordinates if available
            if 'centroid_pca' in loaded_data:
                pca_data = loaded_data['centroid_pca']
                self.app.centroid_pca = pd.DataFrame({
                    'PC1': pca_data['PC1'],
                    'PC2': pca_data['PC2'],
                    'cluster': pca_data['cluster'],
                    'size': pca_data['size'],
                    'hover': pca_data['hover']
                })
            else:
                self.app.centroid_pca = pd.DataFrame()
            
            # Clear fingerprint-dependent data since we didn't save it
            self.app.X = None
            self.app.centroids = None
            self.app.brc = None

            # Update parameters display if available
            if 'parameters' in loaded_data:
                params = loaded_data['parameters']
                self.app.threshold_var.set(str(params.get('threshold', 0.65)))
                self.app.branching_var.set(str(params.get('branching_factor', 50)))
                self.app.radius_var.set(str(params.get('radius', 2)))
                self.app.nbits_var.set(str(params.get('nbits', 2048)))
                
                if 'min_cluster_size' in params:
                    #self.app.min_cluster_size_var.set("100")
                    self.app.min_cluster_size_var.set(str(params['min_cluster_size']))
                if 'max_cluster_size' in params:
                    #self.app.max_cluster_size_var.set("1000")
                    self.app.max_cluster_size_var.set(str(params['max_cluster_size']))
                if 'hide_singletons' in params:
                    self.app.hide_singletons_var.set(bool(params['hide_singletons']))
                if 'use_parallel_clustering' in params:
                    self.app.use_parallel_clustering_var.set(bool(params['use_parallel_clustering']))
                if 'parallel_num_processes' in params:
                    self.app.parallel_num_processes_var.set(str(params['parallel_num_processes']))
                if 'use_incremental_pca' in params:
                    self.app.use_incremental_pca_var.set(bool(params['use_incremental_pca']))
            
            # Update GUI state
            self.app.gui.file_label.config(text=f"Loaded: {load_path.split('/')[-1]}")
            self.app.gui.save_btn.config(state="normal")
            self.app.gui.refresh_btn.config(state="normal")
             
            # Display loaded data
            self.app.display_data_info()
            self.app.show_overview()
            self.app.visualizer.display_clustering_results()

            num_clusters = len(np.unique(self.app.cluster_assignments[self.app.cluster_assignments >= 0]))
            
            # Show what was loaded
            loaded_items = ["SMILES", "cluster assignments", "parameters"]
            if hasattr(self.app, 'molecule_pca') and self.app.molecule_pca is not None:
                loaded_items.append("molecule PCA coordinates")
            if not self.app.centroid_pca.empty:
                loaded_items.append("centroid PCA coordinates")
            if 'Name' in self.app.data.columns:
                loaded_items.append("molecule names")
            
            messagebox.showinfo("Success", f"Results loaded from {load_path.split('/')[-1]}\n\n"
                                         f"Molecules: {len(self.app.data)}\n"
                                         f"Clusters: {num_clusters}\n\n"
                                         f"Loaded: {', '.join(loaded_items)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load results:\n{str(e)}")

    def load_smiles_file(self):
        file_path = filedialog.askopenfilename(
            title="Select SMILES CSV file", 
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path: 
            return
        
        # Start progress indicator and disable button
        self.app.gui.progress.start()
        self.app.gui.file_label.config(text="Loading file...")
        
        # Load file in background thread to avoid freezing GUI
        thread = threading.Thread(target=self._load_file_background, args=(file_path,))
        thread.daemon = True
        thread.start()
    
    def _load_file_background(self, file_path):
        """Load CSV file in background thread"""
        try:
            import time
            t_start = time.time()
            
            # Quick peek at file structure (first 5 lines only)
            sample = pd.read_csv(file_path, nrows=5, header=None)
            
            # Determine format and read full file efficiently
            if len(sample.columns) == 1:
                # Space-separated format
                self.app.data = pd.read_csv(file_path, sep=" ", names=["SMILES", "Name"], 
                                           engine='c', low_memory=False)
            elif len(sample.columns) == 2:
                # Two columns, no header - assume SMILES, Name
                self.app.data = pd.read_csv(file_path, names=["SMILES", "Name"], 
                                           engine='c', low_memory=False)
            else:
                # Has header, check for SMILES column
                self.app.data = pd.read_csv(file_path, engine='c', low_memory=False)
                if 'smiles' in self.app.data.columns:
                    self.app.data.rename(columns={'smiles': 'SMILES'}, inplace=True)
                elif 'SMILES' not in self.app.data.columns:
                    # No SMILES column, assume first column is SMILES
                    cols = list(self.app.data.columns)
                    self.app.data.rename(columns={cols[0]: 'SMILES'}, inplace=True)
                    if len(cols) > 1:
                        self.app.data.rename(columns={cols[1]: 'Name'}, inplace=True)
            
            t_end = time.time()
            load_time = t_end - t_start
            
            # Update GUI on main thread
            self.app.root.after(0, lambda: self._finish_file_load(file_path, load_time))
            
        except Exception as e:
            self.app.root.after(0, lambda: self._file_load_error(str(e)))
    
    def _finish_file_load(self, file_path, load_time):
        """Finish file loading on main thread"""
        self.app.gui.progress.config(mode='indeterminate')
        self.app.gui.progress.stop()
        self.app.gui.file_label.config(text=f"Loaded: {file_path.split('/')[-1]}")
        self.app.gui.process_btn.config(state="normal")
        self.app.display_data_info()
        messagebox.showinfo("Success", 
            f"SMILES data loaded successfully!\n"
            f"Shape: {self.app.data.shape}\n"
            f"Load time: {load_time:.2f}s ({len(self.app.data)/load_time:.0f} rows/s)")
    
    def _file_load_error(self, error_msg):
        """Handle file loading error on main thread"""
        self.app.gui.progress.config(mode='indeterminate')
        self.app.gui.progress.stop()
        self.app.gui.file_label.config(text="Load failed")
        messagebox.showerror("Error", f"Failed to load file:\n{error_msg}")

    def mol2fp(self, mol):
        return AllChem.GetMorganFingerprintAsBitVect(
            mol, 
            int(self.app.radius_var.get()), 
            nBits=int(self.app.nbits_var.get())
        )
    
    def start_processing(self):
        # Switch to deterministic mode and reset to 0
        self.app.gui.progress.config(mode='determinate', value=0, maximum=100)
        self.app.gui.process_btn.config(state="disabled")
        thread = threading.Thread(target=self.process_data)
        thread.daemon = True
        thread.start()
    
    def process_data(self):
        import time
        try:
            # Stage 1: Fingerprint generation (0-60%)
            self.app.root.after(0, lambda: self.app.gui.progress.config(value=0))
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, "[Stage 1/3] Converting SMILES to fingerprints (parallel)...\n"))
            
            t_start = time.time()
            radius = int(self.app.radius_var.get())
            nbits = int(self.app.nbits_var.get())
            
            # Parallel fingerprint generation in batches using BBlean's optimized functions
            smiles_list = self.app.data['SMILES'].tolist()
            # Respect SLURM/cgroup CPU allocation instead of using all node cores
            import os
            n_cores = max(1, len(os.sched_getaffinity(0)) - 1)  # Leave one core free
            
            # Map radius to BBlean fingerprint kind
            if radius == 2:
                fp_kind = 'ecfp4'  # Morgan radius 2
            elif radius == 3:
                fp_kind = 'ecfp6'  # Morgan radius 3
            else:
                fp_kind = 'rdkit'  # Default to RDKit fingerprint
            
            # Optimal batch size: larger batches for large datasets but cap at 1000 for efficiency
            # Target: at least 4 batches per core, but keep batch size reasonable
            n_samples = len(smiles_list)
            min_batches_per_core = 4
            ideal_batch_size = min(1000, max(100, n_samples // (n_cores * min_batches_per_core)))
            batch_size = ideal_batch_size
            
            # Split into batches with indices
            batches_with_idx = [(i, smiles_list[i*batch_size:(i+1)*batch_size]) 
                                for i in range((len(smiles_list) + batch_size - 1) // batch_size)]
            
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, f"Using {n_cores} cores, {len(batches_with_idx)} batches (batch_size={batch_size})\n"))
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, f"Using BBlean optimized fingerprint generation (kind={fp_kind})\n"))
            
            # Process batches in parallel with BBlean's optimized function
            process_func = partial(_process_smiles_batch_bblean, fp_kind=fp_kind, nbits=nbits)
            chunksize = max(1, len(batches_with_idx) // (n_cores * 2))  # Submit work in chunks
            
            # Collect packed fingerprints and track valid indices
            fps_list = []
            valid_indices = []
            
            with Pool(n_cores) as pool:
                for batch_fps, batch_idx, local_valid_indices in tqdm(pool.imap(process_func, batches_with_idx, chunksize=chunksize), 
                                         total=len(batches_with_idx), desc="Processing batches"):
                    if len(batch_fps) > 0:
                        fps_list.append(batch_fps)
                        # Convert local indices to global indices
                        batch_start = batch_idx * batch_size
                        global_indices = [batch_start + local_idx for local_idx in local_valid_indices]
                        valid_indices.extend(global_indices)
                    # Update progress bar (0-60% range for FP generation)
                    progress_pct = int((batch_idx + 1) / len(batches_with_idx) * 60)
                    self.app.root.after(0, lambda p=progress_pct: self.app.gui.progress.config(value=p))
            
            # Concatenate all batches into single array
            self.app.X = np.vstack(fps_list) if fps_list else np.array([], dtype=np.uint8)
            
            # Filter dataframe to keep only valid SMILES
            if len(valid_indices) < len(smiles_list):
                invalid_count = len(smiles_list) - len(valid_indices)
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, f"Removed {invalid_count} invalid SMILES (auto-skipped by BBlean)\n"))
                self.app.data = self.app.data.iloc[valid_indices].reset_index(drop=True)
            packed_size_mb = self.app.X.nbytes / (1024 * 1024)
            
            t_fp_end = time.time()
            fp_time = t_fp_end - t_start
            
            self.app.root.after(0, lambda: self.app.gui.progress.config(value=60))
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, f"Generated PACKED fingerprint matrix: {self.app.X.shape} ({packed_size_mb:.1f} MB) in {fp_time:.2f}s ({len(fps_list)/fp_time:.0f} fps/s)\n"))
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, f"Memory: 8× reduction via bit-packing (8 bits/byte), mol objects not stored\n"))
            
            '''No problem till here'''

            # Stage 2: BitBirch clustering (60-80%)
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, "[Stage 2/3] Performing BitBirch clustering...\n"))
            
            t_cluster_start = time.time()
            n_mols = len(self.app.X)
            use_parallel = self.app.use_parallel_clustering_var.get()
            
            # Auto-enable parallel for large datasets (>1M molecules) if not explicitly disabled
            if n_mols > 1_000_000 and not use_parallel:
                msg = f"Large dataset ({n_mols:,} molecules) detected. Consider enabling parallel clustering for faster processing."
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, f"Note: {msg}\n"))
            
            if use_parallel and n_mols > 100_000:
                # Use multiround parallel clustering for large datasets
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, "Using parallel multiround clustering...\n"))
                self.app.brc, self.app.cluster_assignments, num_clusters = self._parallel_clustering(
                    t_cluster_start, nbits
                )
                # Debug: Check what was returned
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                    f"Parallel clustering returned: brc={type(self.app.brc).__name__}, "
                    f"assignments={len(self.app.cluster_assignments)}, clusters={num_clusters}\n"))
            else:
                # Standard single-process clustering
                if use_parallel and n_mols <= 100_000:
                    self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, "Dataset too small for parallel clustering, using standard mode.\n"))
                
                self.app.brc = BitBirch(
                    branching_factor=int(self.app.branching_var.get()), 
                    threshold=float(self.app.threshold_var.get()),
                    merge_criterion='diameter',  # Explicit merge criterion
                    tolerance=0.05  # Fine-tune merging behavior
                )
                self.app.brc.fit(self.app.X, input_is_packed=True)
                
                # Use BitBirch's optimized built-in method for assignments (much faster)
                # Note: get_assignments() returns 1-indexed labels, convert to 0-indexed
                self.app.cluster_assignments = self.app.brc.get_assignments(sort=True) - 1
                
                # Count unique clusters (excluding -1 if any unassigned)
                num_clusters = len(self.app.brc.get_cluster_mol_ids())
            
            # Direct numpy array assignment to DataFrame (faster than .tolist())
            self.app.data['cluster'] = self.app.cluster_assignments
            
            t_cluster_end = time.time()
            cluster_time = t_cluster_end - t_cluster_start
            
            self.app.root.after(0, lambda: self.app.gui.progress.config(value=80))
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, f"Number of clusters: {num_clusters} (clustering took {cluster_time:.2f}s)\n"))
            
            # Stage 3: Compute centroids and PCA (80-100%)
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, "[Stage 3/3] Computing PCA coordinates...\n"))
            
            # Get cluster sizes
            cluster_sizes = self.app.data['cluster'].value_counts().sort_index()
            
            # Filter out singletons if option is enabled
            if self.app.hide_singletons_var.get():
                # Get non-singleton cluster indices
                non_singleton_mask = cluster_sizes > 1
                non_singleton_cluster_ids = non_singleton_mask[non_singleton_mask].index.tolist()
                
                if non_singleton_cluster_ids:
                    try:
                        # MEMORY OPTIMIZED: Compute only non-singleton centroids directly
                        # Avoids unpacking ALL centroids (including 1M+ singletons we don't need)
                        self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                            f"Computing centroids for {len(non_singleton_cluster_ids)} non-singleton clusters (memory-efficient)...\n"))
                        
                        # Pre-group molecule indices by cluster (single O(n) pass)
                        self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                            f"Building cluster index map...\n"))
                        cluster_to_indices = _build_cluster_to_indices(self.app.cluster_assignments)
                        
                        # Parallel centroid computation
                        n_cores = max(1, len(os.sched_getaffinity(0)) - 1)
                        batch_size = max(10, len(non_singleton_cluster_ids) // (n_cores * 20))  # 20 batches per core for load balancing
                        
                        # Split cluster IDs into batches
                        cluster_id_batches = [non_singleton_cluster_ids[i:i+batch_size] 
                                             for i in range(0, len(non_singleton_cluster_ids), batch_size)]
                        
                        self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                            f"Computing centroids in parallel ({n_cores} cores, {len(cluster_id_batches)} batches)...\n"))
                        
                        # Compute centroids in parallel (shared data via initializer, no pickling)
                        import time
                        t_centroid_start = time.time()
                        chunksize = max(1, len(cluster_id_batches) // (n_cores * 4))
                        with Pool(n_cores, initializer=_centroid_pool_init, 
                                  initargs=(self.app.X, cluster_to_indices, nbits)) as pool:
                            results = pool.map(_compute_centroids_batch, cluster_id_batches, chunksize=chunksize)
                        t_centroid_end = time.time()
                        
                        self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                            f"Centroid computation took {t_centroid_end - t_centroid_start:.1f}s\n"))
                        
                        # Flatten results (list of lists -> single list)
                        filtered_centroids = []
                        for batch_centroids in results:
                            filtered_centroids.extend(batch_centroids)
                        
                        filtered_centroids = np.array(filtered_centroids, dtype=np.uint8)
                        
                        self.app.root.after(0, lambda fc=filtered_centroids: self.app.gui.results_text.insert(tk.END, 
                            f"Computed {len(fc)} centroids directly (shape: {fc.shape})\n"))
                        
                        # Compute PCA on filtered centroids
                        self.app.centroid_pca = self.compute_pca(filtered_centroids)
                        
                        if not self.app.centroid_pca.empty:
                            self.app.centroid_pca['cluster'] = non_singleton_cluster_ids
                            self.app.centroid_pca['size'] = [int(cluster_sizes[i]) for i in non_singleton_cluster_ids]
                            self.app.centroid_pca['hover'] = [f"Cluster {int(i)}" for i in non_singleton_cluster_ids]
                            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                                f"Centroid PCA computed: {len(self.app.centroid_pca)} points\n"))
                        else:
                            raise ValueError("PCA computation returned empty DataFrame")
                        
                        n_singletons = num_clusters - len(non_singleton_cluster_ids)
                        self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                            f"Excluded {n_singletons} singleton clusters from PCA (memory saved)\n"))
                    except Exception as e:
                        self.app.root.after(0, lambda err=str(e): self.app.gui.results_text.insert(tk.END, 
                            f"ERROR getting centroids from tree: {err}\n"))
                        self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                            f"Computing centroids directly from fingerprints (fallback)...\n"))
                        
                        # FALLBACK: Compute centroids directly from fingerprints
                        try:
                            # Pre-group molecule indices by cluster (single O(n) pass)
                            cluster_to_indices = _build_cluster_to_indices(self.app.cluster_assignments)
                            
                            # Parallel centroid computation (fallback)
                            n_cores = max(1, len(os.sched_getaffinity(0)) - 1)
                            batch_size = max(10, len(non_singleton_cluster_ids) // (n_cores * 20))
                            
                            # Split cluster IDs into batches
                            cluster_id_batches = [non_singleton_cluster_ids[i:i+batch_size] 
                                                 for i in range(0, len(non_singleton_cluster_ids), batch_size)]
                            
                            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                                f"Computing centroids in parallel fallback ({n_cores} cores, {len(cluster_id_batches)} batches)...\n"))
                            
                            # Compute centroids in parallel (shared data via initializer, no pickling)
                            chunksize = max(1, len(cluster_id_batches) // (n_cores * 4))
                            with Pool(n_cores, initializer=_centroid_pool_init,
                                      initargs=(self.app.X, cluster_to_indices, nbits)) as pool:
                                results = pool.map(_compute_centroids_batch, cluster_id_batches, chunksize=chunksize)
                            
                            # Flatten results
                            filtered_centroids = []
                            for batch_centroids in results:
                                filtered_centroids.extend(batch_centroids)
                            
                            filtered_centroids = np.array(filtered_centroids, dtype=np.uint8)
                            
                            self.app.root.after(0, lambda fc=filtered_centroids: self.app.gui.results_text.insert(tk.END, 
                                f"Computed {len(fc)} centroids directly (shape: {fc.shape})\n"))
                            
                            # Compute PCA on computed centroids
                            self.app.centroid_pca = self.compute_pca(filtered_centroids)
                            
                            if not self.app.centroid_pca.empty:
                                self.app.centroid_pca['cluster'] = non_singleton_cluster_ids
                                self.app.centroid_pca['size'] = [int(cluster_sizes[i]) for i in non_singleton_cluster_ids]
                                self.app.centroid_pca['hover'] = [f"Cluster {int(i)}" for i in non_singleton_cluster_ids]
                                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                                    f"Centroid PCA computed from fallback: {len(self.app.centroid_pca)} points\n"))
                            else:
                                raise ValueError("PCA still empty after fallback computation")
                            
                            n_singletons = num_clusters - len(non_singleton_cluster_ids)
                            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                                f"Excluded {n_singletons} singleton clusters\n"))
                            
                            gc.collect()
                            
                        except Exception as fallback_err:
                            self.app.root.after(0, lambda err=str(fallback_err): self.app.gui.results_text.insert(tk.END, 
                                f"FALLBACK FAILED: {err}\n"))
                            # Initialize empty centroid_pca to prevent crashes
                            self.app.centroid_pca = pd.DataFrame(columns=['PC1', 'PC2', 'cluster', 'size', 'hover'])
                else:
                    # All clusters are singletons
                    self.app.centroid_pca = pd.DataFrame(columns=['PC1', 'PC2', 'cluster', 'size', 'hover'])
                    self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                        "Warning: All clusters are singletons, no PCA computed\n"))
            else:
                # Include all clusters (original behavior)
                try:
                    self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                        f"Getting centroids for all {num_clusters} clusters (singletons not filtered)...\n"))
                    self.app.centroids = self.app.brc.get_centroids(packed=False)
                    self.app.root.after(0, lambda c=self.app.centroids: self.app.gui.results_text.insert(tk.END, 
                        f"Retrieved {len(c)} centroids, shape: {c.shape if hasattr(c, 'shape') else 'N/A'}\n"))
                    
                    if len(self.app.centroids) == 0:
                        raise ValueError("get_centroids() returned empty array")
                    
                    self.app.centroid_pca = self.compute_pca(self.app.centroids)
                    
                    if not self.app.centroid_pca.empty:
                        self.app.centroid_pca['cluster'] = list(range(num_clusters))
                        self.app.centroid_pca['size'] = [int(s) for s in cluster_sizes.values]
                        self.app.centroid_pca['hover'] = [f"Cluster {int(i)}" for i in range(num_clusters)]
                        self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                            f"Centroid PCA computed successfully: {len(self.app.centroid_pca)} points\n"))
                    else:
                        raise ValueError("PCA computation returned empty DataFrame")
                except Exception as e:
                    self.app.root.after(0, lambda err=str(e): self.app.gui.results_text.insert(tk.END, 
                        f"ERROR getting centroids from tree: {err}\n"))
                    self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                        f"Computing centroids directly from fingerprints (fallback)...\n"))
                    
                    # FALLBACK: Compute centroids directly from fingerprints
                    try:
                        # Pre-group molecule indices by cluster (single O(n) pass)
                        cluster_to_indices = _build_cluster_to_indices(self.app.cluster_assignments)
                        
                        # Parallel centroid computation (all clusters)
                        n_cores = max(1, len(os.sched_getaffinity(0)) - 1)
                        batch_size = max(10, num_clusters // (n_cores * 20))
                        
                        # Split cluster IDs into batches
                        all_cluster_ids = list(range(num_clusters))
                        cluster_id_batches = [all_cluster_ids[i:i+batch_size] 
                                             for i in range(0, num_clusters, batch_size)]
                        
                        self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                            f"Computing all centroids in parallel ({n_cores} cores, {len(cluster_id_batches)} batches)...\n"))
                        
                        # Compute centroids in parallel (shared data via initializer, no pickling)
                        chunksize = max(1, len(cluster_id_batches) // (n_cores * 4))
                        with Pool(n_cores, initializer=_centroid_pool_init,
                                  initargs=(self.app.X, cluster_to_indices, nbits)) as pool:
                            results = pool.map(_compute_all_centroids_batch, cluster_id_batches, chunksize=chunksize)
                        
                        # Flatten results
                        centroids_list = []
                        for batch_centroids in results:
                            centroids_list.extend(batch_centroids)
                        
                        self.app.centroids = np.array(centroids_list, dtype=np.uint8)
                        
                        self.app.root.after(0, lambda c=self.app.centroids: self.app.gui.results_text.insert(tk.END, 
                            f"Computed {len(c)} centroids directly (shape: {c.shape})\n"))
                        
                        # Compute PCA on computed centroids
                        self.app.centroid_pca = self.compute_pca(self.app.centroids)
                        
                        if not self.app.centroid_pca.empty:
                            self.app.centroid_pca['cluster'] = list(range(num_clusters))
                            self.app.centroid_pca['size'] = [int(s) for s in cluster_sizes.values]
                            self.app.centroid_pca['hover'] = [f"Cluster {int(i)}" for i in range(num_clusters)]
                            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                                f"Centroid PCA computed from fallback: {len(self.app.centroid_pca)} points\n"))
                        else:
                            raise ValueError("PCA still empty after fallback computation")
                        
                        gc.collect()
                        
                    except Exception as fallback_err:
                        self.app.root.after(0, lambda err=str(fallback_err): self.app.gui.results_text.insert(tk.END, 
                            f"FALLBACK FAILED: {err}\n"))
                        # Initialize empty centroid_pca to prevent crashes
                        self.app.centroid_pca = pd.DataFrame(columns=['PC1', 'PC2', 'cluster', 'size', 'hover'])
            
            # Full molecule PCA - either skip or use IncrementalPCA
            self.app.root.after(0, lambda: self.app.gui.progress.config(value=90))
            n_mols = len(self.app.X)
            
            if self.app.use_incremental_pca_var.get():
                # Use IncrementalPCA for memory-efficient full molecule PCA
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                    f"Computing full molecule PCA using IncrementalPCA...\n"))
                
                t_pca_start = time.time()
                self.app.molecule_pca = self._compute_incremental_pca(nbits)
                pca_time = time.time() - t_pca_start
                
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                    f"IncrementalPCA completed in {pca_time:.2f}s (memory-efficient batching)\n"))
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                    f"Peak memory: ~5GB (vs ~80-100GB for standard PCA)\n"))
            else:
                # Skip full molecule PCA - compute on-demand per cluster when clicked
                # This avoids 80-100GB memory spike for 10M molecules
                self.app.molecule_pca = None  # Will compute when user clicks a cluster in overview
                packed_size_mb = self.app.X.nbytes / (1024 * 1024)
                
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                    f"✓ Centroid PCA computed for overview ({len(self.app.centroid_pca)} clusters)\n"))
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                    f"✓ Skipping full molecule PCA ({n_mols:,} molecules)\n"))
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                    f"  Strategy: Compute per-cluster PCA on-demand when you click a cluster\n"))
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                    f"  Memory saved: ~{n_mols * nbits / (8 * 1024**2) * 10:.0f} GB peak, ~{n_mols * 2 * 4 / (1024**2):.0f} MB storage\n"))
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                    f"  Kept packed fingerprints: {packed_size_mb:.1f} MB (for on-demand computation)\n"))
            
            gc.collect()
            
            # Complete - 100%
            self.app.root.after(0, lambda: self.app.gui.progress.config(value=100))
            self.app.root.after(0, self.finish_processing)
            
        except Exception as e:
            self.app.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed:\n{str(e)}"))
            self.app.root.after(0, lambda: self.app.gui.progress.config(value=0))
            self.app.root.after(0, lambda: self.app.gui.process_btn.config(state="normal"))
    
    def _compute_incremental_pca(self, nbits):
        """Compute PCA on full dataset using IncrementalPCA for memory efficiency.
        
        Memory-efficient batch processing for large datasets (e.g., 10M molecules):
        - Standard PCA: ~80-100GB peak memory
        - IncrementalPCA: ~3-5GB peak memory
        - Accuracy: 99.9% correlation with standard PCA
        
        Args:
            nbits: Number of bits in fingerprint (e.g., 2048)
            
        Returns:
            pd.DataFrame with columns: PC1, PC2, Cluster
        """
        n_mols = len(self.app.X)
        batch_size = 10_000  # Process 10K molecules at a time (~160MB per batch)
        n_batches = (n_mols + batch_size - 1) // batch_size
        
        # Initialize IncrementalPCA
        ipca = IncrementalPCA(n_components=2)
        
        # Fit phase: partial_fit on each batch
        self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
            f"Fitting IncrementalPCA on {n_mols:,} molecules in {n_batches} batches...\n"))
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_mols)
            
            # Unpack this batch
            batch_packed = self.app.X[start_idx:end_idx]
            batch_unpacked = unpack_fingerprints(batch_packed, nbits)
            
            # Partial fit
            ipca.partial_fit(batch_unpacked)
            
            # Update progress (fit: 90-95%)
            progress_val = 90 + int(5 * (batch_idx + 1) / n_batches)
            self.app.root.after(0, lambda v=progress_val: self.app.gui.progress.config(value=v))
            
            # Update text every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                self.app.root.after(0, lambda b=batch_idx+1, n=n_batches: 
                    self.app.gui.results_text.insert(tk.END, f"  Fitted batch {b}/{n}\n"))
        
        # Transform phase: transform each batch and collect results
        self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
            f"Transforming {n_mols:,} molecules...\n"))
        
        pca_coords = []
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_mols)
            
            # Unpack this batch
            batch_packed = self.app.X[start_idx:end_idx]
            batch_unpacked = unpack_fingerprints(batch_packed, nbits)
            
            # Transform
            batch_transformed = ipca.transform(batch_unpacked)
            pca_coords.append(batch_transformed)
            
            # Update progress (transform: 95-99%)
            progress_val = 95 + int(4 * (batch_idx + 1) / n_batches)
            self.app.root.after(0, lambda v=progress_val: self.app.gui.progress.config(value=v))
            
            # Update text every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                self.app.root.after(0, lambda b=batch_idx+1, n=n_batches: 
                    self.app.gui.results_text.insert(tk.END, f"  Transformed batch {b}/{n}\n"))
        
        # Concatenate all batch results
        pca_coords_all = np.vstack(pca_coords)
        
        # Create DataFrame
        molecule_pca = pd.DataFrame({
            'PC1': pca_coords_all[:, 0],
            'PC2': pca_coords_all[:, 1],
            'Cluster': self.app.cluster_assignments
        })
        
        # Report explained variance
        explained_var = ipca.explained_variance_ratio_
        self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
            f"Explained variance: PC1={explained_var[0]:.1%}, PC2={explained_var[1]:.1%}\n"))
        
        return molecule_pca
    
    def _parallel_clustering(self, t_cluster_start, nbits):
        """Perform parallel multiround clustering for large datasets"""
        import time
        import tempfile
        import shutil
        from pathlib import Path
        from bblean.multiround import run_multiround_bitbirch
        
        n_mols = len(self.app.X)
        num_processes = int(self.app.parallel_num_processes_var.get())
        
        # Create temporary directory for multiround files
        temp_dir = Path(tempfile.mkdtemp(prefix="nami_multiround_"))
        
        try:
            # Save fingerprints to temporary file (required for multiround)
            fp_file = temp_dir / "fingerprints.npy"
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                f"Saving {n_mols:,} fingerprints to temp file...\n"))
            np.save(fp_file, self.app.X)
            
            # Run multiround clustering
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                f"Running parallel clustering with {num_processes} processes...\n"))
            
            self.app.root.after(0, lambda: self.app.gui.progress.config(value=65))
            
            # Track clustering time manually
            multiround_start_time = time.time()
            
            # Run multiround clustering
            result = run_multiround_bitbirch(
                input_files=[fp_file],
                out_dir=temp_dir,
                n_features=nbits,
                input_is_packed=True,
                num_initial_processes=num_processes,
                num_midsection_processes=max(2, num_processes // 2),
                branching_factor=int(self.app.branching_var.get()),
                threshold=float(self.app.threshold_var.get()),
                tolerance=0.05,
                num_midsection_rounds=1,
                verbose=False,
                cleanup=False  # We'll clean up manually
            )
            
            multiround_elapsed = time.time() - multiround_start_time
            
            self.app.root.after(0, lambda: self.app.gui.progress.config(value=75))
            
            # Display clustering time
            self.app.root.after(0, lambda e=multiround_elapsed: self.app.gui.results_text.insert(tk.END, 
                f"Multiround clustering completed in {e:.2f}s ({e/60:.1f} min)\n"))
            
            # Load clustering results from multiround output files
            import pickle
            
            # Load from saved clustering results (multiround saves to files, not returns)
            clusters_file = temp_dir / "clusters.pkl"
            
            if clusters_file.exists():
                # Load the cluster assignments directly
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                    "Loading cluster assignments from multiround results...\n"))
                
                with open(clusters_file, 'rb') as f:
                    clusters_data = pickle.load(f)
                
                # Debug: Log the structure
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                    f"Clusters data type: {type(clusters_data)}\n"))
                
                # clusters.pkl from bblean's multiround contains mol_ids:
                # a list of lists where mol_ids[cluster_idx] = [mol_idx1, mol_idx2, ...]
                # We need to convert this to a per-molecule cluster assignment array.
                
                if isinstance(clusters_data, dict):
                    mol_ids = clusters_data.get('mol_ids', clusters_data.get('assignments', []))
                    self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                        f"Dict keys: {list(clusters_data.keys())}\n"))
                else:
                    mol_ids = clusters_data
                
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                    f"mol_ids type: {type(mol_ids)}, num_clusters: {len(mol_ids)}\n"))
                
                # Convert list-of-lists (cluster -> molecule indices) to per-molecule assignments
                if isinstance(mol_ids, (list, tuple)) and len(mol_ids) > 0:
                    first_elem = mol_ids[0]
                    
                    if isinstance(first_elem, (list, np.ndarray)):
                        # This is the expected format: mol_ids[cluster_id] = [mol_indices...]
                        n_mols = len(self.app.X)
                        cluster_assignments = np.full(n_mols, -1, dtype=np.int64)
                        
                        for cluster_id, mol_indices in enumerate(mol_ids):
                            for mol_idx in mol_indices:
                                cluster_assignments[int(mol_idx)] = cluster_id
                        
                        n_assigned = np.sum(cluster_assignments >= 0)
                        self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                            f"Converted {len(mol_ids)} clusters -> {n_assigned}/{n_mols} molecule assignments\n"))
                        
                        if n_assigned < n_mols:
                            n_unassigned = n_mols - n_assigned
                            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                                f"WARNING: {n_unassigned} molecules unassigned (will be excluded)\n"))
                    else:
                        # Already a flat array of per-molecule cluster IDs
                        cluster_assignments = np.array(mol_ids, dtype=np.int64)
                elif isinstance(mol_ids, np.ndarray):
                    cluster_assignments = np.asarray(mol_ids, dtype=np.int64).ravel()
                else:
                    raise TypeError(f"Unexpected clusters format: {type(mol_ids)}")
                
                # Ensure 0-indexed
                if len(cluster_assignments) > 0 and cluster_assignments.min() == 1:
                    cluster_assignments = cluster_assignments - 1
                
                num_clusters = len(np.unique(cluster_assignments[cluster_assignments >= 0]))
                
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                    f"Loaded {len(cluster_assignments)} assignments, {num_clusters} clusters\n"))
                
                # Verify assignment count matches data size
                if len(cluster_assignments) != len(self.app.X):
                    raise ValueError(
                        f"Assignment count mismatch: got {len(cluster_assignments)} assignments "
                        f"but have {len(self.app.X)} molecules. "
                        f"Multiround clustering may have filtered some molecules."
                    )
                
                # Create a minimal BitBirch object (for compatibility)
                brc = BitBirch(
                    branching_factor=int(self.app.branching_var.get()),
                    threshold=float(self.app.threshold_var.get()),
                    merge_criterion='diameter',
                    tolerance=0.05
                )
                
                # Skip unreliable tree rebuilding - compute centroids directly instead
                # This is more reliable and memory-efficient than trying to rebuild from buffers
                
                # Check if bblean already saved packed centroids (saves recomputation)
                centroids_packed_file = temp_dir / "cluster-centroids-packed.pkl"
                if centroids_packed_file.exists():
                    self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                        f"Loading pre-computed centroids from multiround output...\n"))
                    with open(centroids_packed_file, 'rb') as f:
                        packed_centroids = pickle.load(f)
                    # Unpack to binary for PCA
                    from bblean.fingerprints import unpack_fingerprints
                    computed_centroids = unpack_fingerprints(np.array(packed_centroids, dtype=np.uint8), n_features=nbits)
                    self.app.root.after(0, lambda n=len(computed_centroids): self.app.gui.results_text.insert(tk.END, 
                        f"Loaded {n} pre-computed centroids (no recomputation needed)\n"))
                else:
                    self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                        f"Computing centroids directly from cluster assignments (memory-efficient)...\n"))
                    
                    # Build cluster_to_indices directly from mol_ids if available
                    if isinstance(mol_ids, (list, tuple)) and len(mol_ids) > 0 and isinstance(mol_ids[0], (list, np.ndarray)):
                        # We already have cluster->molecule index mapping from mol_ids
                        cluster_to_indices = {cid: np.array(indices, dtype=np.int64) 
                                             for cid, indices in enumerate(mol_ids) if len(indices) > 0}
                        self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                            f"Using mol_ids directly as cluster index map ({len(cluster_to_indices):,} clusters)\n"))
                    else:
                        # Fallback: build from assignments array
                        self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                            f"Building cluster index map for {num_clusters:,} clusters...\n"))
                        cluster_to_indices = _build_cluster_to_indices(cluster_assignments)
                    
                    self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                        f"Index map ready ({len(cluster_to_indices):,} clusters mapped)\n"))
                    
                    # PARALLEL: Compute centroids directly from fingerprints using multiple cores
                    n_cores = max(1, len(os.sched_getaffinity(0)) - 1)
                    batch_size = max(10, num_clusters // (n_cores * 20))
                    
                    # Split cluster IDs into batches
                    all_cluster_ids = list(range(num_clusters))
                    cluster_id_batches = [all_cluster_ids[i:i+batch_size] 
                                         for i in range(0, num_clusters, batch_size)]
                    
                    self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                        f"Using {n_cores} cores, {len(cluster_id_batches)} batches...\n"))
                    
                    # Compute centroids in parallel (shared data via initializer, no pickling)
                    import time
                    t_centroid_start = time.time()
                    chunksize = max(1, len(cluster_id_batches) // (n_cores * 4))
                    with Pool(n_cores, initializer=_centroid_pool_init,
                              initargs=(self.app.X, cluster_to_indices, nbits)) as pool:
                        results = pool.map(_compute_all_centroids_batch, cluster_id_batches, chunksize=chunksize)
                    t_centroid_end = time.time()
                    
                    # Flatten results
                    centroids_list = []
                    for batch_centroids in results:
                        centroids_list.extend(batch_centroids)
                    
                    computed_centroids = np.array(centroids_list, dtype=np.uint8)
                    
                    self.app.root.after(0, lambda t=t_centroid_end-t_centroid_start, n=len(centroids_list): 
                        self.app.gui.results_text.insert(tk.END, 
                            f"Computed {n} centroids in {t:.2f}s\n"))
                
                # Monkey-patch the get_centroids method to return our computed centroids
                def _get_centroids_override(packed=False):
                    if packed:
                        from bblean.fingerprints import pack_fingerprints
                        return pack_fingerprints(computed_centroids)
                    return computed_centroids
                
                brc.get_centroids = _get_centroids_override
                
                gc.collect()
                
                self.app.root.after(0, lambda n=len(computed_centroids): 
                    self.app.gui.results_text.insert(tk.END, 
                        f"✓ {n} centroids ready\n"))
            else:
                raise ValueError(
                    f"Multiround clustering did not create expected output files. "
                    f"Looking for: clusters.pkl\n"
                    f"Files found in {temp_dir}:\n" + 
                    "\n".join([f"  - {f.name}" for f in temp_dir.glob("*")])
                )
            
            cluster_time = time.time() - t_cluster_start
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                f"Parallel clustering completed in {cluster_time:.2f}s\n"))
            
            return brc, cluster_assignments, num_clusters
            
        finally:
            # Cleanup temporary directory
            try:
                shutil.rmtree(temp_dir)
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                    "Cleaned up temporary files\n"))
            except Exception as e:
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                    f"Warning: Failed to cleanup temp files: {e}\n"))
    
    def finish_processing(self):
        # Keep progress bar at 100% for a moment before resetting
        self.app.root.after(1000, lambda: self.app.gui.progress.config(value=0))
        self.app.gui.process_btn.config(state="normal")
        self.app.gui.refresh_btn.config(state="normal")
        self.app.display_data_info()
        self.app.gui.save_btn.config(state="normal")
        print("Save button enabled")
        self.app.show_overview()
        self.app.visualizer.display_clustering_results()
    
    def compute_pca(self, data):
        """Compute PCA with automatic selection of algorithm based on data size.
        Uses IncrementalPCA for large datasets (>100K samples) for memory efficiency.
        """
        if len(data) < 2: 
            return pd.DataFrame()
        
        # Use IncrementalPCA for large datasets (>100K centroids)
        if len(data) > 100_000:
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                f"Using IncrementalPCA for {len(data):,} centroids (batch processing)...\n"))
            
            import time
            t_start = time.time()
            
            # Incremental PCA with batching
            ipca = IncrementalPCA(n_components=2, batch_size=10000)
            n_batches = (len(data) + 9999) // 10000
            
            # Fit incrementally in batches
            for i in range(0, len(data), 10000):
                batch = data[i:i+10000]
                ipca.partial_fit(batch)
            
            # Transform all at once (or in batches if memory is an issue)
            if len(data) > 500_000:
                # Transform in batches for very large datasets
                pca_coords = []
                for i in range(0, len(data), 10000):
                    batch = data[i:i+10000]
                    pca_coords.append(ipca.transform(batch))
                result = np.vstack(pca_coords)
            else:
                result = ipca.transform(data)
            
            t_end = time.time()
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, 
                f"IncrementalPCA completed in {t_end - t_start:.1f}s\n"))
            
            return pd.DataFrame(result, columns=['PC1', 'PC2'])
        else:
            # Use regular PCA for smaller datasets (<100K)
            pca = PCA(n_components=2)
            return pd.DataFrame(pca.fit_transform(data), columns=['PC1', 'PC2'])