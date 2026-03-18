import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA
from rdkit import Chem
from rdkit.Chem import AllChem
import bitbirch.bitbirch as bb
from tqdm.auto import tqdm
import threading

class Data_Processing_Incremental:
    """Memory-efficient incremental clustering - processes one molecule at a time"""
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
                    'max_cluster_size': self.app.max_cluster_size_var.get()
                }
            }
            
            # Add molecule PCA coordinates if available
            if hasattr(self.app, 'molecule_pca') and self.app.molecule_pca is not None:
                save_data['molecule_pca'] = {
                    'PC1': self.app.molecule_pca['PC1'].tolist(),
                    'PC2': self.app.molecule_pca['PC2'].tolist(),
                    'cluster': self.app.molecule_pca['cluster'].tolist()
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
            if 'molecule_pca' in save_data:
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
            self.app.data['cluster'] = self.app.cluster_assignments
            
            # DO NOT convert SMILES to mol objects here - will create on-demand for selected clusters only
            
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
                    self.app.min_cluster_size_var.set(str(params['min_cluster_size']))
                if 'max_cluster_size' in params:
                    self.app.max_cluster_size_var.set(str(params['max_cluster_size']))
            
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
        
        try:
            sample = pd.read_csv(file_path, nrows=5)
            if len(sample.columns) == 1:
                self.app.data = pd.read_csv(file_path, sep=" ", names=["SMILES", "Name"])
            elif 'SMILES' in sample.columns or 'smiles' in sample.columns:
                self.app.data = pd.read_csv(file_path)
                if 'smiles' in self.app.data.columns:
                    self.app.data.rename(columns={'smiles': 'SMILES'}, inplace=True)
            else:
                self.app.data = pd.read_csv(file_path)
                cols = list(self.app.data.columns)
                self.app.data.rename(columns={cols[0]: 'SMILES'}, inplace=True)
                if len(cols) > 1:
                    self.app.data.rename(columns={cols[1]: 'Name'}, inplace=True)

            self.app.gui.file_label.config(text=f"Loaded: {file_path.split('/')[-1]}")
            self.app.gui.process_btn.config(state="normal")
            self.app.display_data_info()
            messagebox.showinfo("Success", f"SMILES data loaded successfully!\nShape: {self.app.data.shape}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")

    def mol2fp(self, mol):
        return AllChem.GetMorganFingerprintAsBitVect(
            mol, 
            int(self.app.radius_var.get()), 
            nBits=int(self.app.nbits_var.get())
        )
    
    def start_processing(self):
        self.app.gui.progress.start()
        self.app.gui.process_btn.config(state="disabled")
        thread = threading.Thread(target=self.process_data_incremental)
        thread.daemon = True
        thread.start()
    
    def process_data_incremental(self):
        """INCREMENTAL VERSION - Process one molecule at a time to minimize memory"""
        try:
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, "Starting INCREMENTAL clustering...\n"))
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, "Memory-efficient: processing one molecule at a time\n"))
            
            # Initialize BitBIRCH clusterer
            bb.set_merge('diameter')
            self.app.brc = bb.BitBirch(
                branching_factor=int(self.app.branching_var.get()), 
                threshold=float(self.app.threshold_var.get())
            )
            
            # Lists to collect data for batch operations
            valid_indices = []
            fp_batch = []
            batch_size = 1000  # Process in small batches for better performance
            
            # Process molecules one at a time
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, f"Processing {len(self.app.data)} molecules incrementally...\n"))
            
            for idx in tqdm(range(len(self.app.data)), desc="Incremental clustering"):
                smiles = self.app.data.loc[idx, 'SMILES']
                
                # Convert SMILES to mol (temporary)
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                # Generate fingerprint (temporary)
                fp = self.mol2fp(mol)
                fp_array = np.array(list(fp.ToBitString()), dtype=np.uint8).reshape(1, -1)
                
                # Add to batch
                valid_indices.append(idx)
                fp_batch.append(fp_array)
                
                # Process batch when full
                if len(fp_batch) >= batch_size:
                    batch_matrix = np.vstack(fp_batch)
                    self.app.brc.partial_fit(batch_matrix)
                    fp_batch = []  # Clear batch to free memory
                
                # mol and fp are automatically garbage collected here
            
            # Process remaining batch
            if fp_batch:
                batch_matrix = np.vstack(fp_batch)
                self.app.brc.partial_fit(batch_matrix)
                fp_batch = []
            
            # Filter to valid molecules only
            self.app.data = self.app.data.loc[valid_indices].reset_index(drop=True)
            invalid_count = len(valid_indices) - len(self.app.data)
            if invalid_count > 0:
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, f"Removed {invalid_count} invalid SMILES\n"))
            
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, "Extracting cluster assignments...\n"))
            
            # Get cluster assignments
            clust_indices = self.app.brc.get_cluster_mol_ids()
            self.app.cluster_assignments = np.ones(len(self.app.data), dtype='int64') * -1
            
            for label, cluster in enumerate(clust_indices):
                # Adjust for filtered indices
                valid_cluster = [i for i in cluster if i < len(self.app.data)]
                self.app.cluster_assignments[valid_cluster] = label
            
            self.app.data['cluster'] = self.app.cluster_assignments
            num_clusters = len(clust_indices)
            
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, f"Number of clusters: {num_clusters}\n"))
            
            # Compute centroids and PCA
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, "Computing centroid PCA...\n"))
            self.app.centroids = self.app.brc.get_centroids()
            self.app.centroid_pca = self.compute_pca(self.app.centroids)
            
            if not self.app.centroid_pca.empty:
                self.app.centroid_pca['cluster'] = range(num_clusters)
                self.app.centroid_pca['size'] = self.app.data['cluster'].value_counts().sort_index().values
                self.app.centroid_pca['hover'] = self.app.centroid_pca['cluster'].apply(lambda x: f"Cluster {x}")
            
            # Compute molecule PCA using IncrementalPCA to save memory
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, "Computing molecule PCA incrementally...\n"))
            
            ipca = IncrementalPCA(n_components=2, batch_size=batch_size)
            pca_results = []
            
            # Second pass: compute PCA incrementally
            for start_idx in tqdm(range(0, len(self.app.data), batch_size), desc="Computing PCA"):
                end_idx = min(start_idx + batch_size, len(self.app.data))
                batch_fps = []
                
                for idx in range(start_idx, end_idx):
                    smiles = self.app.data.loc[idx, 'SMILES']
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        fp = self.mol2fp(mol)
                        fp_array = np.array(list(fp.ToBitString()), dtype=np.uint8)
                        batch_fps.append(fp_array)
                    else:
                        # Placeholder for invalid molecules
                        batch_fps.append(np.zeros(int(self.app.nbits_var.get()), dtype=np.uint8))
                
                if batch_fps:
                    batch_matrix = np.vstack(batch_fps)
                    pca_batch = ipca.partial_fit(batch_matrix).transform(batch_matrix)
                    pca_results.append(pca_batch)
            
            # Combine PCA results
            if pca_results:
                molecule_pca_coords = np.vstack(pca_results)
                self.app.molecule_pca = pd.DataFrame(molecule_pca_coords, columns=['PC1', 'PC2'])
                self.app.molecule_pca['cluster'] = self.app.cluster_assignments
            
            # No need to delete mol/fp - they were never stored!
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, "✓ Memory optimized: no fingerprint matrix stored!\n"))
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, "✓ Peak memory usage minimized\n"))
            
            # X is not stored in incremental mode
            self.app.X = None
            
            self.app.root.after(0, self.finish_processing)
            
        except Exception as e:
            import traceback
            error_msg = f"Processing failed:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, error_msg))
            self.app.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.app.root.after(0, lambda: self.app.gui.progress.stop())
            self.app.root.after(0, lambda: self.app.gui.process_btn.config(state="normal"))
    
    def finish_processing(self):
        self.app.gui.progress.stop()
        self.app.gui.process_btn.config(state="normal")
        self.app.gui.refresh_btn.config(state="normal")
        self.app.display_data_info()
        self.app.gui.save_btn.config(state="normal")
        print("Save button enabled")
        self.app.show_overview()
        self.app.visualizer.display_clustering_results()
    
    def compute_pca(self, data):
        from sklearn.decomposition import PCA
        if len(data) < 2: 
            return pd.DataFrame()
        pca = PCA(n_components=2)
        return pd.DataFrame(pca.fit_transform(data), columns=['PC1', 'PC2'])
