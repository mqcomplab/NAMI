import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem
import bitbirch.bitbirch as bb
from tqdm.auto import tqdm
import threading

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
            # Compute PCA coordinates for all molecules (not just centroids)
            molecule_pca = None
            if hasattr(self.app, 'X') and self.app.X is not None:
                # Compute PCA for all molecules
                pca = PCA(n_components=2)
                molecule_pca_coords = pca.fit_transform(self.app.X)
                molecule_pca = pd.DataFrame(molecule_pca_coords, columns=['PC1', 'PC2'])
                molecule_pca['cluster'] = self.app.cluster_assignments
            
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
            self.app.data['cluster'] = self.app.cluster_assignments
            
            # Convert SMILES back to molecules for visualization
            self.app.data['mol'] = self.app.data['SMILES'].apply(Chem.MolFromSmiles)
            
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
        thread = threading.Thread(target=self.process_data)
        thread.daemon = True
        thread.start()
    
    def process_data(self):
        try:
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, "Converting SMILES to fingerprints...\n"))
            
            tqdm.pandas(desc="Converting SMILES")
            self.app.data['mol'] = self.app.data.SMILES.progress_apply(Chem.MolFromSmiles)
            
            valid_mols = self.app.data['mol'].notna()
            if not valid_mols.all():
                invalid_count = (~valid_mols).sum()
                self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, f"Removed {invalid_count} invalid SMILES\n"))
                self.app.data = self.app.data[valid_mols].reset_index(drop=True)
            
            tqdm.pandas(desc="Generating fingerprints")
            self.app.data['fp'] = self.app.data.mol.progress_apply(self.mol2fp)
            self.app.X = np.stack(self.app.data.fp.apply(lambda x: np.array(list(x.ToBitString()), dtype=np.int64)))
            
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, f"Generated fingerprint matrix: {self.app.X.shape}\n"))
            
            # BitBirch clustering
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, "Performing BitBirch clustering...\n"))
            bb.set_merge('diameter')
            
            self.app.brc = bb.BitBirch(
                branching_factor=int(self.app.branching_var.get()), 
                threshold=float(self.app.threshold_var.get())
            )
            self.app.brc.fit(self.app.X)
            
            clust_indices = self.app.brc.get_cluster_mol_ids()
            self.app.cluster_assignments = np.ones(self.app.X.shape[0], dtype='int64') * -1
            
            for label, cluster in enumerate(clust_indices):
                self.app.cluster_assignments[cluster] = label
            
            self.app.data['cluster'] = self.app.cluster_assignments
            num_clusters = len(clust_indices)
            
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, f"Number of clusters: {num_clusters}\n"))
            
            # Compute centroids and PCA
            self.app.centroids = self.app.brc.get_centroids()
            self.app.centroid_pca = self.compute_pca(self.app.centroids)
            
            if not self.app.centroid_pca.empty:
                self.app.centroid_pca['cluster'] = range(num_clusters)
                self.app.centroid_pca['size'] = self.app.data['cluster'].value_counts().sort_index().values
                self.app.centroid_pca['hover'] = self.app.centroid_pca['cluster'].apply(lambda x: f"Cluster {x}")
            
            # Also compute PCA for all molecules (for detail view)
            self.app.root.after(0, lambda: self.app.gui.results_text.insert(tk.END, "Computing molecule PCA coordinates...\n"))
            pca = PCA(n_components=2)
            molecule_pca_coords = pca.fit_transform(self.app.X)
            self.app.molecule_pca = pd.DataFrame(molecule_pca_coords, columns=['PC1', 'PC2'])
            self.app.molecule_pca['cluster'] = self.app.cluster_assignments
            
            self.app.root.after(0, self.finish_processing)
            
        except Exception as e:
            self.app.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed:\n{str(e)}"))
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
        if len(data) < 2: 
            return pd.DataFrame()
        pca = PCA(n_components=2)
        return pd.DataFrame(pca.fit_transform(data), columns=['PC1', 'PC2'])