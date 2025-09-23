import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import mplcursors

class Data_Visualization:
    def __init__(self, parent_app):
        self.app = parent_app
    
    def show_overview(self):
        if self.app.centroid_pca is None or self.app.centroid_pca.empty: 
            return
        
        self.app.current_view = 'overview'
        self.app.selected_cluster = None
        self.app.gui.back_btn.config(state="disabled")
        self.app.gui.clear_molecule_info()
        
        # Filter clusters by size range
        range_mask = (self.app.centroid_pca['size'] >= self.app.processor.get_min_cluster_size()) & (self.app.centroid_pca['size'] <= self.app.processor.get_max_cluster_size())
        large_clusters = self.app.centroid_pca[range_mask].copy()
        
        self.app.gui.fig.clear()
        ax = self.app.gui.fig.add_subplot(111)
        
        if large_clusters.empty:
            max_str = str(int(self.app.processor.get_max_cluster_size())) if self.app.processor.get_max_cluster_size() != float('inf') else '∞'
            ax.text(0.5, 0.5, f'No clusters with {self.app.processor.get_min_cluster_size()}-{max_str} molecules found.\nTry adjusting parameters.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('BitBirch Cluster Overview')
            self.app.gui.fig.tight_layout()
            self.app.gui.canvas.draw()
            return
        
        # Plot with improved styling
        sizes = large_clusters['size'].values
        scaled_sizes = 50 + (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-8) * 450
        colors = plt.cm.Set3(np.linspace(0, 1, len(large_clusters)))
        
        ax.scatter(large_clusters['PC1'], large_clusters['PC2'], s=scaled_sizes, c=colors, 
                  alpha=0.8, edgecolors='white', linewidths=2)
        
        # Add labels
        for i, (idx, row) in enumerate(large_clusters.iterrows()):
            ax.annotate(f"C{row['cluster']}\n({row['size']})", (row['PC1'], row['PC2']), 
                       xytext=(0, 0), textcoords='offset points', fontsize=10, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), weight='bold')
        
        ax.set_xlabel('Principal Component 1', fontsize=12, weight='bold')
        ax.set_ylabel('Principal Component 2', fontsize=12, weight='bold')
        max_str = str(int(self.app.processor.get_max_cluster_size())) if self.app.processor.get_max_cluster_size() != float('inf') else '∞'
        ax.set_title(f'BitBirch Cluster Overview ({self.app.processor.get_min_cluster_size()}-{max_str} molecules)\nClick clusters to explore', 
                    fontsize=14, weight='bold', pad=20)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#fafafa')
        
        self.app.gui.setup_zoom_pan(ax)
        self.app.gui.fig.tight_layout()
        self.app.gui.canvas.draw()
    
    def show_cluster_detail(self, cluster_id):
        if self.app.data is None: 
            return
        
        cluster_mask = self.app.data['cluster'] == cluster_id
        cluster_df = self.app.data[cluster_mask].reset_index(drop=True)
        
        if not self.app.processor.is_cluster_in_overview_range(len(cluster_df)):
            max_str = str(int(self.app.processor.get_max_cluster_size())) if self.app.processor.get_max_cluster_size() != float('inf') else '∞'
            messagebox.showinfo("Info", f"Cluster {cluster_id} has {len(cluster_df)} molecules. Overview range is {self.app.processor.get_min_cluster_size()}-{max_str}.")
            return
        
        if len(cluster_df) < 2: 
            return
        
        # Use stored molecule PCA coordinates if available, otherwise compute from fingerprints
        if hasattr(self.app, 'molecule_pca') and self.app.molecule_pca is not None:
            # Get PCA coordinates from stored data
            cluster_indices = cluster_df.index  # Original indices in the full dataset
            # Map back to original indices in the full dataset
            original_indices = self.app.data[self.app.data['cluster'] == cluster_id].index
            cluster_pca_data = self.app.molecule_pca.loc[original_indices].reset_index(drop=True)
            cluster_pca = pd.DataFrame({
                'PC1': cluster_pca_data['PC1'],
                'PC2': cluster_pca_data['PC2']
            })
        else:
            # Fallback: compute PCA from fingerprints (if available)
            if hasattr(self.app, 'X') and self.app.X is not None:
                cluster_data = self.app.X[cluster_mask]
                cluster_pca = self.app.processor.compute_pca(cluster_data)
            else:
                messagebox.showwarning("Warning", "No PCA coordinates or fingerprint data available for detail view.")
                return
        
        if cluster_pca.empty: 
            return
        
        self.app.current_view = 'cluster'
        self.app.selected_cluster = cluster_id
        self.app.gui.back_btn.config(state="normal")
        
        self.app.gui.fig.clear()
        ax = self.app.gui.fig.add_subplot(111)
        
        # Density-based coloring
        xy = np.vstack([cluster_pca['PC1'], cluster_pca['PC2']])
        density = gaussian_kde(xy)(xy)
        
        scatter = ax.scatter(cluster_pca['PC1'], cluster_pca['PC2'], c=density, s=40, alpha=0.7,
                           cmap='viridis', edgecolors='white', linewidths=0.5)
        
        # Interactive hover
        mols, smiles_list = cluster_df['mol'].tolist(), cluster_df['SMILES'].tolist()
        cursor = mplcursors.cursor(scatter, hover=True)
        
        @cursor.connect("add")
        def on_hover(sel):
            self.app.gui.display_molecule_info(sel.index, mols[sel.index], smiles_list[sel.index], cluster_df)
            sel.annotation.set_visible(False)
        
        plt.colorbar(scatter, ax=ax, label='Density')
        ax.set_xlabel('Principal Component 1', fontsize=12, weight='bold')
        ax.set_ylabel('Principal Component 2', fontsize=12, weight='bold')
        ax.set_title(f'Cluster {cluster_id} Detail View | {len(cluster_df)} molecules\nHover over points for details',
                     fontsize=14, weight='bold', pad=20)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#fafafa')
        
        self.app.gui.setup_zoom_pan(ax)
        self.app.gui.fig.tight_layout()
        self.app.gui.canvas.draw()
        self.display_cluster_details(cluster_id, cluster_df)
    
    def on_plot_click(self, event):
        if event.inaxes is None or self.app.current_view != 'overview': 
            return
        if self.app.centroid_pca is None or self.app.centroid_pca.empty: 
            return
        
        # Filter to overview range
        range_mask = (self.app.centroid_pca['size'] >= self.app.processor.get_min_cluster_size()) & (self.app.centroid_pca['size'] <= self.app.processor.get_max_cluster_size())
        large_clusters = self.app.centroid_pca[range_mask].copy()
        if large_clusters.empty: 
            return
        
        # Find closest centroid
        click_point = np.array([event.xdata, event.ydata])
        centroids_points = large_clusters[['PC1', 'PC2']].values
        distances = np.linalg.norm(centroids_points - click_point, axis=1)
        closest_idx = np.argmin(distances)
        
        # Check if click is close enough (scaled by zoom level)
        xlim, ylim = event.inaxes.get_xlim(), event.inaxes.get_ylim()
        threshold = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 0.1
        
        if distances[closest_idx] < threshold:
            cluster_id = large_clusters.iloc[closest_idx]['cluster']
            self.show_cluster_detail(cluster_id)
    
    def display_clustering_results(self):
        if self.app.data is None or self.app.cluster_assignments is None: 
            return
        
        unique_clusters = np.unique(self.app.cluster_assignments[self.app.cluster_assignments >= 0])
        cluster_sizes = pd.Series(self.app.cluster_assignments).value_counts().sort_index()
        
        results = ["BITBIRCH CLUSTERING RESULTS", "=" * 40]
        results.extend([f"\nTotal clusters: {len(unique_clusters)}", 
                       f"Noise points: {np.sum(self.app.cluster_assignments == -1)}"])
        
        # Data source information
        data_source = ""
        if hasattr(self.app, 'X') and self.app.X is not None:
            data_source = "Full fingerprint data available"
        elif hasattr(self.app, 'molecule_pca') and self.app.molecule_pca is not None:
            data_source = "Using saved PCA coordinates (lightweight mode)"
        else:
            data_source = "Limited data available"
        
        results.append(f"Data source: {data_source}")
        
        # Categorize clusters
        overview_clusters, other_clusters = [], []
        for cluster_id, size in cluster_sizes.items():
            if cluster_id >= 0:
                percentage = (size / len(self.app.data)) * 100
                if self.app.processor.is_cluster_in_overview_range(size):
                    overview_clusters.append(f"  Cluster {cluster_id}: {size} molecules ({percentage:.1f}%)")
                else:
                    other_clusters.append((cluster_id, size, percentage))
        
        # Display results
        if overview_clusters:
            max_str = str(int(self.app.processor.get_max_cluster_size())) if self.app.processor.get_max_cluster_size() != float('inf') else '∞'
            results.extend([f"\nClusters in overview ({self.app.processor.get_min_cluster_size()}-{max_str}):", *overview_clusters])
        
        if other_clusters:
            results.extend([f"\nOther clusters (outside overview range): {len(other_clusters)} clusters",
                           f"  Total molecules: {sum(size for _, size, _ in other_clusters)}"])
        
        results.extend([f"\nParameters:", f"  Threshold: {self.app.threshold_var.get()}",
                       f"  Branching factor: {self.app.branching_var.get()}",
                       f"  FP radius: {self.app.radius_var.get()}", f"  FP bits: {self.app.nbits_var.get()}"])
        
        results.extend([f"\n🔍 CONTROLS:", f"• Scroll: zoom • Drag: pan • Click clusters to explore",
                       f"• Overview shows {self.app.processor.get_min_cluster_size()}-{str(int(self.app.processor.get_max_cluster_size())) if self.app.processor.get_max_cluster_size() != float('inf') else '∞'} molecule clusters"])
        
        self.app.gui.results_text.delete(1.0, tk.END)
        self.app.gui.results_text.insert(tk.END, "\n".join(results))
    
    def display_cluster_details(self, cluster_id, cluster_df):
        results = [f"CLUSTER {cluster_id} DETAILS", "=" * 30, f"\nCluster size: {len(cluster_df)} molecules"]
        
        if 'Name' in cluster_df.columns:
            results.append("\nSample molecules:")
            for i, (idx, row) in enumerate(cluster_df.head(10).iterrows()):
                name = row.get('Name', f'Molecule_{idx}')
                results.append(f"  {i+1}. {name}: {row['SMILES']}")
        else:
            results.extend(["\nSample SMILES:", *[f"  {i+1}. {smiles}" for i, smiles in enumerate(cluster_df['SMILES'].head(10))]])
        
        if len(cluster_df) > 10:
            results.append(f"  ... and {len(cluster_df) - 10} more molecules")
        
        results.extend(["\n🔍 Hover over molecules for detailed information →"])
        
        self.app.gui.results_text.delete(1.0, tk.END)
        self.app.gui.results_text.insert(tk.END, "\n".join(results))