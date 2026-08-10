import tkinter as tk
from tkinter import filedialog, messagebox
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.stats import gaussian_kde
import mplcursors
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

from bblean.analysis import cluster_analysis
from bblean.plotting import summary_plot

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
        
        # Optionally hide singletons
        if self.app.hide_singletons_var.get():
            range_mask = range_mask & (self.app.centroid_pca['size'] > 1)
        
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
        scaled_sizes = scaled_sizes.tolist()  # Convert numpy array to Python list
        colors = plt.cm.Set3(np.linspace(0, 1, len(large_clusters)))
        
        ax.scatter(large_clusters['PC1'].values.tolist(), large_clusters['PC2'].values.tolist(), s=scaled_sizes, c=colors, 
                  alpha=0.8, edgecolors='white', linewidths=2)
        
        # Add labels
        for i, (idx, row) in enumerate(large_clusters.iterrows()):
            ax.annotate(f"C{int(row['cluster'])}\n({int(row['size'])})", 
                       (float(row['PC1']), float(row['PC2'])), 
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

    def show_summary_plot(self):
        if self.app.data is None or self.app.cluster_assignments is None:
            messagebox.showwarning(
                "Warning",
                "Load and cluster data before creating the summary plot."
            )
            return

        if self.app.X is None:
            messagebox.showwarning(
                "Warning",
                "Fingerprint data is not available, so the summary plot cannot be built."
            )
            return

        valid_assignments = self.app.cluster_assignments[self.app.cluster_assignments >= 0]
        if valid_assignments.size == 0:
            messagebox.showinfo("Info", "No valid clusters found to plot.")
            return

        cluster_sizes = np.bincount(valid_assignments)
        top_20_ids = np.argsort(cluster_sizes)[-min(20, len(cluster_sizes)):][::-1]
        top_20_ids = [int(cid) for cid in top_20_ids.tolist() if cluster_sizes[cid] > 0]

        if len(cluster_sizes) < 100_000:
            analysis_clusters = [
                np.where(self.app.cluster_assignments == cid)[0].tolist()
                for cid in range(len(cluster_sizes))
                if cluster_sizes[cid] > 0
            ]
            analysis_assume_sorted = True
            analysis_top = None
        else:
            analysis_clusters = [
                np.where(self.app.cluster_assignments == cid)[0].tolist()
                for cid in top_20_ids
            ]
            analysis_assume_sorted = True
            analysis_top = None

        try:
            nbits = int(self.app.nbits_var.get())
        except ValueError:
            nbits = 1024

        analysis = cluster_analysis(
            analysis_clusters,
            self.app.X,
            smiles=self.app.data['SMILES'].tolist(),
            n_features=nbits,
            top=analysis_top,
            assume_sorted=analysis_assume_sorted,
            input_is_packed=True,
            min_size=1,
        )

        if analysis.df.empty:
            messagebox.showinfo("Info", "No clusters were available for the top-20 summary plot.")
            return

        fig, _ = summary_plot(analysis, title="Top 20 clusters")

        window = tk.Toplevel(self.app.root)
        window.title("Top 20 Cluster Summary Plot")
        window.geometry("900x600")

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(window)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(canvas, toolbar_frame)

        controls_frame = tk.Frame(window)
        controls_frame.pack(fill=tk.X, pady=(6, 4))

        tk.Label(controls_frame, text="Cluster for Scaffold Images (Top 20):").pack(side=tk.LEFT, padx=(8, 6))
        self.app.summary_scaffold_cluster_var = tk.StringVar(value=str(top_20_ids[0]))
        cluster_menu = tk.OptionMenu(
            controls_frame,
            self.app.summary_scaffold_cluster_var,
            *[str(cid) for cid in top_20_ids],
        )
        cluster_menu.pack(side=tk.LEFT)

        scaffold_btn = tk.Button(
            controls_frame,
            text="Generate Top 5 Scaffolds",
            command=lambda: self.generate_top_scaffold_plot(
                int(self.app.summary_scaffold_cluster_var.get())
            ),
        )
        scaffold_btn.pack(side=tk.LEFT, padx=(10, 0))

        save_btn = tk.Button(
            window,
            text="Save 600 DPI Image",
            command=lambda: self.save_summary_plot_image(fig),
        )
        save_btn.pack(pady=(0, 8))

        def _close_window():
            plt.close(fig)
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", _close_window)

        self.app.summary_plot_window = window
        self.app.summary_plot_canvas = canvas
        self.app.summary_plot_figure = fig

    def generate_top_scaffold_plot(self, cluster_id):
        cluster_mask = self.app.cluster_assignments == cluster_id
        cluster_smiles = self.app.data.loc[cluster_mask, 'SMILES'].tolist()

        if not cluster_smiles:
            messagebox.showinfo("Info", f"Cluster {cluster_id} has no molecules.")
            return

        scaffold_counter = Counter()
        for smi in cluster_smiles:
            try:
                scaf_smi = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smi)
            except Exception:
                scaf_smi = ""
            if scaf_smi:
                scaffold_counter[scaf_smi] += 1

        if not scaffold_counter:
            messagebox.showinfo("Info", f"No valid scaffolds found for cluster {cluster_id}.")
            return

        top_scaffolds = scaffold_counter.most_common(5)
        mols = []
        legends = []
        for scaf_smi, count in top_scaffolds:
            mol = Chem.MolFromSmiles(scaf_smi)
            if mol is not None:
                mols.append(mol)
                legends.append(f"Count: {count}")

        if not mols:
            messagebox.showinfo("Info", f"Unable to render scaffold molecules for cluster {cluster_id}.")
            return

        grid_img = Draw.MolsToGridImage(
            mols,
            molsPerRow=min(5, len(mols)),
            subImgSize=(320, 320),
            legends=legends,
        )

        fig, ax = plt.subplots(figsize=(14, 4), dpi=600)
        ax.imshow(np.asarray(grid_img))
        ax.axis("off")
        ax.set_title(f"Cluster {cluster_id}: Top 5 Most Frequent Scaffolds", fontsize=12, pad=12)
        fig.tight_layout()

        window = tk.Toplevel(self.app.root)
        window.title(f"Top 5 Scaffolds - Cluster {cluster_id}")
        window.geometry("1200x520")

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(window)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(canvas, toolbar_frame)

        save_btn = tk.Button(
            window,
            text="Save Scaffold Image (600 DPI PNG)",
            command=lambda: self.save_scaffold_plot_image(fig),
        )
        save_btn.pack(pady=(0, 8))

        def _close_window():
            plt.close(fig)
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", _close_window)

        self.app.scaffold_plot_window = window
        self.app.scaffold_plot_canvas = canvas
        self.app.scaffold_plot_figure = fig

    def save_summary_plot_image(self, fig):
        file_path = filedialog.asksaveasfilename(
            title="Save Summary Plot",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("PDF File", "*.pdf"),
                ("SVG File", "*.svg"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            fig.savefig(file_path, dpi=600, bbox_inches="tight")
            messagebox.showinfo("Success", f"Saved image to:\n{file_path}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save image:\n{exc}")

    def save_scaffold_plot_image(self, fig):
        file_path = filedialog.asksaveasfilename(
            title="Save Scaffold Plot",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            fig.savefig(file_path, dpi=600, bbox_inches="tight")
            messagebox.showinfo("Success", f"Saved scaffold image to:\n{file_path}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save scaffold image:\n{exc}")
    
    def show_cluster_detail(self, cluster_id):
        if self.app.data is None: 
            return
        
        cluster_mask = self.app.data['cluster'] == cluster_id
        cluster_df = self.app.data[cluster_mask].reset_index(drop=True)
        
        if not self.app.processor.is_cluster_in_overview_range(len(cluster_df)):
            max_str = str(int(self.app.processor.get_max_cluster_size())) if self.app.processor.get_max_cluster_size() != float('inf') else '∞'
            messagebox.showinfo("Info", f"Cluster {int(cluster_id)} has {len(cluster_df)} molecules. Overview range is {self.app.processor.get_min_cluster_size()}-{max_str}.")
            return
        
        if len(cluster_df) < 2: 
            return
        
        # Use stored molecule PCA coordinates if available, otherwise compute on-demand from packed fingerprints
        if hasattr(self.app, 'molecule_pca') and self.app.molecule_pca is not None:
            # Get PCA coordinates from pre-computed stored data
            original_indices = self.app.data[self.app.data['cluster'] == cluster_id].index
            cluster_pca_data = self.app.molecule_pca.loc[original_indices].reset_index(drop=True)
            cluster_pca = pd.DataFrame({
                'PC1': cluster_pca_data['PC1'],
                'PC2': cluster_pca_data['PC2']
            })
        elif hasattr(self.app, 'X') and self.app.X is not None:
            # On-demand PCA computation: unpack THIS cluster's fingerprints only
            from bblean.fingerprints import unpack_fingerprints
            
            # Show computation message
            self.app.gui.results_text.insert(tk.END, f"\n🔄 Computing PCA for cluster {int(cluster_id)} ({len(cluster_df)} molecules)...\n")
            self.app.root.update()
            
            cluster_fps_packed = self.app.X[cluster_mask]
            
            # Unpack only this cluster (much smaller memory footprint)
            nbits = int(self.app.nbits_var.get())
            cluster_fps_unpacked = unpack_fingerprints(cluster_fps_packed, n_features=nbits)
            
            # Compute PCA on this cluster only
            import time
            t_start = time.time()
            pca = PCA(n_components=2)
            pca_coords = pca.fit_transform(cluster_fps_unpacked)
            cluster_pca = pd.DataFrame(pca_coords, columns=['PC1', 'PC2'])
            pca_time = time.time() - t_start
            
            self.app.gui.results_text.insert(tk.END, f"✓ PCA computed in {pca_time:.2f}s (on-demand)\n")
            
            # Clean up
            del cluster_fps_unpacked
            import gc
            gc.collect()
        else:
            messagebox.showwarning("Warning", "No fingerprint data available for detail view.\nLoad from saved results or rerun clustering.")
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
        
        scatter = ax.scatter(cluster_pca['PC1'].values.tolist(), cluster_pca['PC2'].values.tolist(), 
                           c=density.tolist(), s=40, alpha=0.7,
                           cmap='viridis', edgecolors='white', linewidths=0.5)
        
        # Convert SMILES to mol objects on-demand for this cluster only
        from rdkit import Chem
        if 'mol' not in cluster_df.columns:
            cluster_df['mol'] = cluster_df['SMILES'].apply(Chem.MolFromSmiles)
        
        # Interactive click to show all molecules at a point
        mols, smiles_list = cluster_df['mol'].tolist(), cluster_df['SMILES'].tolist()
        
        # Store cluster data for click detection
        self.app.current_cluster_df = cluster_df
        self.app.current_cluster_pca = cluster_pca
        
        # Group molecules by their PCA coordinates to find duplicates
        # Reset cluster_pca index to match cluster_df
        cluster_pca_reset = cluster_pca.reset_index(drop=True)
        pca_coords = cluster_pca_reset[['PC1', 'PC2']].round(6)  # Round to handle floating point precision
        cluster_df = cluster_df.reset_index(drop=True).copy()
        # Convert to list of Python floats to avoid numpy type issues
        cluster_df['pca_key'] = [f"{float(pc1):.6f},{float(pc2):.6f}" 
                                  for pc1, pc2 in zip(pca_coords['PC1'], pca_coords['PC2'])]
        
        # Update stored reference
        self.app.current_cluster_df = cluster_df
        
        # Count molecules per point for visualization
        point_counts = cluster_df.groupby('pca_key').size().to_dict()
        
        # Hover cursor for showing info
        cursor = mplcursors.cursor(scatter, hover=True)
        
        @cursor.connect("add")
        def on_hover(sel):
            # Find all molecules at this point
            idx = sel.index
            if idx >= len(cluster_df):
                return
            point_key = cluster_df.iloc[idx]['pca_key']
            mols_at_point = cluster_df[cluster_df['pca_key'] == point_key]
            
            if len(mols_at_point) > 1:
                sel.annotation.set_text(f"{len(mols_at_point)} molecules\nat this point\n(Click to view all)")
                sel.annotation.get_bbox_patch().set(facecolor='yellow', alpha=0.8)
            else:
                sel.annotation.set_text(f"Click to view\nmolecule")
                sel.annotation.get_bbox_patch().set(facecolor='lightblue', alpha=0.8)
        
        # Separate click handler using matplotlib's event system
        def on_scatter_click(event):
            if event.inaxes != ax:
                return
            
            # Find closest point to click
            if event.xdata is None or event.ydata is None:
                return
            
            click_point = np.array([event.xdata, event.ydata])
            points = cluster_pca_reset[['PC1', 'PC2']].values
            distances = np.linalg.norm(points - click_point, axis=1)
            closest_idx = np.argmin(distances)
            
            # Check if click is close enough to a point
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            threshold = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 0.05
            
            if distances[closest_idx] < threshold:
                # Find all molecules at this point
                point_key = cluster_df.iloc[closest_idx]['pca_key']
                mols_at_point_indices = cluster_df[cluster_df['pca_key'] == point_key].index.tolist()
                
                print(f"DEBUG: Clicked on point with {len(mols_at_point_indices)} molecules")
                print(f"DEBUG: Indices: {mols_at_point_indices}")
                print(f"DEBUG: Point key: {point_key}")
                
                # Display all molecules at this point with navigation
                self.app.gui.display_multiple_molecules(mols_at_point_indices, cluster_df)
        
        # Connect click event
        self.app.gui.canvas.mpl_connect('button_press_event', on_scatter_click)
        
        plt.colorbar(scatter, ax=ax, label='Density')
        ax.set_xlabel('Principal Component 1', fontsize=12, weight='bold')
        ax.set_ylabel('Principal Component 2', fontsize=12, weight='bold')
        ax.set_title(f'Cluster {int(cluster_id)} Detail View | {len(cluster_df)} molecules\nClick points to view • Hover for count',
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
        
        # Optionally hide singletons
        if self.app.hide_singletons_var.get():
            range_mask = range_mask & (self.app.centroid_pca['size'] > 1)
        
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
                    overview_clusters.append(f"  Cluster {int(cluster_id)}: {int(size)} molecules ({percentage:.1f}%)")
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
        results = [f"CLUSTER {int(cluster_id)} DETAILS", "=" * 30, f"\nCluster size: {len(cluster_df)} molecules"]
        
        if 'Name' in cluster_df.columns:
            results.append("\nSample molecules:")
            for i, (idx, row) in enumerate(cluster_df.head(10).iterrows()):
                name = row.get('Name', f'Molecule_{idx}')
                results.append(f"  {i+1}. {name}: {row['SMILES']}")
        else:
            results.extend(["\nSample SMILES:", *[f"  {i+1}. {smiles}" for i, smiles in enumerate(cluster_df['SMILES'].head(10))]])
        
        if len(cluster_df) > 10:
            results.append(f"  ... and {len(cluster_df) - 10} more molecules")
        
        results.extend(["\n🔍 Click points to view molecules • Use ◀▶ to navigate overlapping molecules →"])
        
        self.app.gui.results_text.delete(1.0, tk.END)
        self.app.gui.results_text.insert(tk.END, "\n".join(results))