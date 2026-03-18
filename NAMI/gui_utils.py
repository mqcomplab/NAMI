import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
from PIL import Image, ImageTk
import io
import psutil
import os

class ZoomPan:
    def __init__(self, canvas, ax):
        self.canvas, self.ax, self.press, self.zoom_factor = canvas, ax, None, 1.1
        for event, handler in [('button_press_event', self.on_press), ('button_release_event', self.on_release), 
                              ('motion_notify_event', self.on_motion), ('scroll_event', self.on_scroll)]:
            canvas.mpl_connect(event, handler)
    
    def on_press(self, event):
        if event.inaxes == self.ax and event.button == 1: 
            self.press = (event.xdata, event.ydata)
    
    def on_motion(self, event):
        if self.press and event.inaxes == self.ax and event.button == 1:
            dx, dy = event.xdata - self.press[0], event.ydata - self.press[1]
            xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
            self.ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            self.ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            self.canvas.draw_idle()
    
    def on_release(self, event): 
        self.press = None
    
    def on_scroll(self, event):
        if event.inaxes != self.ax: return
        scale = 1/self.zoom_factor if event.step > 0 else self.zoom_factor
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        x_range, y_range = (xlim[1] - xlim[0]) * scale, (ylim[1] - ylim[0]) * scale
        new_xlim = (event.xdata - x_range/2, event.xdata + x_range/2)
        new_ylim = (event.ydata - y_range/2, event.ydata + y_range/2)
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw_idle()
    
    def reset_view(self):
        self.ax.relim()
        self.ax.autoscale()
        self.canvas.draw()

class GUIComponents:
    def __init__(self, parent_app):
        self.app = parent_app
        self.process = psutil.Process(os.getpid())
        self.peak_memory_mb = 0
        self.memory_update_active = False

    def create_widgets(self):
        main_frame = ttk.Frame(self.app.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.app.root.grid_rowconfigure(0, weight=1)
        self.app.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        
        # Control panel
        ctrl = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        ctrl.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        
        # Row 0: File and main parameters
        ttk.Button(ctrl, text="Load SMILES CSV", command=self.app.load_smiles_file).grid(row=0, column=0, padx=(0, 10))
        self.file_label = ttk.Label(ctrl, text="No file loaded")
        self.file_label.grid(row=0, column=1, padx=(0, 20))

        for i, (label, var) in enumerate([("BB Threshold:", self.app.threshold_var), ("Branching Factor:", self.app.branching_var)]):
            ttk.Label(ctrl, text=label).grid(row=0, column=2+i*2, padx=(0, 5))
            ttk.Entry(ctrl, textvariable=var, width=8).grid(row=0, column=3+i*2, padx=(0, 10))
        
        # Row 1: Fingerprint and cluster size parameters
        params = [("FP Radius:", self.app.radius_var), ("FP Bits:", self.app.nbits_var), ("Min Large Cluster:", self.app.min_cluster_size_var)]
        for i, (label, var) in enumerate(params):
            ttk.Label(ctrl, text=label).grid(row=1, column=i*2, padx=(0, 5), pady=(5, 0))
            ttk.Entry(ctrl, textvariable=var, width=8).grid(row=1, column=i*2+1, padx=(0, 10), pady=(5, 0))
        
        # Row 2: More cluster parameters
        params2 = [("Max Large Cluster:", self.app.max_cluster_size_var)]
        for i, (label, var) in enumerate(params2):
            ttk.Label(ctrl, text=label).grid(row=2, column=i*2, padx=(0, 5), pady=(5, 0))
            ttk.Entry(ctrl, textvariable=var, width=8).grid(row=2, column=i*2+1, padx=(0, 10), pady=(5, 0))
        
        # Hide singletons checkbox (affects both display and PCA memory usage)
        self.hide_singletons_check = ttk.Checkbutton(ctrl, text="Hide singletons (saves memory)", variable=self.app.hide_singletons_var, 
                                                      command=self.app.show_overview)
        self.hide_singletons_check.grid(row=2, column=2, columnspan=2, padx=(10, 0), pady=(5, 0), sticky="w")
        
        # Parallel clustering checkbox (for large datasets)
        self.parallel_check = ttk.Checkbutton(ctrl, text="Parallel clustering (>1M mols)", 
                                              variable=self.app.use_parallel_clustering_var)
        self.parallel_check.grid(row=2, column=4, padx=(10, 5), pady=(5, 0), sticky="w")
        ttk.Label(ctrl, text="#Procs:").grid(row=2, column=5, padx=(0, 5), pady=(5, 0))
        ttk.Entry(ctrl, textvariable=self.app.parallel_num_processes_var, width=5).grid(row=2, column=6, padx=(0, 10), pady=(5, 0))
        
        # Row 3: Advanced options
        self.incremental_pca_check = ttk.Checkbutton(ctrl, text="Compute full PCA (incremental, memory-efficient)", 
                                                      variable=self.app.use_incremental_pca_var)
        self.incremental_pca_check.grid(row=3, column=0, columnspan=3, padx=(0, 10), pady=(5, 0), sticky="w")
        
        # Row 4: Action buttons
        self.process_btn = ttk.Button(ctrl, text="Process & Cluster", command=self.app.start_processing, state="disabled")
        self.save_btn = ttk.Button(ctrl, text="Save Results", command=self.app.save_results, state="disabled")
        self.load_btn = ttk.Button(ctrl, text="Load Results", command=self.app.load_results)
        self.back_btn = ttk.Button(ctrl, text="← Back to Overview", command=self.app.show_overview, state="disabled")
        self.refresh_btn = ttk.Button(ctrl, text="Refresh View", command=self.refresh_current_view, state="disabled")
        self.reset_zoom_btn = ttk.Button(ctrl, text="Reset Zoom", command=self.reset_zoom, state="disabled")

        for i, btn in enumerate([self.process_btn, self.save_btn, self.load_btn, self.back_btn, self.refresh_btn, self.reset_zoom_btn]):
            btn.grid(row=4, column=i, padx=(0, 10), pady=(5, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(ctrl, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=6, sticky="ew", pady=(10, 0))
        
        # Create text widgets with scrollbars
        for i, (title, attr) in enumerate([("Data Information", "info_text"), ("Clustering Results", "results_text")]):
            frame = ttk.LabelFrame(main_frame, text=title, padding="5")
            frame.grid(row=1+i, column=0, sticky="nsew")
            frame.grid_columnconfigure(0, weight=1)
            text = tk.Text(frame, height=8 if i == 0 else 12, width=35)
            scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
            text.configure(yscrollcommand=scrollbar.set)
            text.grid(row=0, column=0, sticky="nsew")
            scrollbar.grid(row=0, column=1, sticky="ns")
            setattr(self, attr, text)
        
        # Plot frame
        plot_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="5")
        plot_frame.grid(row=1, column=1, rowspan=2, sticky="nsew", padx=(10, 10))
        plot_frame.grid_columnconfigure(0, weight=1)
        plot_frame.grid_rowconfigure(0, weight=1)
        
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.canvas.mpl_connect('button_press_event', self.app.visualizer.on_plot_click)
        
        ttk.Label(plot_frame, text="🔍 Scroll to zoom • Drag to pan • Click clusters to explore", 
                 font=('Arial', 9), foreground='gray').grid(row=2, column=0, pady=(5, 0))
        
        # Molecule information frame
        mol_frame = ttk.LabelFrame(main_frame, text="Molecule Information", padding="5")
        mol_frame.grid(row=1, column=2, sticky="nsew")
        mol_frame.grid_columnconfigure(0, weight=1)
        mol_frame.grid_rowconfigure(2, weight=1)  # Row 2 is the scrollable text area
        
        self.mol_canvas = tk.Canvas(mol_frame, width=200, height=200, bg='white')
        self.mol_canvas.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        # Navigation controls for multiple molecules at same point
        nav_frame = ttk.Frame(mol_frame)
        nav_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        
        self.prev_mol_btn = ttk.Button(nav_frame, text="◀ Previous", command=self.show_prev_molecule, state="disabled")
        self.prev_mol_btn.pack(side=tk.LEFT, padx=2)
        
        self.mol_counter_label = ttk.Label(nav_frame, text="", font=('Arial', 9))
        self.mol_counter_label.pack(side=tk.LEFT, padx=10, expand=True)
        
        self.next_mol_btn = ttk.Button(nav_frame, text="Next ▶", command=self.show_next_molecule, state="disabled")
        self.next_mol_btn.pack(side=tk.RIGHT, padx=2)
        
        self.mol_text = tk.Text(mol_frame, height=8, width=30, wrap=tk.WORD)
        mol_scrollbar = ttk.Scrollbar(mol_frame, orient="vertical", command=self.mol_text.yview)
        self.mol_text.configure(yscrollcommand=mol_scrollbar.set)
        self.mol_text.grid(row=2, column=0, sticky="nsew")
        mol_scrollbar.grid(row=2, column=1, sticky="ns")
        
        # Initialize state for multiple molecules
        self.current_mol_list = []
        self.current_mol_index = 0
        self.current_cluster_df = None
        
        # Additional details frame
        add_frame = ttk.LabelFrame(main_frame, text="Additional Details", padding="5")
        add_frame.grid(row=2, column=2, sticky="nsew")
        add_frame.grid_columnconfigure(0, weight=1)
        
        self.additional_text = tk.Text(add_frame, height=12, width=30)
        add_scrollbar = ttk.Scrollbar(add_frame, orient="vertical", command=self.additional_text.yview)
        self.additional_text.configure(yscrollcommand=add_scrollbar.set)
        self.additional_text.grid(row=0, column=0, sticky="nsew")
        add_scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Memory monitor at bottom right
        memory_frame = ttk.Frame(add_frame)
        memory_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.memory_label = ttk.Label(memory_frame, text="Memory: 0 MB | Peak: 0 MB", 
                                      font=('Arial', 8), foreground='#666')
        self.memory_label.pack(anchor='e', padx=5)
        
        self.clear_molecule_info()
        
        # Start memory monitoring
        self.memory_update_active = True
        self.update_memory_display()

    def setup_zoom_pan(self, ax):
        self.zoom_pan = ZoomPan(self.canvas, ax)
        self.reset_zoom_btn.config(state="normal")
    
    def reset_zoom(self):
        if hasattr(self, 'zoom_pan') and self.zoom_pan: 
            self.zoom_pan.reset_view()
    
    def clear_molecule_info(self):
        self.mol_canvas.delete("all")
        self.mol_canvas.create_text(100, 100, text="Click a molecule\nto see details", 
                                   fill="gray", font=("Arial", 12), justify="center")
        self.mol_text.delete(1.0, tk.END)
        self.mol_text.insert(tk.END, "No molecule selected.\n\nClick molecules in the detail view to see their information here.")
        self.additional_text.delete(1.0, tk.END)
        self.additional_text.insert(tk.END, "Additional molecule properties and cluster statistics will appear here when available.")
        
        # Reset navigation
        self.current_mol_list = []
        self.current_mol_index = 0
        self.current_cluster_df = None
        self.prev_mol_btn.config(state="disabled")
        self.next_mol_btn.config(state="disabled")
        self.mol_counter_label.config(text="")
    
    def display_molecule_info(self, mol_index, mol, smiles, cluster_df):
        self.mol_canvas.delete("all")
        self.mol_text.delete(1.0, tk.END)
        self.additional_text.delete(1.0, tk.END)
        
        # Draw molecule
        if mol:
            try:
                img = Draw.MolToImage(mol, size=(180, 180))
                bio = io.BytesIO()
                img.save(bio, format="PNG")
                bio.seek(0)
                tk_img = ImageTk.PhotoImage(Image.open(bio))
                self.mol_canvas.image = tk_img
                self.mol_canvas.create_image(100, 100, image=tk_img)
            except Exception as e:
                self.mol_canvas.create_text(100, 100, text=f"Error:\n{str(e)}", fill="red", font=("Arial", 10), justify="center")
        
        # Molecule details
        info = [f"Molecule Index: {mol_index}"]
        if 'Name' in cluster_df.columns:
            info.append(f"Name: {cluster_df.iloc[mol_index]['Name'] if mol_index < len(cluster_df) else f'Molecule_{mol_index}'}")
        info.extend([f"\nSMILES:", smiles])
        
        if mol:
            try:
                info.extend([f"\nMolecular Properties:",
                           f"Formula: {rdMolDescriptors.CalcMolFormula(mol)}",
                           f"Mol Weight: {rdMolDescriptors.CalcExactMolWt(mol):.2f}",
                           f"Heavy Atoms: {mol.GetNumHeavyAtoms()}",
                           f"Rings: {rdMolDescriptors.CalcNumRings(mol)}",
                           f"\nLipinski Properties:",
                           f"LogP: {rdMolDescriptors.CalcCrippenDescriptors(mol)[0]:.2f}",
                           f"HBA: {rdMolDescriptors.CalcNumHBA(mol)}",
                           f"HBD: {rdMolDescriptors.CalcNumHBD(mol)}",
                           f"TPSA: {rdMolDescriptors.CalcTPSA(mol):.2f}"])
            except Exception as e:
                info.append(f"\nError calculating properties: {str(e)}")
        
        self.mol_text.insert(tk.END, "\n".join(info))
        
        # Additional info
        add_info = ["Cluster Statistics:"]
        if hasattr(self.app, 'selected_cluster') and self.app.selected_cluster is not None:
            add_info.extend([f"Cluster ID: {int(self.app.selected_cluster)}", f"Cluster Size: {len(cluster_df)}"])
        
        self.additional_text.insert(tk.END, "\n".join(add_info))

    def display_multiple_molecules(self, mol_indices, cluster_df):
        """Display multiple molecules at the same point with navigation"""
        if not mol_indices:
            return
            
        self.current_mol_list = mol_indices
        self.current_mol_index = 0
        self.current_cluster_df = cluster_df
        
        # Enable/disable navigation buttons
        if len(mol_indices) > 1:
            self.next_mol_btn.config(state="normal")
            self.prev_mol_btn.config(state="disabled" if self.current_mol_index == 0 else "normal")
            self.mol_counter_label.config(text=f"Molecule 1 of {len(mol_indices)}")
        else:
            self.next_mol_btn.config(state="disabled")
            self.prev_mol_btn.config(state="disabled")
            self.mol_counter_label.config(text="1 molecule")
        
        # Display first molecule
        mol_idx = mol_indices[0]
        from rdkit import Chem
        mol = cluster_df.iloc[mol_idx]['mol']
        smiles = cluster_df.iloc[mol_idx]['SMILES']
        self.display_molecule_info(mol_idx, mol, smiles, cluster_df)
        
        # Update additional info using helper function
        self._update_additional_info_for_navigation()
    
    def show_next_molecule(self):
        """Show next molecule in the current list"""
        if not self.current_mol_list or self.current_mol_index >= len(self.current_mol_list) - 1:
            return
        
        self.current_mol_index += 1
        mol_idx = self.current_mol_list[self.current_mol_index]
        
        from rdkit import Chem
        mol = self.current_cluster_df.iloc[mol_idx]['mol']
        smiles = self.current_cluster_df.iloc[mol_idx]['SMILES']
        self.display_molecule_info(mol_idx, mol, smiles, self.current_cluster_df)
        
        # Update counter and buttons
        self.mol_counter_label.config(text=f"Molecule {self.current_mol_index + 1} of {len(self.current_mol_list)}")
        self.prev_mol_btn.config(state="normal")
        if self.current_mol_index >= len(self.current_mol_list) - 1:
            self.next_mol_btn.config(state="disabled")
        
        # Update additional info to highlight current molecule
        self._update_additional_info_for_navigation()
    
    def show_prev_molecule(self):
        """Show previous molecule in the current list"""
        if not self.current_mol_list or self.current_mol_index <= 0:
            return
        
        self.current_mol_index -= 1
        mol_idx = self.current_mol_list[self.current_mol_index]
        
        from rdkit import Chem
        mol = self.current_cluster_df.iloc[mol_idx]['mol']
        smiles = self.current_cluster_df.iloc[mol_idx]['SMILES']
        self.display_molecule_info(mol_idx, mol, smiles, self.current_cluster_df)
        
        # Update counter and buttons
        self.mol_counter_label.config(text=f"Molecule {self.current_mol_index + 1} of {len(self.current_mol_list)}")
        self.next_mol_btn.config(state="normal")
        if self.current_mol_index <= 0:
            self.prev_mol_btn.config(state="disabled")
        
        # Update additional info to highlight current molecule
        self._update_additional_info_for_navigation()
    
    def _update_additional_info_for_navigation(self):
        """Update the additional info panel to highlight the current molecule"""
        if not self.current_mol_list or not self.current_cluster_df is not None:
            return
        
        self.additional_text.delete(1.0, tk.END)
        add_info = ["Cluster Statistics:"]
        if hasattr(self.app, 'selected_cluster') and self.app.selected_cluster is not None:
            add_info.extend([f"Cluster ID: {int(self.app.selected_cluster)}", 
                           f"Cluster Size: {len(self.current_cluster_df)}"])
        
        if len(self.current_mol_list) > 1:
            add_info.extend([f"\n{'='*30}",
                           f"MOLECULES AT THIS POINT: {len(self.current_mol_list)}",
                           f"{'='*30}",
                           f"\nThese molecules have identical",
                           f"or very similar fingerprints.\n"])
            
            # List all molecules at this point, highlighting the current one
            for i, idx in enumerate(self.current_mol_list):
                mol_data = self.current_cluster_df.iloc[idx]
                name = mol_data.get('Name', f'Mol_{idx}')
                smiles_short = mol_data['SMILES'][:50] + '...' if len(mol_data['SMILES']) > 50 else mol_data['SMILES']
                
                # Highlight current molecule
                if i == self.current_mol_index:
                    add_info.append(f"▶ {i+1}. {name} ◀ VIEWING")
                else:
                    add_info.append(f"{i+1}. {name}")
                add_info.append(f"   {smiles_short}")
            
            add_info.append(f"\nUse ◀▶ buttons to navigate")
        else:
            add_info.append(f"\nSingle molecule at this point")
        
        self.additional_text.insert(tk.END, "\n".join(add_info))

    def refresh_current_view(self):
        if self.app.current_view == 'overview':
            self.app.show_overview()
        elif self.app.current_view == 'cluster' and self.app.selected_cluster is not None:
            self.app.visualizer.show_cluster_detail(self.app.selected_cluster)
        self.app.visualizer.display_clustering_results()
    
    def update_memory_display(self):
        """Update memory usage display every 2 seconds"""
        if not self.memory_update_active:
            return
        
        try:
            # Get current memory usage in MB
            current_memory_mb = self.process.memory_info().rss / (1024 * 1024)
            
            # Track peak memory
            if current_memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = current_memory_mb
            
            # Update label
            self.memory_label.config(
                text=f"Memory: {current_memory_mb:.0f} MB | Peak: {self.peak_memory_mb:.0f} MB"
            )
            
            # Color coding based on usage
            if current_memory_mb < 1000:
                color = '#2ecc71'  # Green
            elif current_memory_mb < 5000:
                color = '#f39c12'  # Orange
            else:
                color = '#e74c3c'  # Red
            self.memory_label.config(foreground=color)
            
        except Exception:
            pass
        
        # Schedule next update (2 seconds)
        if self.memory_update_active:
            self.app.root.after(2000, self.update_memory_display)

    def display_data_info(self):
        if self.app.data is None: 
            return
        
        info = [f"Dataset Shape: {self.app.data.shape}", f"Columns: {list(self.app.data.columns)}"]
        if 'SMILES' in self.app.data.columns:
            info.append("\nSMILES Examples:")
            info.extend([f"  {i+1}: {smiles}" for i, smiles in enumerate(self.app.data['SMILES'].head(5))])
        
        if hasattr(self.app, 'X') and self.app.X is not None:
            info.append(f"\nFingerprint Matrix: {self.app.X.shape}")
        
        if hasattr(self.app, 'cluster_assignments') and self.app.cluster_assignments is not None:
            unique_clusters = np.unique(self.app.cluster_assignments[self.app.cluster_assignments >= 0])
            info.extend([f"\nNumber of Clusters: {len(unique_clusters)}", 
                        f"Noise Points: {np.sum(self.app.cluster_assignments == -1)}"])
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "\n".join(info))