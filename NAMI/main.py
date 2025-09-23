import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
import bitbirch.bitbirch as bb
from tqdm.auto import tqdm
import threading
from scipy.stats import gaussian_kde
import mplcursors
from PIL import Image, ImageTk
import io
from gui_utils import GUIComponents
from data_processing import Data_Processing
from data_visualization import Data_Visualization

class BitBirchPCAGUI:
    def __init__(self, root):
        self.root = root
        root.title("BitBirch PCA Clustering Visualizer")
        root.geometry("1600x900")
        
        # Initialize all data attributes
        self.data = self.X = self.pca_result = self.brc = None
        self.cluster_assignments = self.centroids = self.centroid_pca = None
        self.molecule_pca = None  # Added for storing individual molecule PCA coordinates
        self.current_view = 'overview'
        self.selected_cluster = None
        self.zoom_pan = None
        
        # Parameter variables
        self.threshold_var = tk.StringVar(value="0.65")
        self.branching_var = tk.StringVar(value="50")
        self.radius_var = tk.StringVar(value="2")
        self.nbits_var = tk.StringVar(value="1024")
        self.min_cluster_size_var = tk.StringVar(value="10")
        self.max_cluster_size_var = tk.StringVar(value="1000")

        # Initialize components
        self.gui = GUIComponents(self)
        self.processor = Data_Processing(self)
        self.visualizer = Data_Visualization(self)
        
        # Create the GUI
        self.gui.create_widgets()
        
    # Delegate methods to appropriate components
    def load_smiles_file(self):
        return self.processor.load_smiles_file()
    
    def start_processing(self):
        return self.processor.start_processing()
    
    def save_results(self):
        return self.processor.save_results()
    
    def load_results(self):
        return self.processor.load_results()
    
    def show_overview(self):
        return self.visualizer.show_overview()
    
    def refresh_current_view(self):
        return self.gui.refresh_current_view()
    
    def reset_zoom(self):
        return self.gui.reset_zoom()
    
    def display_data_info(self):
        return self.gui.display_data_info()

def main():
    root = tk.Tk()
    app = BitBirchPCAGUI(root)
    root.mainloop()

if __name__ == "__main__":

    main()
