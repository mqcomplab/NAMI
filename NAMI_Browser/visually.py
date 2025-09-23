import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.decomposition import PCA
import plotly.express as px
import base64
from tqdm.auto import tqdm
tqdm.pandas()
import bitbirch.bitbirch as bb

# Edit input parameters here!
smiles_url = "https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/main/data/cluster_test.smi"
X = None # If you have precomputed fingerprints, set this to your array of fingerprints
do_bitbirch = True  # Set to False if you want to skip the BitBirch clustering step and use your own cluster assignments.
fit_predict_array = None # If ``do_bitbirch`` is False, please set this to cluster assignments array.
# End of input parameters

df = pd.read_csv(smiles_url, sep=" ", names=["SMILES", "Name"])
# Fingerprint conversion
def mol2fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

overlap_cache = {}  # for storing overlapping molecules per point

if X is None:
    df['mol'] = df.SMILES.progress_apply(Chem.MolFromSmiles)
    df['fp'] = df.mol.progress_apply(mol2fp)
    X = np.stack(df.fp.apply(lambda x: np.array(list(x.ToBitString()), dtype=np.uint64)))

if do_bitbirch:
    # Prepare BB tree
    bb.set_merge('diameter')
    threshold = 0.65
    brc = bb.BitBirch(branching_factor=50, threshold=threshold)

    # Fit fps
    brc.fit(X)

    # Get cluster indices
    clust_indices = brc.get_cluster_mol_ids()

    # Create template array for cluster assignments
    fit_predict_array = np.ones(X.shape[0], dtype='int64') * -1

    # Assign labels
    for label, cluster in enumerate(clust_indices):
        fit_predict_array[cluster] = label

df['cluster'] = fit_predict_array
num_clusters = df['cluster'].nunique()
print(f"Number of clusters: {num_clusters}")

# PCA on centroids
def compute_pca(data):
    if len(data) < 2:
        return pd.DataFrame()
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1
    data = data / norms
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(data)
    return pd.DataFrame(transformed, columns=['PC1', 'PC2'])

centroids = brc.get_centroids()
centroid_pca = compute_pca(centroids)
centroid_pca['cluster'] = range(num_clusters)
centroid_pca['size'] = df['cluster'].value_counts().sort_index().values
centroid_pca['hover'] = centroid_pca['cluster'].apply(lambda x: f"Cluster {x}")

# Helper: draw molecule to base64 PNG
def mol_to_img_tag(mol, size=(280, 280)):
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_bytes = drawer.GetDrawingText()
    return f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"

# Colorscale and discrete colors for clusters
colorscale = px.colors.sequential.Plasma
unique_clusters = centroid_pca['cluster'].unique()
cluster_colors = px.colors.qualitative.Dark24
color_map = {cluster: cluster_colors[i % len(cluster_colors)] for i, cluster in enumerate(unique_clusters)}

# Dash App
app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H3("Centroid PCA Plot", style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'}),
            dcc.Graph(
                id='centroid-pca-plot',
                figure=px.scatter(
                    centroid_pca,
                    x='PC1', y='PC2',
                    color='cluster',
                    size='size',
                    hover_name='hover',
                    color_discrete_map=color_map,
                    title="Click on a cluster centroid"
                ).update_layout(
                    plot_bgcolor='#f9f9f9',
                    paper_bgcolor='#f9f9f9',
                    font=dict(size=14),
                    margin=dict(t=50, l=50, r=50, b=50),
                    xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False, showline=False),
                    yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False, showline=False),
                    shapes=[{
                        'type': 'rect',
                        'xref': 'paper', 'yref': 'paper',
                        'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1,
                        'line': {'color': 'LightGray', 'width': 1},
                        'fillcolor': 'white',
                        'layer': 'below',
                        'opacity': 0.2
                    }]
                ).update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            )
        ], className='card', style={'width': '60%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.H3("Representative Molecule of Cluster", style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'}),
            html.Img(id='cluster-representative-img', style={"border": "1px solid gray", "width": "200px", 'borderRadius': '10px', 'display': 'block', 'margin': '0 auto'}),
            html.Div([
                html.P(id='cluster-smiles', style={'fontSize': 16, 'textAlign': 'center'}),
                html.P(id='cluster-id', style={'fontSize': 16, 'textAlign': 'center'}),
                html.P(id='cluster-size', style={'fontSize': 16, 'textAlign': 'center', 'fontWeight': '600', 'color': '#007aff'})
            ])
        ], className='card', style={'width': '35%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'})
    ], style={'display': 'flex', 'marginBottom': '30px'}),

    html.Div([
        html.Div([
            html.H3("PCA of Cluster Molecules", style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'}),
            dcc.Graph(id='cluster-detail-plot')
        ], className='card', style={'width': '60%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.H3("Selected Molecule", style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'}),
            html.Img(id='selected-molecule-img', style={"border": "1px solid gray", "width": "200px", 'borderRadius': '10px', 'display': 'block', 'margin': '0 auto'}),
            html.Div("Select Molecule:", id='dropdown-label', style={'textAlign': 'center', 'fontWeight': 'bold', 'display': 'none'}),
            dcc.Dropdown(id='overlap-dropdown', style={'width': '90%', 'margin': '10px auto', 'display': 'none'}),
            html.Div([
                html.P(id='selected-smiles', style={'fontSize': 16, 'textAlign': 'center'}),
                html.P(id='selected-cluster', style={'fontSize': 16, 'textAlign': 'center'})
            ])
        ], className='card', style={'width': '35%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'})
    ], style={'display': 'flex'})
])

# Store cluster ID for molecule selection
cluster_id_store = {}

@app.callback(
    [Output('cluster-detail-plot', 'figure'),
     Output('cluster-representative-img', 'src'),
     Output('cluster-smiles', 'children'),
     Output('cluster-id', 'children'),
     Output('cluster-size', 'children')],
    Input('centroid-pca-plot', 'clickData')
)
def update_cluster_panel(clickData):
    if not clickData:
        empty_df = pd.DataFrame({'x': [None], 'y': [None]})
        fig = px.scatter(empty_df, x='x', y='y', title="Click a cluster")
        fig.update_layout(
            plot_bgcolor='#f9f9f9',
            paper_bgcolor='#f9f9f9',
            font=dict(size=14),
            margin=dict(t=50, l=50, r=50, b=50),
            xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False, showline=False),
            yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False, showline=False)
        )
        return fig, "", "", "", ""


    try:
        hover = clickData['points'][0]['hovertext']
        cluster_id = int(hover.split()[-1])
        cluster_id_store['id'] = cluster_id

        sub_df = df[df['cluster'] == cluster_id].reset_index()
        X_cluster = np.stack(sub_df.fp.apply(lambda x: np.array(list(x.ToBitString()), dtype=np.uint8)))
        pca_df = compute_pca(X_cluster)
        pca_df['SMILES'] = sub_df.SMILES

        if len(sub_df) == 1:
            mol = sub_df.iloc[0]['mol']
            smiles = sub_df.iloc[0]['SMILES']
            img = mol_to_img_tag(mol)
            cluster_size_str = "Cluster Size: 1 molecule"

            # Create a placeholder PCA plot with just one point
            fig = px.scatter(x=[0], y=[0], title=f"PCA of Cluster {cluster_id} (1 molecule)")
            fig.update_layout(
                plot_bgcolor='#f9f9f9',
                paper_bgcolor='#f9f9f9',
                font=dict(size=14),
                margin=dict(t=50, l=50, r=50, b=50),
                xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False, showline=False),
                yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False, showline=False)
            )

            return fig, img, f"SMILES: {smiles}", f"Cluster ID: {cluster_id}", cluster_size_str
        # Robust binning even with small PCA components
        scale = 10000  # 1/0.001 bin size
        pca_df['PC1_bin'] = (pca_df['PC1'] * scale).round().astype(int)
        pca_df['PC2_bin'] = (pca_df['PC2'] * scale).round().astype(int)

        bin_counts = pca_df.groupby(['PC1_bin', 'PC2_bin']).size()
        pca_df['density'] = list(zip(pca_df['PC1_bin'], pca_df['PC2_bin']))
        pca_df['density'] = pca_df['density'].map(bin_counts)

        # Drop temp columns
        pca_df.drop(columns=['PC1_bin', 'PC2_bin'], inplace=True)


        # Plot with density color
        fig = px.scatter(
            pca_df,
            x='PC1', y='PC2',
            color='density',
            hover_name='SMILES',
            color_continuous_scale='pinkyl',  # Different than Plasma
            title=f"PCA of Cluster {cluster_id}"
        )

        # Integer ticks on colorbar
        min_density = int(pca_df['density'].min())
        max_density = int(pca_df['density'].max())
        
        # Calculate axis ranges
        x_min, x_max = pca_df['PC1'].min(), pca_df['PC1'].max()
        y_min, y_max = pca_df['PC2'].min(), pca_df['PC2'].max()

        x_range = x_max - x_min
        y_range = y_max - y_min

        x_pad = max(x_range * 0.1, 0.001)
        y_pad = max(y_range * 0.1, 0.001)
        
        # Force range to 0.001 if too narrow
        if x_range < 0.001:
            x_center = (x_max + x_min) / 2
            x_min = x_center - 0.0005
            x_max = x_center + 0.0005

        if y_range < 0.001:
            y_center = (y_max + y_min) / 2
            y_min = y_center - 0.0005
            y_max = y_center + 0.0005

        from numpy import linspace

        tick_min = int(pca_df['density'].min())
        tick_max = int(pca_df['density'].max())

        tick_count = 5  # or whatever number you want
        tick_vals = np.linspace(tick_min, tick_max, tick_count, dtype=int)
        tick_vals = np.unique(tick_vals)  # remove duplicates
        # Update layout with optional fixed range
        fig.update_layout(
            plot_bgcolor='#f9f9f9',
            paper_bgcolor='#f9f9f9',
            font=dict(size=14),
            margin=dict(t=50, l=50, r=50, b=50),
            xaxis=dict(
                range=[x_min - x_pad, x_max + x_pad],
                showgrid=True, gridcolor='lightgray',
                zeroline=False, showline=False
            ),
            yaxis=dict(
                range=[y_min - y_pad, y_max + y_pad],
                showgrid=True, gridcolor='lightgray',
                zeroline=False, showline=False
            ),
            coloraxis_colorbar=dict(
                title='Overlap Density',
                tickmode='array',
                tickvals=tick_vals,
                ticktext=[str(t) for t in tick_vals]
            )

        )

        fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color='black')))

        # Representative mol
        mol = sub_df.iloc[0]['mol']
        smiles = sub_df.iloc[0]['SMILES']
        img = mol_to_img_tag(mol)
        cluster_size_str = f"Cluster Size: {len(sub_df)} molecules"

        return fig, img, f"SMILES: {smiles}", f"Cluster ID: {cluster_id}", cluster_size_str

    except Exception as e:
        return px.scatter(title=f"Error: {e}"), "", "", "", ""
overlap_cache = {}  # for storing overlapping molecules per point


# Callback: update molecule from detailed PCA plot click
# Callback: update molecule from detailed PCA plot click
@app.callback(
    [Output('selected-molecule-img', 'src'),
     Output('selected-smiles', 'children'),
     Output('selected-cluster', 'children'),
     Output('overlap-dropdown', 'options'),
     Output('overlap-dropdown', 'value'),
     Output('overlap-dropdown', 'style'),
     Output('dropdown-label', 'style')],
    Input('cluster-detail-plot', 'clickData')
)

def update_selected_molecule(clickData):
    if not clickData:
        return "", "", "", [], None, {'display': 'none'}, {'display': 'none'}

    try:
        cluster_id = cluster_id_store.get('id', None)
        if cluster_id is None:
            return "", "", "", [], None, {'display': 'none'}, {'display': 'none'}

        sub_df = df[df['cluster'] == cluster_id].reset_index(drop=True)
        X_cluster = np.stack(sub_df.fp.apply(lambda x: np.array(list(x.ToBitString()), dtype=np.uint8)))
        pca_df = compute_pca(X_cluster)
        pca_df['SMILES'] = sub_df['SMILES']
        pca_df['mol'] = sub_df['mol']

        clicked_smiles = clickData['points'][0]['hovertext']
        clicked_row = pca_df[pca_df['SMILES'] == clicked_smiles].iloc[0]
        x, y = round(clicked_row['PC1'], 5), round(clicked_row['PC2'], 5)

        overlaps = []
        for _, row in pca_df.iterrows():
            if round(row['PC1'], 5) == x and round(row['PC2'], 5) == y:
                overlaps.append((row['SMILES'], row['mol']))

        key = f"{x}_{y}"
        overlap_cache[key] = overlaps

        if len(overlaps) > 1:
            dropdown_options = [{'label': smi, 'value': smi} for smi, _ in overlaps]
            return mol_to_img_tag(clicked_row['mol']), f"SMILES: {clicked_row['SMILES']}", f"Cluster: {cluster_id}", dropdown_options, clicked_row['SMILES'], {'display': 'block'}, {'display': 'block'}
        else:
            return (
                mol_to_img_tag(clicked_row['mol']),
                f"SMILES: {clicked_row['SMILES']}",
                f"Cluster: {cluster_id}",
                [], None,
                {'display': 'none'},
                {'display': 'none'}  # hide the label too
            )

    except Exception as e:
        print(f"Error in update_selected_molecule: {e}")
        return "", "", "", [], None, {'display': 'none'}, {'display': 'none'}
    
from dash.exceptions import PreventUpdate

@app.callback(
    Output('selected-molecule-img', 'src', allow_duplicate=True),
    Input('overlap-dropdown', 'value'),
    prevent_initial_call=True
)
def update_selected_from_dropdown(smiles_selected):
    if not smiles_selected:
        raise PreventUpdate

    for key in overlap_cache:
        for smi, mol in overlap_cache[key]:
            if smi == smiles_selected:
                return mol_to_img_tag(mol)
    return ""


if __name__ == '__main__':
    app.run(debug=True)
    # http://127.0.0.1:8050/