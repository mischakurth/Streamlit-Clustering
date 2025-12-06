import streamlit as st
# --- PyArrow Monkeypatch for stlite/Pyodide ---
# Fixes AttributeError: module 'pyarrow' has no attribute 'RecordBatch'/'ChunkedArray'
import sys
try:
    import pyarrow
    # Define dummy classes for missing attributes to satisfy sklearn checks
    if not hasattr(pyarrow, 'RecordBatch'):
        pyarrow.RecordBatch = type("RecordBatch", (), {})
    if not hasattr(pyarrow, 'ChunkedArray'):
        pyarrow.ChunkedArray = type("ChunkedArray", (), {})
    if not hasattr(pyarrow, 'Table'):
        pyarrow.Table = type("Table", (), {})
    if not hasattr(pyarrow, 'Array'):
        pyarrow.Array = type("Array", (), {})
except ImportError:
    pass
# ----------------------------------------------

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_gen import generate_data
from algorithms import KMeansManual

st.set_page_config(page_title="Clustering Visualizer", layout="wide")

st.title("Clustering Algorithmen Visualisierung")

import json
import time

# --- Callbacks ---

def update_from_slider():
    if 'slider_step' in st.session_state:
        st.session_state['algo_step'] = st.session_state['slider_step']
        st.session_state['autoplay'] = False
        # Update convergence status based on slider
        if 'algo' in st.session_state:
             algo = st.session_state['algo']
             st.session_state['algo_converged'] = (st.session_state['algo_step'] == len(algo.get_history()) - 1)

def prev_step():
    if 'algo' in st.session_state:
        st.session_state['autoplay'] = False
        if st.session_state['algo_step'] > 0:
            st.session_state['algo_step'] -= 1
            st.session_state['algo_converged'] = False
            st.session_state['slider_step'] = st.session_state['algo_step']

def next_step():
    if 'algo' in st.session_state:
        algo = st.session_state['algo']
        st.session_state['autoplay'] = False
        if st.session_state['algo_step'] < len(algo.get_history()) - 1:
            st.session_state['algo_step'] += 1
            st.session_state['slider_step'] = st.session_state['algo_step']
        
        # Check convergence for UI
        st.session_state['algo_converged'] = (st.session_state['algo_step'] == len(algo.get_history()) - 1)

def end_step():
    if 'algo' in st.session_state:
        algo = st.session_state['algo']
        st.session_state['autoplay'] = False
        # algo is already fitted
        st.session_state['algo_step'] = len(algo.get_history()) - 1
        st.session_state['slider_step'] = st.session_state['algo_step']
        st.session_state['algo_converged'] = True

def toggle_autoplay():
    st.session_state['autoplay'] = not st.session_state.get('autoplay', False)




# --- Sidebar: Data Generation ---
st.sidebar.header("1. Daten Generierung")
data_source = st.sidebar.radio("Datenquelle", ["Generieren", "Importieren"])

if data_source == "Generieren":
    data_type = st.sidebar.selectbox("Datensatz Typ", ["uniform", "blobs", "moons", "circles", "aniso"])
    n_samples = st.sidebar.slider("Anzahl Punkte", 100, 4000, 400)
    n_features = 2 # Fixed for 2D visualization
    
    centers = 4
    if data_type in ["blobs", "aniso"]:
        centers = st.sidebar.slider("Anzahl Zentren", 2, 8, 4)
        
    cluster_std = st.sidebar.slider("Streuung / Rauschen", 0.1, 3.0, 1.0)

    if st.sidebar.button("Neue Daten generieren"):
        X, y = generate_data(n_samples, n_features, centers, cluster_std, data_type)
        st.session_state['X'] = X
        st.session_state['y'] = y
        st.session_state['data_generated'] = True
        st.session_state['data_params'] = {
            "type": data_type,
            "n_samples": n_samples,
            "centers": centers if data_type == "blobs" else None,
            "cluster_std": cluster_std
        }
        # Reset algorithm state
        if 'algo' in st.session_state:
            del st.session_state['algo']

elif data_source == "Importieren":
    import_data = st.sidebar.text_area("Daten eingeben (JSON Format: [[x1, y1], ...])", "[[0, 0], [1, 1], [0, 1]]")
    if st.sidebar.button("Daten importieren"):
        try:
            data = json.loads(import_data)
            X = np.array(data)
            if X.ndim != 2 or X.shape[1] != 2:
                st.error("Daten müssen 2-dimensional sein (N x 2).")
            else:
                st.session_state['X'] = X
                st.session_state['y'] = np.zeros(X.shape[0]) # Dummy labels
                st.session_state['data_generated'] = True
                st.session_state['data_params'] = {"type": "Importiert", "n_samples": X.shape[0]}
                # Reset algorithm state
                if 'algo' in st.session_state:
                    del st.session_state['algo']
        except json.JSONDecodeError:
            st.error("Ungültiges JSON Format.")
        except Exception as e:
            st.error(f"Fehler beim Import: {e}")

# Initialize session state if not present
if 'data_generated' not in st.session_state:
    st.session_state['data_generated'] = False
if 'autoplay' not in st.session_state:
    st.session_state['autoplay'] = False

# --- Sidebar: Algorithm Selection ---
st.sidebar.header("2. Algorithmus")
algo_name = st.sidebar.selectbox("Algorithmus", ["K-Means"])

if algo_name == "K-Means":
    k = st.sidebar.slider("K (Anzahl Cluster)", 2, 10, 3)
    max_iter = st.sidebar.slider("Max Iterationen", 10, 1000, 100)
    init_method = st.sidebar.selectbox("Initialisierung", ["Random", "K-Means++"])
    
    col_start, col_step = st.sidebar.columns(2)
    
    if col_start.button("Initialisieren"):
        if st.session_state['data_generated']:
            algo = KMeansManual(k=k, max_iter=max_iter)
            method = "random" if init_method == "Random" else "k-means++"
            algo.initialize(st.session_state['X'], method=method)
            # Pre-calculate everything
            algo.fit(st.session_state['X'])
            
            st.session_state['algo'] = algo
            st.session_state['algo_step'] = 0
            st.session_state['slider_step'] = 0
            # False initially (unless history has only 1 step which means immediate convergence)
            st.session_state['algo_converged'] = (len(algo.get_history()) <= 1)
        else:
            st.error("Bitte zuerst Daten generieren!")

    
    # Main visualization fragment (handles plot, controls, and dynamic sidebar updates)
    @st.fragment
    def render_visualization():
        X = st.session_state['X']
        algo_name = "K-Means" # Hardcoded for now based on current logic, or pass as arg if dynamic
        
        # Pre-calculate current step data
        current_step_idx = st.session_state.get('algo_step', 0)
        history_item = None
        inertia = None
        labels = None
        centroids = None
        
        if 'algo' in st.session_state and len(st.session_state['algo'].history) > 0:
            current_step_idx = min(current_step_idx, len(st.session_state['algo'].history) - 1)
            history_item = st.session_state['algo'].history[current_step_idx]
            labels = history_item['labels']
            centroids = history_item['centroids']
            inertia = history_item.get('inertia')

        # --- Sidebar: Details (Updated dynamically) ---
        # We use an empty container placeholder created outside or just overwrite using st.sidebar methods
        # Streamlit allows writing to sidebar from anywhere
        st.sidebar.markdown("---")
        st.sidebar.subheader("Details")
        st.sidebar.write(f"Punkte: {X.shape[0]}")
        st.sidebar.write(f"Algorithmus: {algo_name}")
        # Note: 'k' variable is available from global scope (closure)
        if algo_name == "K-Means":
            st.sidebar.write(f"K: {k}")
        
        if inertia is not None:
            st.sidebar.metric("Varianz (Inertia)", f"{inertia:.2f}")
            
            if history_item and 'cluster_inertia' in history_item:
                st.sidebar.write("Varianz pro Cluster:")
                for c_id, c_val in history_item['cluster_inertia'].items():
                    percentage = (c_val / inertia * 100) if inertia > 0 else 0
                    st.sidebar.write(f"- Cluster {c_id}: {c_val:.2f} ({percentage:.1f}%)")


        # --- Main Area ---
        # Title logic
        if history_item:
            title = f"Ergebnis: {algo_name} (Schritt {current_step_idx})"
            if 'action' in history_item and history_item['action']:
                title += f" - {history_item['action']}"
            if st.session_state.get('algo_converged', False):
                title += " - Konvergiert!"
        else:
             labels = np.zeros(X.shape[0], dtype=int) # Default color
             title = "Rohdaten"

        # Create Plotly figure
        df = pd.DataFrame(X, columns=['x', 'y'])
        
        # Add counts to labels
        if labels is not None:
             unique_labels, counts = np.unique(labels, return_counts=True)
             label_map = {lbl: f"Cluster {lbl} (n={count})" for lbl, count in zip(unique_labels, counts)}
             df['label_desc'] = [label_map[l] for l in labels]
             df = df.sort_values('label_desc')
             color_col = 'label_desc'
        else:
             color_col = None

        fig = px.scatter(df, x='x', y='y', color=color_col, title=title, 
                         color_discrete_sequence=px.colors.qualitative.G10,
                         render_mode='svg')
        
        # Add centroids if available
        if centroids is not None:
            fig.add_trace(go.Scatter(
                x=centroids[:, 0], y=centroids[:, 1],
                mode='markers',
                marker=dict(symbol='x', size=14, color='black', line=dict(width=3, color='white')),
                name='Zentren'
            ))
        

        # --- Controls (Moved to Top) ---
        if 'algo' in st.session_state:
            algo = st.session_state['algo']
            
            # --- Timeline Slider ---
            real_max_step = max(0, len(algo.get_history()) - 1)
            slider_max = max(1, real_max_step)
            slider_disabled = (real_max_step == 0)

            # Sync slider with logic state before rendering
            st.session_state['slider_step'] = st.session_state['algo_step']
            
            st.slider("Zeitachse (Schritt)", 0, slider_max, key="slider_step", on_change=update_from_slider, disabled=slider_disabled)

            # Using columns for centering or spacing controls
            c1, c2, c3, c4 = st.columns(4)
            
            c1.button("Prev", use_container_width=True, on_click=prev_step)
            c2.button("Next", use_container_width=True, on_click=next_step)
            # Autoplay using fragment-safe rerun
            if st.session_state.get('autoplay', False):
                 c3.button("Stop", use_container_width=True, on_click=toggle_autoplay)
            else:
                 c3.button("Autoplay", use_container_width=True, on_click=toggle_autoplay)
                 
            c4.button("End", use_container_width=True, on_click=end_step)

            st.write(f"Schritt: {st.session_state['algo_step']}")
            if st.session_state['algo_converged']:
                st.success("Konvergiert!")

        # Render Plot (Full Width)
        fig.update_layout(height=600) # Ensure it's tall enough
        st.plotly_chart(fig, use_container_width=True)
        
        # Autoplay Logic (Inside fragment)
        if 'algo' in st.session_state and st.session_state.get('autoplay', False):
            algo = st.session_state['algo']
            if st.session_state['algo_step'] < len(algo.get_history()) - 1:
                time.sleep(0.3)
                st.session_state['algo_step'] += 1
                st.session_state['algo_converged'] = (st.session_state['algo_step'] == len(algo.get_history()) - 1)
                st.rerun()
            else:
                st.session_state['autoplay'] = False
                st.rerun()


    if st.session_state['data_generated']:
        render_visualization()
    else:
        st.info("Generiere Daten über die Sidebar, um zu beginnen.")
