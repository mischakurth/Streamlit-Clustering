import streamlit as st
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

# --- Sidebar: Data Generation ---
st.sidebar.header("1. Daten Generierung")
data_source = st.sidebar.radio("Datenquelle", ["Generieren", "Importieren"])

if data_source == "Generieren":
    data_type = st.sidebar.selectbox("Datensatz Typ", ["blobs", "moons", "circles"])
    n_samples = st.sidebar.slider("Anzahl Punkte", 100, 1000, 300)
    n_features = 2 # Fixed for 2D visualization
    centers = st.sidebar.slider("Anzahl Zentren (nur Blobs)", 2, 8, 4)
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

# --- Sidebar: Algorithm Selection ---
st.sidebar.header("2. Algorithmus")
algo_name = st.sidebar.selectbox("Algorithmus", ["K-Means"])

if algo_name == "K-Means":
    k = st.sidebar.slider("K (Anzahl Cluster)", 2, 10, 3)
    max_iter = st.sidebar.slider("Max Iterationen", 10, 1000, 100)
    
    col_start, col_step = st.sidebar.columns(2)
    
    if col_start.button("Initialisieren"):
        if st.session_state['data_generated']:
            algo = KMeansManual(k=k, max_iter=max_iter)
            algo.initialize(st.session_state['X'])
            st.session_state['algo'] = algo
            st.session_state['algo_step'] = 0
            st.session_state['algo_converged'] = False
        else:
            st.error("Bitte zuerst Daten generieren!")

    if 'algo' in st.session_state:
        algo = st.session_state['algo']
        
        # Navigation Controls
        c1, c2, c3, c4 = st.sidebar.columns(4)
        if c1.button("Prev"):
            if st.session_state['algo_step'] > 0:
                st.session_state['algo_step'] -= 1
                st.session_state['algo_converged'] = False
        
        if c2.button("Next"):
            if st.session_state['algo_step'] < len(algo.get_history()) - 1:
                st.session_state['algo_step'] += 1
                converged = algo.step(st.session_state['X'])
                st.session_state['algo_converged'] = converged


        if c3.button("Play"):
            if not st.session_state['algo_converged']:
                progress_bar = st.sidebar.progress(0)
                for i in range(max_iter):
                    converged = algo.step(st.session_state['X'])
                    st.session_state['algo_converged'] = converged
                    st.session_state['algo_step'] = len(algo.get_history()) - 1
                    progress_bar.progress((i + 1) / max_iter)
                    time.sleep(0.2)
                    st.rerun()
                    if converged:
                        break
                progress_bar.empty()

        if c4.button("End"):
             algo.fit(st.session_state['X'])
             st.session_state['algo_converged'] = True
             st.session_state['algo_step'] = len(algo.get_history()) - 1

        st.sidebar.write(f"Schritt: {st.session_state['algo_step']}")
        if st.session_state['algo_converged']:
            st.sidebar.success("Konvergiert!")

# --- Main Area: Visualization ---
col1, col2 = st.columns([3, 1])

with col1:
    if st.session_state['data_generated']:
        X = st.session_state['X']
        
        # Determine colors and centroids based on current step
        if 'algo' in st.session_state and len(st.session_state['algo'].history) > 0:
            step_idx = st.session_state['algo_step']
            # Ensure index is within bounds (safety check)
            step_idx = min(step_idx, len(st.session_state['algo'].history) - 1)
            
            history_item = st.session_state['algo'].history[step_idx]
            labels = history_item['labels']
            centroids = history_item['centroids']
            title = f"Ergebnis: {algo_name} (Schritt {step_idx})"
        else:
            labels = np.zeros(X.shape[0]) # Default color
            centroids = None
            title = "Rohdaten"
            
        # Create Plotly figure
        df = pd.DataFrame(X, columns=['x', 'y'])
        df['label'] = labels.astype(str)
        
        fig = px.scatter(df, x='x', y='y', color='label', title=title, 
                         color_discrete_sequence=px.colors.qualitative.G10)
        
        # Add centroids if available
        if centroids is not None:
            fig.add_trace(go.Scatter(
                x=centroids[:, 0], y=centroids[:, 1],
                mode='markers',
                marker=dict(symbol='x', size=12, color='black', line=dict(width=2, color='white')),
                name='Zentren'
            ))
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Generiere Daten über die Sidebar, um zu beginnen.")

with col2:
    st.subheader("Details")
    if st.session_state['data_generated']:
        st.write(f"Punkte: {st.session_state['X'].shape[0]}")
        if 'algo' in st.session_state:
            st.write(f"Algorithmus: {algo_name}")
            if algo_name == "K-Means":
                st.write(f"K: {k}")
