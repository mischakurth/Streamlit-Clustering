import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_gen import generate_data
from algorithms import KMeansManual

st.set_page_config(page_title="Clustering Visualizer", layout="wide")

st.title("Clustering Algorithmen Visualisierung")

# --- Sidebar: Data Generation ---
st.sidebar.header("1. Daten Generierung")
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
    # Reset algorithm state
    if 'algo' in st.session_state:
        del st.session_state['algo']

# Initialize session state if not present
if 'data_generated' not in st.session_state:
    st.session_state['data_generated'] = False

# --- Sidebar: Algorithm Selection ---
st.sidebar.header("2. Algorithmus")
algo_name = st.sidebar.selectbox("Algorithmus", ["K-Means"])

if algo_name == "K-Means":
    k = st.sidebar.slider("K (Anzahl Cluster)", 2, 10, 3)
    max_iter = st.sidebar.slider("Max Iterationen", 10, 200, 100)
    
    if st.sidebar.button("Algorithmus starten"):
        if st.session_state['data_generated']:
            algo = KMeansManual(k=k, max_iter=max_iter)
            algo.fit(st.session_state['X'])
            st.session_state['algo'] = algo
        else:
            st.error("Bitte zuerst Daten generieren!")

# --- Main Area: Visualization ---
col1, col2 = st.columns([3, 1])

with col1:
    if st.session_state['data_generated']:
        X = st.session_state['X']
        
        # Determine colors
        if 'algo' in st.session_state:
            labels = st.session_state['algo'].get_labels()
            title = f"Ergebnis: {algo_name}"
        else:
            labels = np.zeros(X.shape[0]) # Default color
            title = "Rohdaten"
            
        # Create Plotly figure
        df = pd.DataFrame(X, columns=['x', 'y'])
        df['label'] = labels.astype(str)
        
        fig = px.scatter(df, x='x', y='y', color='label', title=title, 
                         color_discrete_sequence=px.colors.qualitative.G10)
        
        # Add centroids if available
        if 'algo' in st.session_state:
            centroids = st.session_state['algo'].get_centroids()
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
