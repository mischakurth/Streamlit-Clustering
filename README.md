# Streamlit Clustering Visualization

An interactive visualization of clustering algorithms (K-Means) built with Streamlit. This project allows you to understand how clustering algorithms work by visualizing them step-by-step on various datasets.

## 🚀 Live Demo

[**Open the Visualization App**](https://mischakurth.github.io/Streamlit-Clustering/)

## About the Project

This application provides a playground for experimenting with K-Means clustering. You can generate synthetic datasets and watch how the algorithm iteratively finds cluster centers.

### Key Features
*   **Interactive Controls**: Adjustable parameters for dataset generation (Samples, Centers, Noise).
*   **Algorithm Visualization**: Watch the K-Means algorithm converge in real-time.
    *   **Step-by-Step**: Manually advance the algorithm one step at a time.
    *   **Autoplay**: Watch the full process with finding centroids and assigning clusters.
*   **Metrics**: Real-time display of cluster counts and total variance.
*   **Initialization Options**: Compare Random initialization vs. K-Means++.

## local Installation & Usage

If you want to run the app locally:

1.  Clone the repository:
    ```bash
    git clone https://github.com/mischakurth/Streamlit-Clustering.git
    cd Streamlit-Clustering
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Technologies

*   Python
*   Streamlit
*   Scikit-Learn
*   Plotly
*   NumPy
