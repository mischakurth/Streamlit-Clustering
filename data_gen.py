import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles

def generate_data(n_samples=300, n_features=2, centers=4, cluster_std=1.0, type='blobs'):
    """
    Generates synthetic data for clustering.
    
    Args:
        n_samples: Total number of points.
        n_features: Number of features (dimensions).
        centers: Number of centers (for blobs).
        cluster_std: Standard deviation of clusters (for blobs).
        type: 'blobs', 'moons', or 'circles'.
        
    Returns:
        X: The generated samples.
        y: The integer labels for cluster membership of each sample (ground truth).
    """
    if type == 'blobs':
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=42)
    elif type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=cluster_std/10.0, random_state=42)
    elif type == 'circles':
        X, y = make_circles(n_samples=n_samples, factor=0.5, noise=cluster_std/10.0, random_state=42)
    else:
        raise ValueError(f"Unknown data type: {type}")
        
    return X, y
