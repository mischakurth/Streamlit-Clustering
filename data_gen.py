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
    elif type == 'uniform':
        # Generate points uniformly in a box [-10, 10]
        X = np.random.uniform(low=-10, high=10, size=(n_samples, n_features))
        y = np.zeros(n_samples, dtype=int)
    elif type == 'aniso':
        # Anisotropicly distributed data
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
    else:
        raise ValueError(f"Unknown data type: {type}")
        
    return X, y
