from abc import ABC, abstractmethod
import numpy as np

class ClusteringAlgorithm(ABC):
    """
    Abstract base class for clustering algorithms.
    User should implement the `run` or `step` methods.
    """
    
    def __init__(self, params: dict):
        self.params = params
        
    @abstractmethod
    def fit(self, X: np.ndarray):
        """
        Fits the algorithm to the data.
        Should return self.
        """
        pass
    
    @abstractmethod
    def get_labels(self) -> np.ndarray:
        """
        Returns the cluster labels for each point.
        """
        pass
    
    @abstractmethod
    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroids (if applicable).
        Return None if not applicable.
        """
        pass

class KMeansManual(ClusteringAlgorithm):
    """
    Manual K-Means implementation with step-by-step execution support.
    """
    def __init__(self, k=3, max_iter=100):
        super().__init__({"k": k, "max_iter": max_iter})
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        self.history = [] # List of dicts: {'centroids': ..., 'labels': ...}
        self.initialized = False
        
    def initialize(self, X: np.ndarray):
        """Initializes centroids randomly."""
        n_samples = X.shape[0]
        # Random initialization
        indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[indices].copy()
        self.labels = np.zeros(n_samples, dtype=int)
        self.history = []
        self._save_state()
        self.initialized = True
        return self

    def _save_state(self):
        self.history.append({
            'centroids': self.centroids.copy(),
            'labels': self.labels.copy()
        })

    def step(self, X: np.ndarray) -> bool:
        """
        Performs one step of K-Means.
        Returns True if converged, False otherwise.
        """
        if not self.initialized:
            self.initialize(X)

        # 1. Assign labels based on closest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        new_labels = np.argmin(distances, axis=1)
        
        # Check for convergence (labels didn't change)
        if np.array_equal(self.labels, new_labels) and len(self.history) > 1:
             return True

        self.labels = new_labels

        # 2. Update centroids
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.k):
            points = X[self.labels == i]
            if len(points) > 0:
                new_centroids[i] = points.mean(axis=0)
            else:
                # Handle empty cluster: keep old centroid or re-initialize
                new_centroids[i] = self.centroids[i] 
        
        self.centroids = new_centroids
        self._save_state()
        
        return False

    def fit(self, X: np.ndarray):
        self.initialize(X)
        for _ in range(self.max_iter):
            if self.step(X):
                break
        return self
        
    def get_labels(self) -> np.ndarray:
        return self.labels
        
    def get_centroids(self) -> np.ndarray:
        return self.centroids
    
    def get_history(self):
        return self.history
