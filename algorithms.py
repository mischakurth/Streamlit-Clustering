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
    Placeholder for the user's manual K-Means implementation.
    """
    def __init__(self, k=3, max_iter=100):
        super().__init__({"k": k, "max_iter": max_iter})
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        
    def fit(self, X: np.ndarray):
        # TODO: USER IMPLEMENTATION HERE
        # This is just a dummy implementation for visualization purposes
        n_samples = X.shape[0]
        self.labels = np.random.randint(0, self.k, size=n_samples)
        self.centroids = np.random.rand(self.k, X.shape[1])
        return self
        
    def get_labels(self) -> np.ndarray:
        return self.labels
        
    def get_centroids(self) -> np.ndarray:
        return self.centroids
