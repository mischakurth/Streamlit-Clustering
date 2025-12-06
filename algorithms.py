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
        self.phase = None
        
    def initialize(self, X: np.ndarray, method='random'):
        """Initializes centroids."""
        n_samples = X.shape[0]
        self.history = []
        self.phase = 'assign'
        
        if method == 'random':
            indices = np.random.choice(n_samples, self.k, replace=False)
            self.centroids = X[indices].copy()
        elif method == 'k-means++':
            # 1. Choose first center randomly
            self.centroids = np.zeros((self.k, X.shape[1]))
            first_idx = np.random.randint(n_samples)
            self.centroids[0] = X[first_idx]
            
            for i in range(1, self.k):
                # 2. Compute distances to nearest existing center
                # Distances from each point to each existing centroid
                distances = np.linalg.norm(X[:, np.newaxis] - self.centroids[:i], axis=2)
                # Min distance for each point to any existing centroid
                min_distances = np.min(distances, axis=1)
                
                # 3. Choose next center with probability proportional to distance squared
                probs = min_distances**2 / np.sum(min_distances**2)
                next_idx = np.random.choice(n_samples, p=probs)
                self.centroids[i] = X[next_idx]
        else:
            raise ValueError(f"Unknown initialization method: {method}")

        self.labels = np.zeros(n_samples, dtype=int)
        self._save_state(X, action_description="Initialisierung")
        self.initialized = True
        return self

    def _save_state(self, X: np.ndarray, action_description: str = ""):
        # Calculate inertia (sum of squared distances to closest centroid)
        inertia = 0.0
        cluster_inertia = {}
        if self.centroids is not None and self.labels is not None:
             for i in range(self.k):
                 points = X[self.labels == i]
                 if len(points) > 0:
                     # Squared Euclidean distance
                     sq_dists = np.sum((points - self.centroids[i])**2, axis=1)
                     c_inertia = np.sum(sq_dists)
                     inertia += c_inertia
                     cluster_inertia[i] = c_inertia
                 else:
                     cluster_inertia[i] = 0.0

        self.history.append({
            'centroids': self.centroids.copy(),
            'labels': self.labels.copy(),
            'inertia': inertia,
            'cluster_inertia': cluster_inertia,
            'action': action_description
        })

    def step(self, X: np.ndarray) -> bool:
        """
        Performs one step of K-Means (either assignment or update).
        Returns True if converged, False otherwise.
        """
        if not self.initialized:
            self.initialize(X)

        if self.phase == 'assign':
            # 1. Assign labels based on closest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            new_labels = np.argmin(distances, axis=1)
            
            # Check for convergence (labels didn't change) after assignment
            # Only return True if we've done at least one full cycle (history > 1)
            if np.array_equal(self.labels, new_labels) and len(self.history) > 1:
                 return True

            self.labels = new_labels
            self.phase = 'update'
            self._save_state(X, action_description="Punkte zugewiesen")
            return False

        elif self.phase == 'update':
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
            self.phase = 'assign'
            self._save_state(X, action_description="Zentren angepasst")
            return False
            
        return False

    def fit(self, X: np.ndarray):
        if not self.initialized:
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
