import numpy as np
from scipy.spatial.distance import cdist
from base_experiment import BaseExperiment


class NearestNeighborsModel():
    def __init__(self, params, train_acc, train_emb):
        self.k_nn = params['k_nn']
        self.train_acc = train_acc
        self.train_emb = train_emb

    def compute_pdists(self, x, y, dis="Euclidean"):
        if dis == "Euclidean":
            distance = cdist(x, y)
        elif dis == "Cosine":
            x = x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)
            y = y/np.linalg.norm(y, ord=2, axis=1, keepdims=True)
            distance = 1 - np.dot(x, y.T)
        return distance


    def __call__(self, test_emb, k_nn=None):
        distances = self.compute_pdists(test_emb, self.train_emb, dis="Cosine")
        if k_nn is None:
            k_nn = self.k_nn

        # Find the nearest embeddings and dist away
        nbr_inds = np.argsort(distances, axis=1)[:, 0:k_nn]
        nbr_dists = np.sort(distances, axis=1)[:, 0:k_nn]

        # Calculate score
        scores = np.mean(self.train_acc[nbr_inds, :], axis=1)
        return scores, {'mean_dist': nbr_dists.mean(axis=0)}


class NearestNeighborsExperiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def train(self, params, train_data):
        model = NearestNeighborsModel(params, train_data[0], train_data[1])
        return model
