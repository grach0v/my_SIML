#%%
import numpy as np
import numba
from numba import types, typed, typeof, float64, int64, boolean
from numba.experimental import jitclass
import matplotlib.pyplot as plt

#%%

spec = [
    ('data_X', float64[:, :]),
    ('cluster_labels', int64[:]),
    ('update_parts', int64),
    ('iter_limit', int64),
    ('debug', boolean),
    ('number_clusters', int64),
    ('means', float64[:, :]),
    ('covariances', float64[:, :, :]),
    ('determinants', float64[:]),
    ('inverse_mats', float64[:, :, :]),
    ('changed_clusters', boolean[:]), #?
    ('labels_history', typeof(typed.List.empty_list(int64[:]))), #FIXME
    ('likelihoods_history', types.ListType(types.float64))
]


@jitclass(spec)
class SIML:
    def __init__(self, data_X, cluster_labels, update_parts=10, iter_limit=1e3, debug=False, labels_history=typed.List.empty_list(int64[:])): #REMOVE labels_history
        self.data_X = data_X
        self.cluster_labels = cluster_labels.copy()
        self.update_parts = update_parts
        self.iter_limit = iter_limit
        self.debug = debug

        self.number_clusters = len(np.unique(cluster_labels))

        self.means = np.zeros(shape=(self.number_clusters, self.data_X.shape[1]))
        self.covariances = np.zeros(shape=(self.number_clusters, self.data_X.shape[1], self.data_X.shape[1]))
        self.determinants = np.zeros(self.number_clusters)
        self.inverse_mats = np.zeros(shape=(self.number_clusters, self.data_X.shape[1], self.data_X.shape[1]))
        self.changed_clusters = np.ones(self.number_clusters, dtype=boolean)


        self.labels_history = labels_history
        self.likelihoods_history = typed.List.empty_list(types.float64)

        if self.debug:
            self.labels_history.append(self.cluster_labels.copy())
            self.likelihoods_history.append(self.count_likelihood())

    def count_likelihood(self): #FIXME
        return 0

    def update_vars(self):
        for cluster in np.arange(self.number_clusters)[self.changed_clusters]:
            for i in range(self.data_X.shape[1]): #CHANGE if remove numba
                self.means[cluster][i] = np.mean(self.data_X[self.cluster_labels == cluster][:, i])

            self.covariances[cluster] = np.cov(self.data_X[self.cluster_labels == cluster].T)
            self.determinants[cluster] = np.linalg.det(self.covariances[cluster])
            self.inverse_mats[cluster] = np.linalg.inv(self.covariances[cluster])

    def make_iter(self, lhs, rhs):
        self.update_vars()

        deltas = np.zeros(self.number_clusters)
        size = self.data_X.shape[0]
        dimension = self.data_X.shape[1]

        changed = False
        self.changed_clusters = np.zeros(self.number_clusters, dtype=boolean)

        for i in range(lhs, rhs):
            x = self.data_X[i]
            cur_label = self.cluster_labels[i]

            if np.sum(self.cluster_labels == cur_label) <= 1:
                continue

            for next_cluster in range(self.number_clusters):
                n = np.sum(self.cluster_labels == next_cluster)
                mean = self.means[next_cluster]
                det = self.determinants[next_cluster]
                inv = self.inverse_mats[next_cluster]

                if cur_label == next_cluster:
                    deltas[next_cluster] = -1/2 * np.log(det) + \
                        (n - 1) / 2 * np.log(1 - 1 / (n - 1) * (x - mean).T @ inv @ (x - mean)) + \
                        np.log(n / size) - \
                        (n - 1) * (dimension / 2 + 1) * np.log((n + 1) / n)
                        
                else:
                    deltas[next_cluster] = -1/2 * np.log(det) - \
                        (n + 1) / 2 * np.log(1 + 1 / (n + 1) * (x - mean).T @ inv @ (x - mean)) + \
                        np.log(n / size) + \
                        (n + 1) * (dimension / 2 + 1) * np.log((n - 1) / n)

            new_label = np.argmax(deltas)

            if new_label != cur_label:
                changed = True
                self.changed_clusters[cur_label] = True
                self.changed_clusters[cur_label] = True
                self.cluster_labels[i] = new_label

        if self.debug:
            self.labels_history.append(self.cluster_labels.copy())
            self.likelihoods_history.append(self.count_likelihood())

        return changed

    def cluster(self):
        part_size = (len(self.data_X) + self.update_parts - 1) // self.update_parts
        changed = True
        counter = 0

        while changed and counter < self.iter_limit:
            counter += 1
            lhs = 0
            rhs = part_size
            changed = False

            while lhs < len(self.data_X):
                changed = changed or self.make_iter(lhs, rhs)

                lhs += part_size
                rhs += part_size
                rhs = min(len(self.data_X), rhs)


    

#%%
if __name__ == '__main__':

    size = 100
    a = np.random.normal(0, 0.4, size)
    b = np.random.normal(0, 0.4, size)
    c = np.random.normal(0, 0.4, size)
    d = np.random.normal(0, 0.4, size)
    e = np.random.normal(0, 0.4, size)


    x = np.concatenate((np.asarray([a - 1, b - 1]).T,
                        np.asarray([b - 1, c + 1]).T, 
                        np.asarray([c + 1, d - 1]).T,
                        np.asarray([d + 1, e + 1]).T
                        ), axis=0)

    labels = np.concatenate([np.random.uniform(-0.3, 0.7, size), 
                            np.random.uniform(0.3, 1.7, size),
                            np.random.uniform(1.3, 2.7, size),
                            np.random.uniform(2.3, 3.3, size)\
                            ]).round().astype(int)
    
    siml = SIML(x, labels, debug=True)
    siml.cluster()

    plt.scatter(x[:, 0], x[:, 1], c=labels)
    plt.show()

    new_labels = siml.cluster_labels
    plt.scatter(x[:, 0], x[:, 1], c=new_labels)
    plt.show()



# %%
