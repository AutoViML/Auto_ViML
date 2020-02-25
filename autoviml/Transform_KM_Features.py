import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import sklearn
import scipy

class KMeansFeaturizer:
    """Transforms numeric data into k-means cluster memberships.
    From the Feature Engineering Book by Alice Zheng
    https://github.com/alicezheng/feature-engineering-book/blob/master/kmeans_featurizer.py
    This transformer runs k-means on the input data and converts each data point
    into the id of the closest cluster. If a target variable is present, it is
    scaled and included as input to k-means in order to derive clusters that
    obey the classification boundary as well as group similar points together.
    Parameters
    ----------
    k: integer, optional, default 100
        The number of clusters to group data into.
    target_scale: float, [0, infty], optional, default 5.0
        The scaling factor for the target variable. Set this to zero to ignore
        the target. For classification problems, larger `target_scale` values
        will produce clusters that better respect the class boundary.
    random_state : integer or numpy.RandomState, optional
        This is passed to k-means as the generator used to initialize the
        kmeans centers. If an integer is given, it fixes the seed. Defaults to
        the global numpy random number generator.
    Attributes
    ----------
    cluster_centers_ : array, [k, n_features]
        Coordinates of cluster centers. n_features does count the target column.
    """

    def __init__(self, k=100, target_scale=5.0, random_state=None):
        self.k = k
        self.target_scale = target_scale
        self.random_state = random_state

    def fit(self, X, y=None):
        """Runs k-means on the input data and find centroids.
        If no target is given (`y` is None) then run vanilla k-means on input
        `X`.
        If target `y` is given, then include the target (weighted by
        `target_scale`) as an extra dimension for k-means clustering. In this
        case, run k-means twice, first with the target, then an extra iteration
        without.
        After fitting, the attribute `cluster_centers_` are set to the k-means
        centroids in the input space represented by `X`.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_data_points, n_features)
        y : vector of length n_data_points, optional, default None
            If provided, will be weighted with `target_scale` and included in
            k-means clustering as hint.
        """
        n_features = X.shape[1]
        if y is None:
            # No target variable, just do plain k-means
            km_model = KMeans(n_clusters=self.k,
                                  n_init=20,
                                  random_state=self.random_state)
            km_model.fit(X)

            self.km_model_ = km_model
            self.cluster_centers_ = km_model.cluster_centers_
            return self

        # There is target information. Apply appropriate scaling and include
        # into input data to k-means
        data_with_target = np.hstack((X, y[:,np.newaxis]*self.target_scale))

        # Build a pre-training k-means model on data and target
        km_model_pretrain = KMeans(n_clusters=self.k,
                                   n_init=20,
                                   random_state=self.random_state)
        km_model_pretrain.fit(data_with_target)

        # Run k-means a second time to get the clusters in the original space
        # without target info. Initialize using centroids found in pre-training.
        # Go through a single iteration of cluster assignment and centroid
        # recomputation.
        km_model = KMeans(n_clusters=self.k,
                          init=km_model_pretrain.cluster_centers_[:,:n_features],
                          n_init=1,
                          max_iter=1)
        km_model.fit(X)

        self.km_model = km_model
        self.cluster_centers_ = km_model.cluster_centers_
        return self

    def transform(self, X, y=None):
        """Output the closest cluster id for each input data point.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_data_points, n_features)
        y : vector of length n_data_points, optional, default None
            Target vector is ignored even if provided.
        Returns
        -------
        cluster_ids : array, shape[n_data_points,1]
        """
        clusters = self.km_model.predict(X)
        return clusters[:,np.newaxis]

    def fit_transform(self, X, y=None):
        """Runs fit followed by transform.
        """
        self.fit(X, y)
        return self.transform(X, y)


from collections import defaultdict
import operator
import pdb
import copy
from sklearn.model_selection import train_test_split
def Transform_KM_Features(training_data, training_labels, test_data, km_max=0):
    seed = 99
    preds = list(training_data)
    target = training_labels.name
    train_index =  training_data.index
    test_index =  test_data.index
    if km_max == 0:
        km_max = int(np.log10(training_data.shape[0])+0.49)
    if km_max <= 2:
        k_max = 2
    else:
        k_max = copy.deepcopy(km_max)
    kmf =  KMeansFeaturizer(k=k_max, target_scale=0, random_state=seed)
    kmf_hint = kmf.fit(training_data, training_labels)
    ### Just return it with the cluster column => no need to return the data frame ###
    training_cluster_features = kmf_hint.transform(training_data)
    test_cluster_features = kmf_hint.transform(test_data)
    npx = np.c_[training_data, training_labels.values]
    training_with_cluster = np.c_[npx,training_cluster_features]
    test_with_cluster = np.c_[test_data, test_cluster_features]
    ### We are going to just return the cluster values ######
    train_with_cluster_df = training_with_cluster[:,-1]
    test_with_cluster_df = test_with_cluster[:,-1]
    #train_with_cluster_df = pd.DataFrame(training_with_cluster,index=train_index,
    #                                  columns=preds+[target,'cluster'])
    #test_with_cluster_df = pd.DataFrame(test_with_cluster,index=test_index,
    #                                  columns=preds+['cluster'])
    return train_with_cluster_df, test_with_cluster_df
