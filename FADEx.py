import pandas as pd

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.interpolate import RBFInterpolator

from concurrent.futures import ProcessPoolExecutor

import cupy as cp
from cuml.neighbors import NearestNeighbors as CudaNearestNeighbors
from cuml.decomposition import PCA as CudaPCA
from cupyx.scipy.interpolate import RBFInterpolator as CudaRBFInterpolator

from tqdm import tqdm

from FADEx_plotting import _image_explanation_plot, _explanation_plot, _interactive_plot, _importance_plot

cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

class FADEx:
    '''
    Local explainability method for dimensionality reduction (DR) algorithms using Jacobian-based analysis.
    This class provides tools to explain the importance of features in the original high-dimensional space
    by analyzing the local behavior of the DR mapping.

    Parameters
    ----------
    high_dim_data : np.ndarray, shape (n_samples, n_features)
        The dataset in the original high-dimensional space.

    low_dim_data : np.ndarray, shape (n_samples, d)
        The dataset in the low-dimensional space (embedding produced by the DR algorithm).

    n_neighbors : int, optional
        Number of neighbors to consider for local explanations. If None, all points are used.

    feature_names : list of str, optional
        Names of the original features. If None, generic names are used for plotting.

    classes_names : list of str, optional
        Names of the classes for each instance.

    RBFkernel : str, default='cubic'
        The kernel used by the RBFInterpolator.

    pre_dr : int, optional
        If not None, applies preliminary dimensionality reduction (PCA) to the high-dimensional data,
        reducing it to `pre_dr` dimensions.

    RBFepsilon : float, default=0.001
        The epsilon parameter for the RBF kernel.

    RBFdegree : float, default=1
        The degree parameter for the RBF kernel.

    RBFsmoothing : float, default=0
        The smoothing parameter for the RBF kernel.

    useGPU : bool, default=False
        If True, uses GPU acceleration for computations.

    distanceAdjustMethod : str, default='sqrt_inv'
        Method to adjust the minimum distance for numerical stability. Options include 'sqrt_inv', 
        'inv', 'log_inv', 'exp_neg', and 'power_neg'.

    distAlpha : float, default=0.01
        The alpha parameter for distance adjustment methods.

    image : bool, default=False
        If True, treats the data as images for visualization purposes.

    dist_sample : int, optional
        Number of samples to use for distance computation. If None, all data points are used.
    '''
        
    def __init__(self, high_dim_data: np.ndarray, low_dim_data: np.ndarray, 
                n_neighbors: int = None, feature_names: list = None, 
                classes_names: list = None, RBFkernel: str = 'cubic',
                pre_dr : int = None, RBFepsilon : float = 0.001, 
                RBFdegree : float = 1, RBFsmoothing : float = 0, 
                useGPU : bool = False, distanceAdjustMethod: str = 'sqrt_inv',
                distAlpha : float = 0.01, image : bool = False,
                dist_sample : int = None):

        self.n_neighbors = n_neighbors
        self.classes_names=classes_names
        self.RBFkernel = RBFkernel
        self.all_phis = None
        self.h = None
        self.pre_dr = pre_dr
        self.RBFdegree = RBFdegree
        self.RBFsmoothing = RBFsmoothing
        self.RBFepsilon = RBFepsilon
        self.useGPU = useGPU
        self.method = distanceAdjustMethod
        self.alpha = distAlpha
        self.image = image
        self.dist_sample = dist_sample

        if(self.useGPU):
            self.high_dim_data = cp.array(high_dim_data)
            self.low_dim_data = cp.array(low_dim_data)

            self.xp = cp
            self.nbrs = CudaNearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')

            if(self.pre_dr):
                self.pca = CudaPCA(n_components=self.pre_dr)

        else:
            self.high_dim_data = np.array(high_dim_data)
            self.low_dim_data = np.array(low_dim_data)

            self.xp = np
            self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree', n_jobs=2)

            if(self.pre_dr is not None):
                self.pca = PCA(n_components=self.pre_dr)

        self.n_samples = len(self.high_dim_data)
        self.n_features = len(self.high_dim_data[0])

        # Feature Names
        if(feature_names is None):
            num_features = self.high_dim_data.shape[1]
            self.feature_names = [f"feature_{i}" for i in range(num_features)]
        else:
            self.feature_names = feature_names
    
    def _compute_importance(self, jacobian, x):
                
        U, S, VT = self.xp.linalg.svd(jacobian, full_matrices=True)
        V = VT.T

        phi = self.xp.zeros_like(x)  

        n_singular_vectors = V.shape[1]
        if n_singular_vectors < 2:
            raise ValueError("Jacobian matrix does not have enough singular vectors.")
        
        
        for j in range(len(x)):
            phi[j] = self.xp.abs(V[j, 0] * x[j]) + (S[1]/S[0])*self.xp.abs(V[j, 1] * x[j])

        return phi
    
    def _nearest_neighbors(self, data, point, return_indices=False):

        self.nbrs.fit(data)

        distances, indices = self.nbrs.kneighbors(point.reshape(1, -1))

        if return_indices:
            return indices[0]
        else:
            return data[indices[0]]
    
    def _compute_distances_vec(self, data):
        
        num_points = len(data)
        if num_points < 2:
            raise ValueError('Not enough neighbors to compute distances.')

        # Pre DR
        if(self.pre_dr is not None):
            data = self.pca.fit_transform(data)

        # Min Distance Computing

        if(self.dist_sample is not None):
            indices = self.xp.random.choice(data.shape[0], size=self.dist_sample, replace=False)
            sampled_data = data[indices]
            distances = self.xp.linalg.norm(sampled_data[:, None, :] - sampled_data[None, :, :], axis=-1)
            min_distance = self.xp.min(distances[distances > 0])
        else:
            distances = self.xp.linalg.norm(data[:, None, :] - data[None, :, :], axis=-1)
            min_distance = self.xp.min(distances[distances > 0])

    
        # Distance Adjustment
        D = data.shape[1]

        return (1 / self.xp.sqrt(D)) * min_distance

    def _compute_jac_column(self, rbf, x, jac, i, j):
        x_plus_h = x.copy()
        x_plus_h[j] = x_plus_h[j] + self.h
        x_minus_h = x.copy()
        x_minus_h[j] = x_minus_h[j] - self.h

        f_plus_h = rbf(x_plus_h.reshape(1, -1))[0]
        f_minus_h = rbf(x_minus_h.reshape(1, -1))[0]

        # if(h == 0):
        #     h = 0.0001

        derivative = (f_plus_h - f_minus_h) / (2*self.h)

        jac[i, j] = derivative
    
    def _cuda_compute_jac_column(self, rbf, x, jac, i, j):

        x_plus_h = x.copy()
        x_plus_h[j] += self.h
        x_minus_h = x.copy()
        x_minus_h[j] -= self.h

        f_plus_h = rbf(cp.asarray(x_plus_h).reshape(1, -1))[0]
        f_minus_h = rbf(cp.asarray(x_minus_h).reshape(1, -1))[0]

        # if h == 0:
        #     h = 0.0001

        derivative = (f_plus_h - f_minus_h) / (2 * self.h)
        jac[i, j] = derivative

    def _compute_jac_vec(self, high_dim_point, low_dim_point, high_dim_nei, low_dim_nei):

        if(self.useGPU):

            jac = cp.zeros((len(low_dim_point), len(high_dim_point)), dtype=cp.float32)  
            x = high_dim_point

            batch_size = 201
            num_batches = (len(low_dim_point) + batch_size - 1) // batch_size

            # For each line 
            for i in range(len(low_dim_point)):

                # Calculating columns in batches
                for batch in range(num_batches):
                    start = batch * batch_size
                    end = min((batch + 1) * batch_size, len(high_dim_nei))

                    # Creating a batch interpolator (OLHAR ISSO AQUI DEPOIS)
                    rbf = CudaRBFInterpolator(
                        cp.asarray(high_dim_nei[start:end], dtype=cp.float32),
                        cp.asarray(low_dim_nei[start:end, i].reshape(-1, 1), dtype=cp.float32),
                        kernel=self.RBFkernel, 
                        epsilon=self.RBFepsilon, 
                        degree=self.RBFdegree, 
                        smoothing=self.RBFsmoothing
                    )

                    x_plus_list = []
                    x_minus_list = []

                    # Making disturbances
                    for j in range(len(high_dim_point)):
                        x_plus = x.copy()
                        x_minus = x.copy()
                        
                        x_plus[j]  += self.h
                        x_minus[j] -= self.h
                        
                        x_plus_list.append(x_plus)
                        x_minus_list.append(x_minus)

                    x_plus_batch  = cp.stack(x_plus_list, axis=0) 
                    x_minus_batch = cp.stack(x_minus_list, axis=0) 

                    # Evaluating the function
                    f_plus_batch  = rbf(x_plus_batch) 
                    f_minus_batch = rbf(x_minus_batch)  

                    # Computing the derivative
                    for j in range(len(high_dim_point)):
                        derivative_j = (f_plus_batch[j] - f_minus_batch[j]) / (2 * self.h)
                        jac[i, j] = derivative_j

            return jac

        else:

            jac = np.zeros((len(low_dim_point), len(high_dim_point)))
            x = high_dim_point

            # For each line
            for i in range(len(low_dim_point)):
                rbf = RBFInterpolator(
                    high_dim_nei, 
                    low_dim_nei[:, i].reshape(-1, 1),
                    kernel=self.RBFkernel,
                    epsilon=self.RBFepsilon,
                    degree=self.RBFdegree,
                    smoothing=self.RBFsmoothing
                )

                x_plus_array = []
                x_minus_array = []

                # Computing columns in batches
                for j in range(len(high_dim_point)):
                    x_plus = x.copy()
                    x_plus[j] += self.h

                    x_minus = x.copy()
                    x_minus[j] -= self.h

                    x_plus_array.append(x_plus)
                    x_minus_array.append(x_minus)

                x_plus_array = np.array(x_plus_array)
                x_minus_array = np.array(x_minus_array)

                f_plus = rbf(x_plus_array)  
                f_minus = rbf(x_minus_array)

                for j in range(len(high_dim_point)):
                    derivative_j = (f_plus[j] - f_minus[j]) / (2 * self.h)
                    jac[i, j] = derivative_j

            return jac
        

    def _compute_jac(self, high_dim_point, low_dim_point, high_dim_nei, low_dim_nei):

        if self.useGPU:

            jac = cp.zeros((len(low_dim_point), len(high_dim_point)), dtype=cp.float32)
            x = high_dim_point 

            for i in range(len(low_dim_point)):
    
                rbf = CudaRBFInterpolator(
                    cp.asarray(high_dim_nei, dtype=cp.float32),
                    cp.asarray(low_dim_nei[:, i].reshape(-1, 1), dtype=cp.float32),
                    kernel=self.RBFkernel, 
                    epsilon=self.RBFepsilon, 
                    degree=self.RBFdegree, 
                    smoothing=self.RBFsmoothing
                )
      
                for j in range(len(high_dim_point)):
                    self._cuda_compute_jac_column(rbf, x, jac, i, j)
            return jac
        else:

            jac = np.zeros((len(low_dim_point), len(high_dim_point)))
            x = high_dim_point
 
            for i in range(len(low_dim_point)):
          
                rbf = RBFInterpolator(
                    high_dim_nei, 
                    low_dim_nei[:, i].reshape(-1, 1), 
                    kernel=self.RBFkernel, 
                    epsilon=self.RBFepsilon, 
                    degree=self.RBFdegree,
                    smoothing=self.RBFsmoothing
                )

                for j in range(len(high_dim_point)):
                    self._compute_jac_column(rbf, x, jac, i, j)
            return jac


    def fit(self, explain_index : int, show : bool = True, p : float = 0.5, width : int = 10, height : int = 8):
        '''
        Computes the feature importance for a specific instance in the dataset.

        Parameters
        ----------
        explain_index : int
            The index of the instance to explain.

        show : bool, default=True
            If True, displays the explanation plot.

        p : float, default=0.5
            The importance threshold used during the image importance plot

        width : int, default=8
            The width of the plot.

        height : int, default=10
            The height of the plot.

        Returns
        -------
        phi : np.ndarray or cp.ndarray, shape (n_features,)
            The importance values for each feature.

        importance_vector : list of int
            The indices of features sorted by importance.

        spectral_norm : float
            The spectral norm of the Jacobian matrix.
        '''
        high_dim_point = self.high_dim_data[explain_index]
        self.high_dim_point = self.high_dim_data[explain_index]
        low_dim_point = self.low_dim_data[explain_index]


        # Nearest Neighbors
        if(self.n_neighbors is not None):
            indices = self._nearest_neighbors(
                data=self.high_dim_data, 
                point=high_dim_point, 
                return_indices=True
            )

            low_dim_nei = self.low_dim_data[indices]
            high_dim_nei = self.high_dim_data[indices]
        else:
            low_dim_nei = self.low_dim_data
            high_dim_nei = self.high_dim_data

        # Distance Computing
        if(self.h is None):
            self.h = self._compute_distances_vec(self.high_dim_data)
            print(f'h = {self.h}')

        # Jacobian Computing
        self.jac = self._compute_jac_vec(
            high_dim_point,
            low_dim_point,
            high_dim_nei,
            low_dim_nei,
        )

        # Importance Computing
        phi = self._compute_importance(self.jac, high_dim_point)

        # Importance Vector
        indexed_phi = list(enumerate(phi))
        sorted_indices = sorted(indexed_phi, key=lambda x: x[1], reverse=True)

        # Spectral Norm
        spectral_norm = self.xp.linalg.norm(self.jac, ord=2)

        if(np.isnan(spectral_norm)):
            print('Spectral Norm = NaN')
        

        if(show):

            if(self.image):
                _image_explanation_plot(self, phi, spectral_norm, high_dim_point, p)
            else:
                _explanation_plot(self, phi, spectral_norm, explain_index, width, height)
                


        return phi, self.jac, spectral_norm
    
    
    def print_error(self, jacobian):
        '''
        outdated function
        '''

        self.jac = self.jac.get()

        jac_true = jacobian(self.high_dim_point)

        jac_error = np.linalg.norm(self.jac - jac_true)

        jac_true_norm = np.linalg.norm(jac_true)

        if jac_true_norm == 0:
            raise ValueError("The norm of the true Jacobian is zero. Cannot compute relative error.")
        else:
            relative_error = (jac_error / jac_true_norm) * 100

            print("Computed Jacobian error (absolute):", jac_error)
            print("Computed Jacobian relative error (%): {:.2f}%".format(relative_error))

    # Auxiliary function for parallelism
    def _compute_phi(self, i):
        phi, _, spec_norm = self.fit(i, show=False)
        return phi, spec_norm
    
    # Applies fit function in the entire dataset
    def _fit_all(self):

        with ProcessPoolExecutor(max_workers=2) as executor:
            results = list(tqdm(
                executor.map(self._compute_phi, range(self.n_samples)), 
                total=self.n_samples, 
                desc="Processing samples", 
                unit="sample"
            ))

        all_phis, all_norms = zip(*results)
        self.all_phis = np.array(all_phis)
        self.all_norms = np.array(all_norms)   
 

    def _fit_all_sequential(self):
        results = [self._compute_phi(i) for i in tqdm(range(self.n_samples), desc="Processing samples", unit="sample")]

        all_phis, all_norms = zip(*results)
        self.all_phis = np.array([arr.get() if isinstance(arr, cp.ndarray) else arr for arr in all_phis])
        self.all_norms = np.array([arr.get() if isinstance(arr, cp.ndarray) else arr for arr in all_norms])



    def interactive_plot(self, width=None, height=None, fit_all_method='sequential'):
        '''
        Generates an interactive plot that shows the feature importance for every instance.

        Parameters
        ----------
        width : int, optional
            The width of the plot.

        height : int, optional
            The height of the plot.

        fit_all_method : str, default='sequential'
            The method to compute feature importance for all instances. Options are 'sequential' or 'parallel'.
        '''

        if(self.all_phis is None):

            if(fit_all_method == 'sequential'):
                self._fit_all_sequential()
            elif(fit_all_method == 'parallel'):
                self._fit_all()

        _interactive_plot(self, width, height)

    def importance_plot(self, width=10, height=8, fit_all_method='sequential', n_top=10):
        '''
        Generates a plot of feature importance for all instances.

        Parameters
        ----------
        fit_all_method : str, default='sequential'
            The method to compute feature importance for all instances. Options are 'sequential' or 'parallel'.
        '''

        if(self.all_phis is None):

            if(fit_all_method == 'sequential'):
                self._fit_all_sequential()
            elif(fit_all_method == 'parallel'):
                self._fit_all()


        _importance_plot(self, width, height, n_top)

    def get_mean_spectral_norm(self):
        if(self.all_norms is not None):
            return np.mean(self.all_norms)