# CPU
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.interpolate import RBFInterpolator

try:
    import cupy as cp
    from cuml.neighbors import NearestNeighbors as CudaNearestNeighbors
    from cuml.decomposition import PCA as CudaPCA
    from cupyx.scipy.interpolate import RBFInterpolator as CudaRBFInterpolator
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    GPU_availability = True
except ImportError:
    GPU_availability = False
    import warnings
    warnings.warn("Unable to import one of the GPU-based libraries. GPU computing unavailable.", ImportWarning)


import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def _explanation_plot(self, phi, spectral_norm, explain_index, width, height, n_top):
    
    if(self.use_GPU):
        phi = phi.get()
        spectral_norm = spectral_norm.get()

    indices = np.argsort(np.abs(phi))[::-1]
    
    top_indices = indices[:n_top]
    phi_top = phi[top_indices]
    feature_names_top = [self.feature_names[i] for i in top_indices]

    plt.figure(figsize=(width, height))
    y_pos = np.arange(len(phi_top))
    plt.barh(y_pos, phi_top, color=['red' if val < 0 else 'green' for val in phi_top])
    plt.yticks(y_pos, feature_names_top, fontsize=12)
    plt.xticks([])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance Values", fontsize=14)
    plt.title(f"Feature Importance for instance {explain_index} (spectral norm = {spectral_norm:.3f})")
    plt.axvline(x=0, color='black', linewidth=1)
    plt.tight_layout()
    plt.show()


def _interactive_plot(self, width, height):

    width = width * 100
    height = height * 100

    if(self.use_GPU):
        low_dim_data = self.low_dim_data.get()
    else:
        low_dim_data = self.low_dim_data

    formatted_text = []
    for i, phi_row in enumerate(self.all_phis):

        sorted_data = sorted(
            zip(self.feature_names, phi_row),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        features_str = "<br>".join(
            f"{feature_name}: {phi_value:.3f}"
            for feature_name, phi_value in sorted_data
        )


        classe_str = ""
        if self.class_names is not None:
            classe_str = f"Class: {self.class_names[i]}<br>"

        norma_str = f"Spectral Norm: {self.all_norms[i]:.3f}"

        info_str = (
            f"{classe_str}"
            f"ID: {i}<br>"
            f"{norma_str}"
            f"<br><br>"
            f"{features_str}"
        )
        formatted_text.append(info_str)

    norm_diff = np.abs(self.all_norms - 1)
    cmin_val = float(np.min(norm_diff))
    cmax_val = float(np.max(norm_diff))

    colorscale = [
        [0.0, 'green'],
        [0.5, 'yellow'],
        [1.0, 'red']
    ]

    fig = go.Figure(data=go.Scatter(
        x=low_dim_data[:, 0],
        y=low_dim_data[:, 1],
        mode='markers',
        marker=dict(
            size=7,
            color=norm_diff,
            colorscale=colorscale,
            cmin=cmin_val,
            cmax=cmax_val,
            showscale=True
        ),
        text=formatted_text,
        hovertemplate='%{text}<extra></extra>'
    ))

    mean_val = np.mean(self.all_norms)

    fig.update_layout(
        title=f"Interactive Plot | Mean Spectral Norm: {mean_val:.3f}",
        hovermode='closest',
        width=width,
        height=height,
    )

    fig.update_xaxes(showgrid=False, showticklabels=False, ticks='')
    fig.update_yaxes(showgrid=False, showticklabels=False, ticks='')


    fig.show()



def _importance_plot(self, width, height, n_top):

    phi_df = pd.DataFrame(self.all_phis, columns=[f'Feature {i}' for i in range(self.n_features)])
    if self.feature_names is not None:
        phi_df.columns = self.feature_names

    feature_sums = phi_df.sum()
    feature_sums_sorted = feature_sums.sort_values(ascending=False)
    phi_df = phi_df[feature_sums_sorted.index]

    feature_sums = phi_df.sum()
    feature_sums_sorted = feature_sums.sort_values(ascending=False)
    feature_sums_sorted = feature_sums_sorted.head(n_top)

    plt.figure(figsize=(width, height))
    plt.barh(feature_sums_sorted.index, feature_sums_sorted.values)
    plt.title("FADEx Feature Importance Plot")
    plt.xlabel("Importance Values")

    plt.tick_params(
        axis='y',          
        which='both',     
        labelsize=16
    )
    plt.xticks([])

    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # plt.savefig("synthetic_importance_ranking.svg", format="svg", dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.show()

class FADEx:
    '''
    Local explainability method for dimensionality reduction (DR) algorithms using Jacobian-based analysis.

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

    class_names : list of str, optional
        Names of the classes for each instance.

    RBF_kernel : str, default='cubic'
        The kernel used by the RBFInterpolator.

    pre_dr : int, optional
        If not None, applies preliminary dimensionality reduction (PCA) to the high-dimensional data,
        reducing it to `pre_dr` dimensions, before computing the h in the finite differences method.

    RBF_epsilon : float, default=0.001
        The epsilon parameter for the RBF kernel.

    RBF_degree : float, default=1
        The degree parameter for the RBF kernel.

    RBF_smoothing : float, default=0
        The smoothing parameter for the RBF kernel.

    use_GPU : bool, default=False
        If True, uses GPU acceleration for computations.

    dist_sample : int, optional
        Number of samples to use for distance computation. If None, all data points are used.
    '''
        
    def __init__(self, high_dim_data: np.ndarray, low_dim_data: np.ndarray, 
                n_neighbors: int = None, feature_names: list = None, 
                class_names: list = None, RBF_kernel: str = 'cubic',
                pre_dr : int = None, RBF_epsilon : float = 0.001, 
                RBF_degree : float = 1, RBF_smoothing : float = 0, 
                use_GPU : bool = False, dist_sample : int = None):

        self.n_neighbors = n_neighbors
        self.class_names=class_names
        self.RBF_kernel = RBF_kernel
        self.all_phis = None
        self.h = None
        self.pre_dr = pre_dr
        self.RBF_degree = RBF_degree
        self.RBF_smoothing = RBF_smoothing
        self.RBF_epsilon = RBF_epsilon
        self.use_GPU = use_GPU
        self.dist_sample = dist_sample

        if(self.use_GPU and not GPU_availability):
            raise ImportError("The GPU option has been selected, but the required libraries are not available. "
            "Please install cupy and cuml or set use_GPU=False.")
        elif(self.use_GPU and GPU_availability):
            self.high_dim_data = cp.array(high_dim_data)
            self.low_dim_data = cp.array(low_dim_data)

            self.xp = cp
            self.nbrs = CudaNearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')

            if(self.pre_dr is not None):
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

    def _compute_jac_vec(self, high_dim_point, low_dim_point, high_dim_nei, low_dim_nei, batch_size):

        if(self.use_GPU):

            jac = cp.zeros((len(low_dim_point), len(high_dim_point)), dtype=cp.float32)  
            x = high_dim_point

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
                        kernel=self.RBF_kernel, 
                        epsilon=self.RBF_epsilon, 
                        degree=self.RBF_degree, 
                        smoothing=self.RBF_smoothing
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
                    kernel=self.RBF_kernel,
                    epsilon=self.RBF_epsilon,
                    degree=self.RBF_degree,
                    smoothing=self.RBF_smoothing
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


    def fit(self, explain_index : int, show : bool = True, width : int = 10, height : int = 8, batch_size : int = 200, n_top : int = 10):
        '''
        Computes the feature importance for a specific instance in the dataset.

        Parameters
        ----------
        explain_index : int
            The index of the instance to explain.

        show : bool, default=True
            If True, displays the explanation plot.

        width : int, default=8
            The width of the plot.

        height : int, default=10
            The height of the plot.
        
        batch_size : int, default=200
            Batch size for the Jacobian computation.
        
        n_top : int
            Number of top features to be plotted.

        Returns
        -------
        phi : np.ndarray or cp.ndarray, shape (n_features,)
            The importance values for each feature.

        spectral_norm : float
            The spectral norm of the Jacobian matrix.
        '''

        high_dim_point = self.high_dim_data[explain_index]
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

        # Jacobian Computing
        jac = self._compute_jac_vec(
            high_dim_point,
            low_dim_point,
            high_dim_nei,
            low_dim_nei,
            batch_size
        )

        # Importance Computing
        phi = self._compute_importance(jac, high_dim_point)

        # Spectral Norm
        spectral_norm = self.xp.linalg.norm(jac, ord=2)

        if(np.isnan(spectral_norm)):
            raise ValueError("Spectral Norm is NaN.")
        

        if(show):
            _explanation_plot(self, phi, spectral_norm, explain_index, width, height, n_top)
                


        return phi, spectral_norm
    

    # Auxiliary function for parallelism
    def _compute_phi(self, i):
        phi, spec_norm = self.fit(i, show=False)
        return phi, spec_norm
    
    # Applies fit function in the entire dataset
    def _fit_all(self):
        results = [self._compute_phi(i) for i in tqdm(range(self.n_samples), desc="Processing samples", unit="sample")]

        all_phis, all_norms = zip(*results)
        self.all_phis = np.array([arr.get() if GPU_availability and isinstance(arr, cp.ndarray) else arr for arr in all_phis])
        self.all_norms = np.array([arr.get() if GPU_availability and isinstance(arr, cp.ndarray) else arr for arr in all_norms])



    def interactive_plot(self, width : int = 10, height : int =8):
        '''
        Generates an interactive plot that shows the feature importance for every instance.

        Parameters
        ----------
        width : int, optional
            The width of the plot.

        height : int, optional
            The height of the plot.

        '''

        if(self.all_phis is None):
            self._fit_all()

        _interactive_plot(self, width, height)

    def importance_plot(self, width : int = 10, height : int = 8, n_top : int = 10):
        '''
        Generates a plot of feature importance for all instances.

        Parameters
        ----------
        width : int, optional
            The width of the plot.

        height : int, optional
            The height of the plot.

        n_top : int
            Number of top features to be plotted.
        '''

        if(self.all_phis is None):
            self._fit_all()


        _importance_plot(self, width, height, n_top)