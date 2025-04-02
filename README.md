<center>
<h1><b>FADEx</b></h1>
A feature attribution for dimensionality reduction algorithms.
</center>

---

**FADEx** is a feature attribution method designed for dimensionality reduction (DR) algorithms. It locally approximates the mapping function of the DR algorithm using a Taylor expansion and derives feature importance scores through a Singular Value Decomposition (SVD). The local approximation is constructed via Radial Basis Function (RBF) interpolation, while the Jacobian matrix is estimated using the finite difference method.

When the data is mapped to a two-dimensional space by the DR algorithm, the importance of each high-dimensional feature $j$ is given by:

$$
\phi_j = \left|v_{1j} x_j\right| + \frac{\lambda_2}{\lambda_1} \left|v_{2j} x_j\right|
$$

where $v_{ij}$ denotes the $j$-th entry of the $i$-th right singular vector, $\lambda_i$ is the $i$-th singular value, and $x_j$ is the value of feature $j$. 

# Instalation

# FADEx Class

### Class Constructor

The FADEx Class has the following signature:

```
def __init__(self, high_dim_data: np.ndarray, low_dim_data: np.ndarray, 
                n_neighbors: int = None, feature_names: list = None, 
                classes_names: list = None, RBF_kernel: str = 'cubic',
                pre_dr : int = None, RBF_epsilon : float = 0.001, 
                RBF_degree : float = 1, RBF_smoothing : float = 0, 
                use_GPU : bool = False, dist_sample : int = None):
``` 

#### Required Parameters

**high_dim_data** (n_samples, n_features) - High dimensional space<br>
**low_dim_data** (n_samples, 2) - Low dimensional space <br>
**n_neighbors** - Number of neighbors to consider in the local approximation. When it's `None`, the entire dataset will be used. <br>

#### Optional Parameters

**feature_names** - A list with the feature names.<br>
**classes_names** - A list with the class for each sample.<br>
**RBF_kernel, RBF_epsilon, RBF_degree and RBF_smoothing** - `RBFInterpolator` parameters. <br>
**pre_dr** - If not None, applies preliminary dimensionality reduction (PCA) to the high-dimensional data, reducing it to `pre_dr` dimensions, before computing the h in the finite differences method. This is important to avoid the curse of dimensionality.<br>
**dist_sample** - Number of samples to use for distance computation. If None, all data points are used. This is important to avoid memory consumption. <br>
**use_GPU** - If True, uses GPU acceleration for computations.

### Fit Method