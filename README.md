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

# Installation

Use the following command in your terminal to install the module without the GPU library dependencies:

```
pip install fadex-exp
```


If you want to enable **GPU acceleration**, you need to install the **RAPIDS** framework. Please follow the official instructions at [https://rapids.ai/start.html](https://rapids.ai/start.html).


# FADEx Class

### Class Constructor

The FADEx Class has the following signature:

```
def __init__(self, high_dim_data: np.ndarray, low_dim_data: np.ndarray, 
                n_neighbors: int = None, feature_names: list = None, 
                class_names: list = None, RBF_kernel: str = 'cubic',
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
**class_names** - A list with the class for each sample.<br>
**RBF_kernel, RBF_epsilon, RBF_degree, and RBF_smoothing** - `RBFInterpolator` parameters. <br>
**pre_dr** - If provided, a preliminary dimensionality reduction (PCA) is applied to the high-dimensional data, reducing it to pre_dr dimensions before computing the step size in the finite differences method. This is crucial to avoid the curse of dimensionality.<br>
**dist_sample** - The number of samples to use for distance computation. If set to `None`, all data points are used. This parameter helps reduce memory consumption. <br>
**use_GPU** - If True, uses GPU acceleration for computations.

### Fit Method

```
def fit(self, explain_index : int, show : bool = True, width : int = 10, height : int = 8, batch_size : int = 200):
```

This method applies the FADEx algorithm to a single data instance. When `show=True`, it displays the feature importance ranking for that specific point, as illustrated below:

![fit importance ranking](https://raw.githubusercontent.com/greffao/fadex/main/figs/fit.png)

### Importance Plot

```
def importance_plot(self, width : int = 10, height : int = 8, n_top : int = 10):
```

This method applies the FADEx algorithm to the entire dataset, sums the importance values for each feature, and plots a general feature importance ranking, as shown below:

![importance plot](https://raw.githubusercontent.com/greffao/fadex/main/figs/importance.png)

### Interactive Plot

```
def interactive_plot(self, width : int = 10, height : int =8):
```

This method applies the FADEx algorithm to the entire dataset and displays the results for each individual point in an interactive plot. The points are colored according to their spectral norm's deviation from 1, with red indicating the most distorted points and green indicating the least distorted ones.

![interactive plot](https://raw.githubusercontent.com/greffao/fadex/main/figs/interactive.png)