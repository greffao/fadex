'''
FADEx — Feature Attribution and Distortion-based Explanation of
Dimensionality Reduction

Locally approximates the DR mapping with a weighted ridge least-squares fit,
then derives per-feature importance scores from the SVD of the Jacobian.

-------------------------------------------------------------------------------
Author  : Lucas Greff Meneses
Email   : lucasgreffmeneses@usp.br
GitHub  : https://github.com/greffao
-------------------------------------------------------------------------------

Notes
-----
- Requires scikit-learn, numpy, scipy, pandas, matplotlib, plotly, and tqdm.
- The code was commented partially by Claude Code

'''

# Numerics
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import plotly.graph_objects as go
import colorsys

# Utils
from tqdm.auto import tqdm
from typing import List, Optional, Tuple, Dict
import warnings

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_VAR_THRESHOLD  = 1e-6    # Minimum neighborhood variance to keep a feature
_MIN_KEEP       = 5       # Minimum number of features kept after variance filtering
_EPS            = 1e-12   # Small epsilon to avoid division by zero


def _build_cmap(n=256, darken=0.90):
    '''
    Builds a customized colormap based on the Spectral_r standard colormap from
    matplotlib.
    '''
    spectral = cm.get_cmap("Spectral_r", n)
    dark_colors = []
    for i in range(n):
        r, g, b, a = spectral(i / (n - 1))
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        v *= darken
        r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
        dark_colors.append((r2, g2, b2, a))
    return mcolors.LinearSegmentedColormap.from_list("spectral_r_dark", dark_colors)


def _cmap_to_plotly(cmap, n=256):
    """Convert matplotlib colormap to Plotly colorscale."""
    return [
        [i / (n - 1), mcolors.to_hex(cmap(i / (n - 1)))]
        for i in range(n)
    ]

_CMAP = _build_cmap()
_PLOTLY_CMAP = _cmap_to_plotly(_CMAP)


class FADEx:

    def __init__(self, high_dim_data: np.ndarray, low_dim_data: np.ndarray,
                 n_neighbors: int = None, feature_names: list = None,
                 class_names: list = None, remove_const_feat: bool = False,
                 use_pca: bool = False, ridge_lambda: float = 0.1,
                 verbose: bool = False):
        '''
        Local explainability method for dimensionality reduction (DR) algorithms.

        Parameters
        ----------
        high_dim_data : np.array, shape (n_samples, n_features)
            The dataset in the original high-dimensional space.

        low_dim_data : np.array, shape (n_samples, reduced_dimension)
            The dataset in the reduced dimension after transformation by DR algorithm.
            If you want to visualize FADEx explanations, the reduced dimension must be 2.

        n_neighbors : int
            Number of neighbors used in the Jacobian approximation. Recommended: 10% of
            the dataset size.

        feature_names : list, optional
            List with the feature names. If None, standard names (feature_0, feature_1, ...)
            will be used.

        class_names : list, optional, shape (n_samples,)
            List with the class name for each instance. If None, class names won't be displayed.

        remove_const_feat : bool, default=True
            Removes near-constant features (variance < 1e-6) before computing the local
            Jacobian. This helps deal with the curse of dimensionality.

        use_pca : bool, default=True
            Applies PCA retaining 95% of variance before Jacobian estimation, further
            reducing the effective dimensionality.

        ridge_lambda : float, default=0.1
            Regularization strength for the weighted ridge least-squares Jacobian estimation.

        verbose : bool, default=False
            If True, prints debug information during computation.
        '''

        # Hyperparameters
        self.class_names = class_names
        self.ridge_lambda = ridge_lambda
        self.verbose = verbose
        self.remove_const_feat = remove_const_feat
        self.use_pca = use_pca
        self.n_neighbors = n_neighbors

        # Internal state — populated by _fit_all()
        self.all_phis  = None
        self.all_norms = None
        self.all_jacs  = None

        self.high_dim_data = np.array(high_dim_data)
        self.low_dim_data  = np.array(low_dim_data)

        self.n_samples  = len(self.high_dim_data)
        self.n_features = len(self.high_dim_data[0])

        # Use generic labels when no feature names are provided
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
        else:
            self.feature_names = feature_names

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _drop_const_features(
        self,
        high_dim_point: np.ndarray,
        high_dim_nei: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Removes near-constant features from the local neighborhood to reduce
        the effective dimensionality before Jacobian estimation.

        Features whose variance across neighbors is below _VAR_THRESHOLD are
        dropped, but at least _MIN_KEEP features (those with the highest variance)
        are always retained.

        Returns
        -------
        high_dim_point_f : filtered query point
        high_dim_nei_f   : filtered neighborhood matrix
        keep             : boolean mask over original features
        kept_idx         : integer indices of kept features
        '''
        var  = np.var(high_dim_nei, axis=0)
        keep = var > _VAR_THRESHOLD

        # Guarantee a minimum number of features
        if int(keep.sum()) < _MIN_KEEP:
            top  = np.argsort(var)[-_MIN_KEEP:]
            keep = np.zeros_like(var, dtype=np.bool_)
            keep[top] = True

        kept_idx        = np.where(keep)[0]
        high_dim_nei_f  = high_dim_nei[:, kept_idx]
        high_dim_point_f = high_dim_point[kept_idx]

        return high_dim_point_f, high_dim_nei_f, keep, kept_idx

    def _reinflate_jac(self, jac_filtered: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
        '''
        Expands a Jacobian computed on the reduced feature set back to the full
        feature space, padding the dropped feature columns with zeros.
        '''
        keep_mask = np.asarray(keep_mask)
        jac_full = np.zeros((jac_filtered.shape[0], self.n_features), dtype=jac_filtered.dtype)
        jac_full[:, keep_mask] = jac_filtered
        return jac_full

    def _compute_importance(self, jacobian: np.ndarray, x: np.ndarray) -> np.ndarray:
        '''
        Derives per-feature importance scores from the Jacobian via SVD.
        '''
        _, S, VT = np.linalg.svd(jacobian, full_matrices=True)
        V = VT.T

        if V.shape[1] < 2:
            raise ValueError("Jacobian matrix does not have enough singular vectors.")

        phi = np.zeros_like(x)
        for j in range(len(x)):
            phi[j] = (np.abs(V[j, 0] * x[j])
                      + (S[1] / (S[0] + 1e-12)) * np.abs(V[j, 1] * x[j]))

        if np.isnan(phi).any():
            raise ValueError('NaN values computed for phi.')

        return phi

    def _nearest_neighbors(
        self, data: np.ndarray, point: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Returns the distances and indices of the k nearest neighbors of `point`
        within `data` using a ball-tree structure.
        '''
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree', n_jobs=2)
        nbrs.fit(data)
        dist, indices = nbrs.kneighbors(point.reshape(1, -1))
        return dist, indices

    @staticmethod
    def _ridge_wls(X: np.ndarray, Y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
        '''
        Solves the weighted ridge least-squares problem:
        '''
        p  = X.shape[1]
        Xw = X * w[:, None]                      # (k, p)
        A  = X.T @ Xw + lam * np.eye(p)          # (p, p)
        B  = X.T @ (Y * w[:, None])              # (p, d)

        return np.linalg.solve(A, B)

    def _compute_jac_ls(
        self,
        high_dim_point: np.ndarray,
        low_dim_point: np.ndarray,
        high_dim_nei: np.ndarray,
        low_dim_nei: np.ndarray,
    ) -> np.ndarray:
        '''
        Estimates the Jacobian of the DR mapping at `high_dim_point` using
        locally-weighted ridge least squares.

        Returns
        -------
        J : np.ndarray, shape (d, D)
        '''
        x  = np.asarray(high_dim_point, dtype=np.float64).ravel()   # (D,)
        y  = np.asarray(low_dim_point,  dtype=np.float64).ravel()   # (d,)
        Xn = np.asarray(high_dim_nei,   dtype=np.float64)           # (k, D)
        Yn = np.asarray(low_dim_nei,    dtype=np.float64)           # (k, d)

        # Local displacements centered at the query point
        delta_X = Xn - x[None, :]   # (k, D)
        delta_Y = Yn - y[None, :]   # (k, d)

        # Gaussian distance weights
        distances          = np.linalg.norm(delta_X, axis=1)                            # (k,)
        positive_distances = distances[distances > 0]
        sigma              = float(max(
            np.median(positive_distances) if positive_distances.size > 0 else 1.0,
            _EPS
        ))
        weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))             # (k,)

        # Solve the weighted ridge system and transpose to Jacobian shape
        Bt = self._ridge_wls(delta_X, delta_Y, weights, self.ridge_lambda)  # (D, d)
        J  = Bt.T.astype(np.float64)                                         # (d, D)
        return J

    def _get_top_features(
        self, phi: np.ndarray, n_top: int
    ) -> Tuple[np.ndarray, List[str]]:
        '''
        Returns the indices and names of the top-n features sorted by
        descending absolute importance score.
        '''
        top_indices = np.argsort(np.abs(phi))[::-1][:n_top]
        top_names   = [self.feature_names[i] for i in top_indices]
        return top_indices, top_names

    def _get_jacobian(self, idx: int) -> np.ndarray:
        '''
        Returns the Jacobian for instance `idx`. Uses the cached value from
        _fit_all() if available; otherwise calls fit() on the fly.
        '''
        if self.all_jacs is not None:
            return self.all_jacs[idx]
        return self.fit(idx, show=False)['jac']

    def _resolve_feature(self, feature) -> Tuple[int, str]:
        '''
        Maps a feature identifier (name string or integer index) to its
        (index, name) pair. Raises informative errors for invalid inputs.
        '''
        feature_names_list = list(self.feature_names)

        if isinstance(feature, str):
            if feature not in feature_names_list:
                raise ValueError(f"Feature '{feature}' not found in feature_names.")
            return feature_names_list.index(feature), feature

        elif isinstance(feature, (int, np.integer)):
            if not (0 <= feature < self.n_features):
                raise ValueError(f"Feature index {feature} is out of bounds.")
            return int(feature), feature_names_list[int(feature)]

        else:
            raise TypeError("Feature must be a string or integer.")

    def _configure_clean_axes(self, ax) -> None:
        '''
        Removes all ticks, tick labels, and spines from a matplotlib Axes,
        producing a clean embedding plot with no decorations.
        '''
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _compute_vector_color_and_scale(
        self,
        norm: float,
        vec: np.ndarray,
        max_norm: float,
        cmap,
        vec_mult: float,
    ) -> Tuple:
        '''
        Computes the display color and scaled arrow vector for a grid cell based
        on the vector's magnitude norm:

        - norm < 1  : plotted at natural scale with the lowest colormap color
                      (no distortion signal).
        - norm >= 1 : capped to unit length; color maps linearly from the lowest
                      color (norm=1) to the highest (norm=max_norm).

        Returns
        -------
        color    : RGBA tuple from the colormap
        plot_vec : scaled vector to display
        '''
        if norm < 1.0:
            color    = cmap(0.0)
            plot_vec = vec_mult * vec
        else:
            # Cap length to 1 and map excess magnitude to colormap range [0, 1]
            plot_vec  = vec_mult * (vec / norm)
            color_val = (norm - 1.0) / (max_norm - 1.0) if max_norm > 1.0 else 0.0
            color     = cmap(color_val)

        return color, plot_vec

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def fit(self, explain_index: int, show: bool = True, width: int = 10,
            height: int = 8, n_top: int = 10, save_fig: bool = False) -> dict:
        '''
        Computes the FADEx explanation for a single instance.

        Steps:
          1. Retrieve the local neighborhood.
          2. Optionally filter near-constant features and apply PCA.
          3. Estimate the Jacobian via weighted ridge least squares.
          4. Derive per-feature importance scores through SVD of the Jacobian.
          5. Compute the spectral norm as a local distortion measure.

        Parameters
        ----------
        explain_index : int
            Index of the instance to explain.
        show : bool, default=True
            If True, displays the feature importance bar chart.
        width, height : int
            Figure dimensions in inches.
        n_top : int
            Number of top features to display.
        save_fig : bool, default=False
            If True, saves the plot as a PDF.

        Returns
        -------
        dict with keys:
            "phi"               — importance scores, shape (n_features,)
            "SND"               — spectral norm of the Jacobian (float)
            "jac"               — full Jacobian, shape (d, n_features)
            "feature_names_top" — names of the top-n features
        '''
        high_dim_point = self.high_dim_data[explain_index]   # (D,)
        low_dim_point  = self.low_dim_data[explain_index]    # (d,)

        # --- Neighborhood selection ---
        if self.n_neighbors is None:
            high_dim_nei = self.high_dim_data
            low_dim_nei  = self.low_dim_data
        else:
            _, indices   = self._nearest_neighbors(data=self.high_dim_data, point=high_dim_point)
            idx          = indices[0] if indices.ndim == 2 else indices
            high_dim_nei = self.high_dim_data[idx]
            low_dim_nei  = self.low_dim_data[idx]

        # Store the original point before any preprocessing
        x0 = high_dim_point

        # --- Dimensionality reduction preprocessing ---
        if self.remove_const_feat:
            high_dim_point, high_dim_nei, keep_mask, _ = self._drop_const_features(
                high_dim_point, high_dim_nei
            )

        if self.use_pca:
            pca          = PCA(n_components=0.95, svd_solver='full')
            high_dim_nei = pca.fit_transform(high_dim_nei)
            high_dim_point = pca.transform(high_dim_point[None, :])[0]

        if self.verbose: print(f'[FADEx] New Dimension: {high_dim_nei.shape[1]}')

        # --- Jacobian estimation ---
        jac = self._compute_jac_ls(high_dim_point, low_dim_point, high_dim_nei, low_dim_nei)

        # Validate and sanitize
        if np.any(np.isnan(jac)):
            raise ValueError('NaN values computed for Jacobian.')
        if np.any(np.isinf(jac)):
            warnings.warn('Inf values computed for Jacobian. Turning them into 0 or max...')
        jac = np.nan_to_num(jac)
        jac = jac.astype(np.float32)

        # --- Restore full feature space ---
        if self.use_pca:
            jac = jac @ pca.components_
        if self.remove_const_feat:
            jac = self._reinflate_jac(jac, keep_mask)

        # --- Importance and distortion ---
        phi           = self._compute_importance(jac, x0)
        SND = np.linalg.norm(jac, ord=2)

        if np.isnan(SND):
            raise ValueError("Spectral Norm is NaN.")

        # --- Top features ---
        _, feature_names_top = self._get_top_features(phi, n_top)

        if show:
            self._explanation_plot(phi, SND, explain_index,
                                   feature_names_top, width, height, n_top, save_fig)

        return {
            "phi":               phi,
            "SND":               SND,
            "jac":               jac,
            "feature_names_top": feature_names_top,
        }

    def _fit_all(self) -> None:
        '''
        Runs fit() on every instance and caches results in self.all_phis,
        self.all_norms, and self.all_jacs.

        Called automatically by interactive_plot(), importance_plot(),
        plot_feature_heatmap(), and plot_grid_feature_vectors() when cached
        results are not yet available.
        '''

        # Turns verbose off to avoid clutter
        switch = False
        if self.verbose: 
            self.verbose = False
            switch = True

        results = [
            self.fit(i, show=False)
            for i in tqdm(range(self.n_samples), desc="[FADEx] Processing Instances", unit="instance")
        ]

        if switch:
            self.verbose = True

        self.all_phis  = np.asarray([r["phi"]          for r in results])
        self.all_norms = np.asarray([r["SND"]          for r in results])
        self.all_jacs  = np.asarray([r["jac"]          for r in results])

    def fit_cluster(self, cluster_ids: np.ndarray, n_top: int = 10,
                    show: bool = False, save_fig: bool = False) -> Dict:
        '''
        Computes a cluster-level feature importance by averaging phi over all
        instances in `cluster_ids`.

        Parameters
        ----------
        cluster_ids : array-like of int
            Indices of the instances belonging to the cluster.
        n_top : int
            Number of top features to return and optionally display.
        show : bool, default=False
            If True, displays the aggregated feature importance bar chart.
        save_fig : bool, default=False
            If True, saves the plot as a PDF.

        Returns
        -------
        phi_cluster : np.ndarray, shape (n_features,)
            Averaged importance scores over the cluster.
        feature_names_top : list of str
            Names of the top-n features.
        '''
        phi_cluster = np.zeros(shape=self.n_features)

        for idx in tqdm(cluster_ids):
            phi_cluster += self.fit(idx, show=False, n_top=n_top)["phi"]

        # Normalize by cluster size
        phi_cluster /= len(cluster_ids)

        _, feature_names_top = self._get_top_features(phi_cluster, n_top)

        if show:
            self._explanation_plot(phi_cluster, spectral_norm=None, explain_index=None,
                                   feature_names_top=feature_names_top, n_top=n_top,
                                   save_fig=save_fig)

        return {
            "phi_cluster":       phi_cluster,
            "feature_names_top": feature_names_top
        }

    def plot_distortion(self, width: int = 10, height: int = 10, save_fig: bool = False) -> None:
        '''
        Plots the projected space colored with the SND distortion metric. Cold colors correspond
        to shrinking (SND < 1) and warm colors correspond to stretching (SND > 1).
        '''

        if self.all_norms is None:
            self._fit_all()
        
        low_dim_data = self.low_dim_data
        norms = np.asarray(self.all_norms, dtype=float).flatten()

        vmin = float(np.nanmin(norms))
        vmax = float(np.nanmax(norms))
        vmin = min(vmin, 1.0 - 1e-9)
        vmax = max(vmax, 1.0 + 1e-9)
        div_norm = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(figsize=(width, height))
        sc = ax.scatter(
            low_dim_data[:, 0],
            low_dim_data[:, 1],
            c=norms,
            cmap=_CMAP,
            norm=div_norm,
            s=20,
            alpha=0.7,
        )

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Spectral Norm", rotation=270, labelpad=15)
        cbar.set_ticks([vmin, 1.0, vmax])
        cbar.set_ticklabels([f"{vmin:.2f}", "1.00", f"{vmax:.2f}"])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        plt.tight_layout()

        if save_fig:
            fig.savefig('FADEx_SND_distortion_plot.pdf')

        plt.show()

    def interactive_plot(self, width: int = 10, height: int = 8, 
                        show_features: bool = False) -> None:
        '''
        Generates an interactive Plotly scatter plot of the 2D embedding.

        Each point is colored by its spectral norm and has a hover tooltip
        showing sorted feature importances. Calls _fit_all() automatically
        if results are not yet cached.
        '''
        if self.all_phis is None:
            self._fit_all()

        width_px  = width  * 100
        height_px = height * 100

        # Build hover tooltip for each instance
        formatted_text = []
        for i, phi_row in enumerate(self.all_phis):

            if show_features:
                # Sort features by absolute importance (descending)
                sorted_features = sorted(
                    zip(self.feature_names, phi_row),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                features_str = "<br>".join(f"{name}: {val:.3f}" for name, val in sorted_features)
            else: features_str = ""

            # Optional class label prefix
            class_str = ""
            if self.class_names is not None:
                class_str = f"Class: {self.class_names[i]}<br>"

            norm_str = f"Spectral Norm: {self.all_norms[i]:.3f}"
            formatted_text.append(f"{class_str}ID: {i}<br>{norm_str}<br><br>{features_str}")

        norms = np.asarray(self.all_norms, dtype=float)

        # Normalizing so 1 maps to 0.5 (no distortion color)
        vmin = float(np.nanmin(norms))
        vmax = float(np.nanmax(norms))
        vmin = min(vmin, 1.0 - 1e-9)
        vmax = max(vmax, 1.0 + 1e-9)

        normed = np.where(
            norms <= 1.0,
            0.5 * (norms - vmin) / (1.0 - vmin),
            0.5 + 0.5 * (norms - 1.0) / (vmax - 1.0),
        )

        fig = go.Figure(data=go.Scatter(
            x=self.low_dim_data[:, 0],
            y=self.low_dim_data[:, 1],
            mode='markers',
            marker=dict(
                size=7,
                color=normed,
                colorscale=_PLOTLY_CMAP,
                cmin=0,
                cmax=1,
                showscale=True,
                colorbar=dict(
                    tickvals=[0, 0.5, 1],
                    ticktext=[f"{vmin:.2f}", "1.00", f"{vmax:.2f}"],
                ),
            ),
            text=formatted_text,
            hovertemplate='%{text}<extra></extra>',
        ))

        fig.update_layout(
            title=f"Interactive Plot",
            hovermode='closest',
            width=width_px,
            height=height_px,
        )
        fig.update_xaxes(showgrid=False, showticklabels=False, ticks='')
        fig.update_yaxes(showgrid=False, showticklabels=False, ticks='')
        fig.show()

    def importance_plot(self, width: int = 8, height: int = 8,
                        n_top: int = 10, save_fig: bool = False) -> None:
        '''
        Plots a global feature importance ranking by summing phi across all
        instances. Calls _fit_all() automatically if needed.

        Parameters
        ----------
        width, height : int
            Figure dimensions in inches.
        n_top : int
            Number of top features to display.
        save_fig : bool, default=False
            If True, saves the plot as a PDF.
        '''
        if self.all_phis is None:
            self._fit_all()

        # Sum importance across instances, sort, and pick top-n
        phi_df = pd.DataFrame(self.all_phis, columns=self.feature_names)
        feature_sums_sorted = phi_df.sum().sort_values(ascending=False).head(n_top)

        plt.figure(figsize=(width, height))
        plt.barh(feature_sums_sorted.index, feature_sums_sorted.values)
        plt.tick_params(axis='y', which='both', labelsize=16)
        plt.xticks([])
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_fig:
            plt.savefig("FADEx_importance_plot.pdf")

        plt.show()

    def _explanation_plot(self, phi: np.ndarray, spectral_norm: Optional[float],
                          explain_index: Optional[int], feature_names_top: List[str],
                          width: int = 4, height: int = 4,
                          n_top: int = 10, save_fig: bool = False) -> List[str]:
        '''
        Plots a horizontal bar chart of per-feature importance for a single
        instance or a cluster aggregate.

        Bars are colored green (positive) or red (negative). The title includes
        the spectral norm when available.
        '''
        # Extract the top-n phi values in importance-descending order
        top_indices = np.argsort(np.abs(phi))[::-1][:n_top]
        phi_top     = phi[top_indices]

        plt.figure(figsize=(width, height))
        y_pos  = np.arange(len(phi_top))
        colors = ['red' if val < 0 else 'green' for val in phi_top]

        plt.barh(y_pos, phi_top, color=colors)
        plt.yticks(y_pos, feature_names_top, fontsize=12)
        plt.xticks([])
        plt.gca().invert_yaxis()
        plt.axvline(x=0, color='black', linewidth=1)

        # Build title depending on context (single instance vs. cluster)
        title = (f"Feature Importance for instance {explain_index}"
                 if explain_index is not None
                 else "Cluster Feature Importance")
        if spectral_norm is not None:
            title += f" (spectral norm = {spectral_norm:.3f})"
        plt.title(title)

        plt.tight_layout()
        if save_fig:
            plt.savefig(f'FADEx_explanation_plot_instance_{explain_index}_top_{n_top}.pdf')

        plt.show()
        return feature_names_top

    def plot_grid_feature_vectors(self, feature, labels=None,
                                  mask=None, grid_bins: int = 15, min_points: int = 1,
                                  scale_factor: float = 1.0, width: int = 12,
                                  height: int = 10, save_fig: bool = False) -> None:
        """
        Overlays Jacobian-based feature vectors on the 2D embedding using a grid.

        The embedding plane is divided into a (grid_bins × grid_bins) grid. For
        each occupied cell the mean Jacobian is computed and the selected feature
        column is drawn as an arrow. Arrow color encodes magnitude:

        - norm < 1  : minimum colormap color.
        - norm >= 1 : length capped to 1, color scales with excess magnitude.

        Parameters
        ----------
        feature : str or int
            Feature to visualize (by name or index).
        labels : array-like, optional
            Class labels for background scatter coloring.
        mask : array-like of bool, optional
            Boolean mask to highlight a subset of points.
        grid_bins : int, default=15
            Number of bins along each axis of the grid.
        min_points : int, default=1
            Minimum points required in a cell before drawing vectors.
        scale_factor : float, default=1.0
            Additional multiplier applied to vector magnitudes.
        width, height : int
            Figure dimensions in inches.
        save : bool, default=False
            If True, saves the plot as a PDF.
        """

        feat_idx, feat_name = self._resolve_feature(feature)

        # --- Build 2D grid over the embedding ---
        x_data = self.low_dim_data[:, 0]
        y_data = self.low_dim_data[:, 1]

        x_margin = (x_data.max() - x_data.min()) * 0.05
        y_margin = (y_data.max() - y_data.min()) * 0.05

        x_bins = np.linspace(x_data.min() - x_margin, x_data.max() + x_margin, grid_bins + 1)
        y_bins = np.linspace(y_data.min() - y_margin, y_data.max() + y_margin, grid_bins + 1)

        x_indices = np.digitize(x_data, x_bins) - 1
        y_indices = np.digitize(y_data, y_bins) - 1

        # --- Background scatter ---
        fig, ax = plt.subplots(figsize=(width, height))

        if labels is not None:
            num_classes   = len(np.unique(labels))
            discrete_cmap = plt.get_cmap('Pastel1', num_classes)
            if mask is not None:
                mask = np.array(mask, dtype=bool)
                ax.scatter(x_data[~mask], y_data[~mask], color='lightgray',
                           s=30, alpha=0.3, edgecolors='none', zorder=1)
                ax.scatter(x_data[mask], y_data[mask], c=labels[mask],
                           cmap=discrete_cmap, s=30, alpha=0.9, edgecolors='none', zorder=2)
            else:
                ax.scatter(x_data, y_data, c=labels, cmap=discrete_cmap,
                           s=30, alpha=0.75, edgecolors='none', zorder=1)
        else:
            ax.scatter(x_data, y_data, s=30, color='lightgray',
                       alpha=0.75, edgecolors='none', zorder=1)

        # --- Collect occupied cells ---
        valid_cells = []
        for i in range(grid_bins):
            for j in range(grid_bins):
                pts_in_cell = np.where((x_indices == i) & (y_indices == j))[0]
                if len(pts_in_cell) >= min_points:
                    valid_cells.append((i, j, pts_in_cell))

        # --- Compute mean Jacobian per cell and collect arrow data ---
        vectors_data = []
        max_norm     = 1.0

        for i, j, pts_idx in tqdm(valid_cells, desc="[FADEx] Computing grid cells"):
            cx = (x_bins[i] + x_bins[i + 1]) / 2.0
            cy = (y_bins[j] + y_bins[j + 1]) / 2.0

            mean_jac = np.mean([self._get_jacobian(idx) for idx in pts_idx], axis=0)
            vec  = mean_jac[:, feat_idx] * scale_factor
            norm = np.linalg.norm(vec)
            if norm > max_norm:
                max_norm = norm
            vectors_data.append({'cx': cx, 'cy': cy, 'vec': vec, 'norm': norm})

        # --- Draw arrows ---
        cmap     = plt.get_cmap('plasma_r')
        vec_mult = 0.8 * (x_data.max() - x_data.min()) / grid_bins

        for data in vectors_data:
            cx, cy = data['cx'], data['cy']
            color, plot_vec = self._compute_vector_color_and_scale(
                data['norm'], data['vec'], max_norm, cmap, vec_mult
            )
            ax.annotate(
                "",
                xy=(cx + plot_vec[0], cy + plot_vec[1]),
                xytext=(cx, cy),
                arrowprops=dict(
                    arrowstyle="simple, tail_width=1.0, head_width=1.4, head_length=1.4",
                    facecolor=color,
                    edgecolor='black',
                    linewidth=1.0,
                    alpha=1.0,
                ),
                zorder=4,
            )

        # --- Colorbar ---
        if max_norm > 1.0:
            norm_cb = mcolors.Normalize(vmin=1.0, vmax=max_norm)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_cb)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Original vector magnitude (capped at 1)", rotation=270, labelpad=15)

        ax.set_title(f"Vector Field: {feat_name}", fontsize=14)
        ax.axis('equal')

        self._configure_clean_axes(ax)
        plt.tight_layout()
        if save_fig:
            safe_name = str(feat_name).replace("/", "_").replace(" ", "_")
            plt.savefig(f'FADEx_influence_vectors_{safe_name}.pdf')
        plt.show()

    def plot_feature_heatmap(self, feature, gridsize: int = 30, cmap: str = 'magma',
                             width: int = 10, height: int = 8, save: bool = False) -> None:
        """
        Renders a hexbin heatmap on the 2D embedding showing where a specific
        feature has the highest importance (phi value).

        The figure background is set to the darkest tone of the colormap for
        contrast, and real data points are shown as faint white dots to provide
        spatial context.

        Parameters
        ----------
        feature : str or int
            The feature to visualize (by name or index).
        gridsize : int, default=30
            Hexbin resolution (number of hexagons across the x-axis).
        cmap : str, default='magma'
            Matplotlib colormap name.
        width, height : int
            Figure dimensions in inches.
        save : bool, default=False
            If True, saves the plot as a PDF.
        """
        # --- Feature validation ---
        feature_idx, feature_name = self._resolve_feature(feature)

        # --- Ensure all phis are computed ---
        if self.all_phis is None:
            self._fit_all()

        x_data     = self.low_dim_data[:, 0]
        y_data     = self.low_dim_data[:, 1]
        phi_values = np.abs(np.asarray(self.all_phis)[:, feature_idx])

        # --- Figure with dark background matching the colormap's lowest tone ---
        colormap = plt.get_cmap(cmap)
        bg_color = colormap(0.0)

        fig, ax = plt.subplots(figsize=(width, height))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        # Faint white dots for spatial context
        ax.scatter(x_data, y_data, s=10, color='white', alpha=0.10, edgecolors='none', zorder=1)

        # --- Hexbin heatmap: each cell shows the mean phi of its points ---
        hb = ax.hexbin(
            x_data, y_data,
            C=phi_values,
            gridsize=gridsize,
            cmap=cmap,
            reduce_C_function=np.mean,
            alpha=1.0,
            edgecolors='none',
            mincnt=1,
            zorder=2,
        )

        # --- Colorbar with white labels to contrast against the dark background ---
        cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean Importance (Phi)", rotation=270, labelpad=15, color='white')
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

        ax.set_title(f"Feature Importance Heatmap: {feature_name}", fontsize=14, color='white')
        ax.axis('equal')

        self._configure_clean_axes(ax)
        plt.tight_layout()

        if save:
            plt.savefig(f'FADEx_heatmap_importance_{feature_name}.pdf')

        plt.show()
