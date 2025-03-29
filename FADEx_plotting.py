import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def _image_explanation_plot(self, phi, spectral_norm, high_dim_point, p):

    if(self.useGPU):
        phi = phi.get()
        spectral_norm = spectral_norm.get()
        high_dim_point = high_dim_point.get()

    mean_phi = np.mean(phi)
    std_phi = np.std(phi)
    threshold = mean_phi + p * std_phi

    original_2d = high_dim_point.reshape(28, 28) # Olhar isso aqui depois
    explained_2d = original_2d.copy()   
    
    for i in range(phi.shape[0]):
        if phi[i] <= threshold:
            row = i // 28
            col = i % 28
            explained_2d[row, col] = 0

    # Plot lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # 1) Imagem original
    im0 = axes[0].imshow(original_2d, cmap='gray', interpolation='nearest')
    axes[0].set_title("Original Image")

    # 2) Imagem explicada
    im1 = axes[1].imshow(explained_2d, cmap='gray', interpolation='nearest')
    axes[1].set_title(f"Explanaiton (p={p})\nSpectral Norm={spectral_norm:.2f}")

    plt.tight_layout()
    plt.show()

def _explanation_plot(self, phi, spectral_norm, explain_index, width, height):

    if(self.useGPU):
        phi = phi.get()
        spectral_norm = spectral_norm.get()

    indices = np.argsort(np.abs(phi))[::-1]
    phi_sorted = phi[indices]
    feature_names_sorted = [self.feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(6, height))
    y_pos = np.arange(len(phi_sorted))
    ax.barh(y_pos, phi_sorted, color=['red' if val < 0 else 'green' for val in phi_sorted])
    ax.set_yticks(y_pos)
    ax.set_xticks([])
    ax.set_yticklabels(feature_names_sorted, fontsize=60)
    ax.invert_yaxis()

    ax.set_xlabel("Importance Values", fontsize=14)
    ax.set_title(f"Feature Importance for instance {explain_index} (spectral norm = {spectral_norm:.3f})")
    ax.axvline(x=0, color='black', linewidth=1) 
    plt.tight_layout()
    plt.show()

def _interactive_plot(self, width, height):

    if(self.useGPU):
        low_dim_data = self.low_dim_data.get()
    else:
        low_dim_data = np.array(self.low_dim_data)

    formatted_text = []
    for i, phi_row in enumerate(self.all_phis):

        if(~self.image):
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
        if self.classes_names is not None:
            classe_str = f"Class: {self.classes_names[i]}<br>"

        norma_str = f"Spectral Norm: {self.all_norms[i]:.3f}"

        if(~self.image):
            info_str = (
                f"{classe_str}"
                f"ID: {i}<br>"
                f"{norma_str}"
                f"<br><br>"
                f"{features_str}"
            )
        else:
            info_str = (
                f"{classe_str}"
                f"ID: {i}<br>"
                f"{norma_str}"
            )

        formatted_text.append(info_str)

    # all_norms = np.array([arr.get() if isinstance(arr, cp.ndarray) else arr for arr in all_norms])
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
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        hovermode='closest',
        width=width,
        height=height,
    )

    fig.update_xaxes(showgrid=False, showticklabels=False, ticks='')
    fig.update_yaxes(showgrid=False, showticklabels=False, ticks='')


    fig.show()


# def _interactive_plot(self, width=1000, height=800):

#     if self.useGPU:
#         low_dim_data = self.low_dim_data.get()
#     else:
#         low_dim_data = np.array(self.low_dim_data)
    
#     norm_diff = np.abs(self.all_norms - 1)
    

#     norm_diff_norm = (norm_diff - np.min(norm_diff)) / (np.max(norm_diff) - np.min(norm_diff))

#     norm_diff_norm = np.abs(norm_diff_norm - 1)

#     mean_val = np.mean(self.all_norms)
    
#     dpi = 100
#     fig_width = width / dpi
#     fig_height = height / dpi

#     plt.figure(figsize=(fig_width, fig_height))
    
#     scatter = plt.scatter(
#         low_dim_data[:, 0],
#         low_dim_data[:, 1],
#         c=norm_diff_norm,
#         cmap='viridis',
#         s=50
#     )
    
#     plt.colorbar()
#     # plt.title(f"Distortion Plot | Mean Spectral Norm: {mean_val:.3f}", fontsize=14)
#     # plt.xlabel("t-SNE 1", fontsize=12)
#     # plt.ylabel("t-SNE 2", fontsize=12)
#     plt.xticks([])
#     plt.yticks([])
    
#     plt.tight_layout()

#     plt.savefig("synthetic_spectral_norm_plot.svg", format="svg", dpi=300, bbox_inches='tight', pad_inches=0.1)
#     plt.show()


def _importance_plot(self, width, height, n_top):

    if(self.image):
    

        phi_summed = self.xp.sum(self.all_phis, axis=0)

        phi_image = phi_summed.reshape(28, 28) # Olhar isso aqui


        cmap = mcolors.LinearSegmentedColormap.from_list(
            "black_to_blue", ["black", "blue"]
        )

        # Plotar a imagem
        plt.figure(figsize=(4, 4)) # Olhar isso aqui
        im = plt.imshow(phi_image, cmap=cmap, interpolation='nearest')

        plt.title("FADEx Feature Importance plot")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()

    else:

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
            labelsize=14
        )
        plt.xticks([])

        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        # plt.savefig("synthetic_importance_ranking.svg", format="svg", dpi=300, bbox_inches='tight', pad_inches=0.1)

        plt.show()