"""
A test code to try PCA for SD directionality detection.
"""
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


data = np.zeros((512, 512, 1000))  # 512x512 pixels, 1000 frames # TODO: use actual recording data
n_frames = data.shape[2]
# TODO: load example data
data_reshaped = data.reshape(512*512, n_frames)
standardized_data = (data_reshaped - np.mean(data_reshaped, axis=1, keepdims=True)) / np.std(data_reshaped, axis=1, keepdims=True)

pca = PCA(n_components=10)  # Start with the top 10 components
principal_components = pca.fit_transform(standardized_data)


for i in range(10):
    plt.plot(pca.components_[i])
    plt.title(f"Principal Component {i+1}")
    plt.show()

# reconstruct signal
selected_component = pca.components_[0]  # Choose the component representing the sweeping signal
reconstructed_signal = np.dot(principal_components[:, 0].reshape(-1, 1), selected_component.reshape(1, -1))
reconstructed_signal = reconstructed_signal.reshape((512, 512, 1000))


spatial_pattern = np.mean(reconstructed_signal, axis=2)
plt.imshow(spatial_pattern, cmap='hot')
plt.colorbar()
plt.title("Sweeping Signal Spatial Pattern")
plt.show()


explained_variance_ratio = pca.explained_variance_ratio_
plt.bar(range(10), explained_variance_ratio[:10])
plt.title("Explained Variance Ratio")
plt.show()


# TODO: use sklearn.decomposition.IncrementalPCA to handle large dataset?