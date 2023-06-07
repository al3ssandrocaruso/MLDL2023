import copy

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

"""
inputs: pickle file containing all features related to a Domain, it's a dictionary with one single key whose value is a list of dictionaries (of len 1500 for D1 train)
each clip-action is a dictionary: {'uid':int, 'video_name': str, 'features_RGB': np.array of shape (5, 1024)

"""

data_path = "./inputs"

def getImage(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=zoom)

def main():
    # EXTRACT LABELS/narration
    train_val = pd.read_pickle("../train_val/D1_train.pkl")
    train_val = train_val.set_index('uid')
    print(train_val.head().to_string())

    # Pickle file is a list of dictionaries, each storing features data
    train_pickle_file = pd.read_pickle(os.path.join(data_path, "dense_5_D1_train.pkl"))
    print(train_pickle_file["features"][1])

    # extract all central frames
    central_frame_dict = extract_central_frames()

    ###################################################################

    # join two pickles to get LABELS and FEATURES
    for inner_dict in train_pickle_file["features"]:
        uid = inner_dict['uid']
        if uid in train_val.index:
            inner_dict['narration'] = train_val.loc[uid, 'narration']

    ###################################################################

    features_2d = []
    uid_s = []
    true_labels = []
    image_paths = []
    # Extract usful data of dictionaries
    for train_act in train_pickle_file["features"][:100]:
        # Flattened RGB features LIST
        feature_arr = np.array(train_act["features_RGB"]).flatten()
        features_2d.append(feature_arr)

        # Label LIST
        label = train_act["narration"]
        true_labels.append(label)

        # Uid LIST
        uid = train_act["uid"]
        uid_s.append(uid)

        # Image paths LIST
        central_frame_path = central_frame_dict[uid]
        image_paths.append(central_frame_path)

    ############################################
    # feature reduction to have better KNN
    features_2d_copy = copy.deepcopy(features_2d)

    features_2d = np.array(features_2d)
    pca = PCA(50)
    features_2d = pca.fit_transform(features_2d)
    ###################################################################
    # PCA for plotting
    features_2d_copy = np.array(features_2d_copy)

    # PCA
    pca = PCA(2)
    scaled_arr = pca.fit_transform(features_2d_copy)

    # TSNE
    # tsne = TSNE(n_components=2)
    # scaled_arr = tsne.fit_transform(features_2d_copy)

    # Scaled array contains scaled data after PCA, order is preserved
    ###################################################################

    # K-NN search
    n_neighbors = 8
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(features_2d)

    # Randomly select a feature vector
    # random_idx = np.random.randint(len(features_2d))
    random_idx = 56
    query_feature = features_2d[random_idx]

    # Perform k-NN search
    _, indices = knn.kneighbors([query_feature])

    # Retrieve the nearest neighbors' information
    nn_features_PCA = scaled_arr[indices.flatten()]
    nn_image_paths = [image_paths[i] for i in indices.flatten()]
    nn_true_labels = [true_labels[i] for i in indices.flatten()]

    ###################################################################

    # Call the plot_KNN function
    plot_KNN(nn_features_PCA, nn_true_labels, nn_image_paths, random_idx)




# def plot_KNN(features_PCA, labels_pred, image_paths, random_index, zoom=0.2):
#     dpi = 300  # Set the DPI value here
#     fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)  # Set the figsize parameter here
#     ax.scatter(features_PCA[:, 0], features_PCA[:, 1], s=0, cmap='viridis')
#     ax.set_xticks([])  # Remove x-axis scale (ticks)
#     ax.set_yticks([])  # Remove y-axis scale (ticks)
#     for x0, y0, path, label in zip(features_PCA[:, 0], features_PCA[:, 1], image_paths, labels_pred):
#         ab = AnnotationBbox(getImage(path, zoom), (x0, y0), frameon=False)
#         ax.add_artist(ab)
#         ax.text(x0, y0, label, color='white', fontsize=8, ha='center', va='center')
#     plt.title(random_index)
#     plt.show()


def plot_KNN(features_PCA, labels_pred, image_paths, random_index):
    dpi = 300  # Set the DPI value here
    zoom = 0.2  # Set the zoom value here
    fig, ax = plt.subplots(dpi=dpi)
    ax.scatter(features_PCA[:, 0], features_PCA[:, 1], s=0, cmap='viridis')
    for x0, y0, path, label in zip(features_PCA[:, 0], features_PCA[:, 1], image_paths, labels_pred):
        ab = AnnotationBbox(getImage(path, zoom), (x0, y0), frameon=False)
        ax.add_artist(ab)

    # for x0, y0, label in zip(features_PCA[:, 0], features_PCA[:, 1], labels_pred):
    #     ax.text(x0, y0 + 0.6, label, color='black', fontsize=8, ha='center', va='center')

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_visible(False)

    ax.set_xlabel("Principal component 1", fontsize=10, ha="right", va="top")  # X-axis label
    ax.set_ylabel("Principal component 2", fontsize=10, ha="left", va="top")  # Y-axis label

    ax.xaxis.set_label_coords(1, -0.1)  # Adjust x-axis label position
    ax.yaxis.set_label_coords(-0.1, 0.7)  # Adjust y-axis label position

    plt.show()

# def plot_KNN(features_PCA, labels_pred, image_paths, random_index):
#     dpi = 300  # Set the DPI value here
#     zoom = 0.2  # Set the zoom value here
#     fig, ax = plt.subplots(dpi=dpi)
#     ax.scatter(features_PCA[:, 0], features_PCA[:, 1], s=0, cmap='viridis')
#
#     for x0, y0, path, label in zip(features_PCA[:, 0], features_PCA[:, 1], image_paths, labels_pred):
#         ab = AnnotationBbox(OffsetImage(plt.imread(path), zoom=zoom), (x0, y0), frameon=False)
#         ax.add_artist(ab)
#
#     ax.set_xlabel("Principal component 1", fontsize=10, ha="right", va="top")  # X-axis label
#     ax.set_ylabel("Principal component 2", fontsize=10, ha="left", va="top")  # Y-axis label
#
#     ax.xaxis.set_label_coords(1, -0.1)  # Adjust x-axis label position
#     ax.yaxis.set_label_coords(-0.1, 0.7)  # Adjust y-axis label position
#
#     plt.show()


# def plot_and_save_KNN(features_PCA, labels_pred, image_paths, random_index, zoom=0.2, out_folder=r"C:\Users\bracc\Desktop\Clustering"):
#     import os
#     import matplotlib.pyplot as plt
#     from matplotlib.offsetbox import AnnotationBbox, OffsetImage
#
#     fig, ax = plt.subplots()
#     ax.scatter(features_PCA[:, 0], features_PCA[:, 1], s=0, cmap='viridis')
#     for x0, y0, path, label in zip(features_PCA[:, 0], features_PCA[:, 1], image_paths, labels_pred):
#         ab = AnnotationBbox(getImage(path, zoom), (x0, y0), frameon=False)
#         ax.add_artist(ab)
#         ax.text(x0, y0, label, color='white', fontsize=8, ha='center', va='center')
#     plt.title(random_index)
#
#     # Generate the file name
#     file_name = f"plot{random_index}.jpg"
#
#     # Save the figure with high DPI
#     fig.savefig(os.path.join(out_folder, file_name), dpi=300)
#
#     ####################################




def extract_central_frames():
    # read pickle file to find central frame information
    train_val = pd.read_pickle("../train_val/D1_train.pkl")
    train_val = train_val.set_index('uid')

    # for each action find central frame and store it a dictionary
    s_dict = dict()
    for uid, line in train_val.iterrows():
        central_frame = (int(line['start_frame']) + int(line['stop_frame'])) // 2 + 1
        central_frame = f'{central_frame}'.zfill(10)    # used to match format of path name of image
        video_id = line['video_id']
        s_dict[uid] =  f'C:/Users/bracc/Desktop/p08/{video_id}/img_{central_frame}.jpg'

    return s_dict

def pca_number_of_components(features_2d):
    """
    80% = 50 components
    """
    # Test of PCA
    pca = PCA()
    pca.fit(features_2d)
    # Calculate the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Calculate the cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    ##################################################################
    # Increase the figure size
    plt.figure(figsize=(8, 6))

    # Plot the cumulative explained variance
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Cumulative Explained Variance')

    # Add lines for the thresholds
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
    plt.axhline(y=0.9, color='g', linestyle='--', label='90% Threshold')
    plt.axhline(y=0.95, color='b', linestyle='--', label='95% Threshold')

    # Find the number of components for the thresholds
    n_components_80 = np.argmax(cumulative_variance >= 0.8) + 1
    n_components_90 = np.argmax(cumulative_variance >= 0.9) + 1
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

    # Add vertical lines for the number of components
    plt.axvline(x=n_components_80, color='r', linestyle='--')
    plt.axvline(x=n_components_90, color='g', linestyle='--')
    plt.axvline(x=n_components_95, color='b', linestyle='--')

    # Add text annotations for the number of components
    plt.text(n_components_80 + 5, 0.81, f'{n_components_80}', color='r')
    plt.text(n_components_90 + 5, 0.91, f'{n_components_90}', color='g')
    plt.text(n_components_95 + 5, 0.96, f'{n_components_95}', color='b')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()



if __name__ == "__main__":
    main()
