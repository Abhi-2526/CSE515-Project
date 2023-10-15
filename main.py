import warnings
from collections import defaultdict
import torch
import json
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import Caltech101
from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np
from PIL import Image
import pandas
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
import sqlite3
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from skimage.color import rgb2gray
import torch
import os, sys
from tqdm import tqdm
import networkx as nx
import datetime
import types
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


warnings.simplefilter(action='ignore', category=Warning)

# Initialize the ResNet50 model
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

conn = sqlite3.connect('image_features.db')
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS features (
    imageID INTEGER PRIMARY KEY,
    label TEXT,
    ColorMoments BLOB,
    HOG BLOB,
    AvgPool BLOB,
    Layer3 BLOB,
    FCLayer BLOB,
    RESNET BLOB
)
''')


def store_in_database(imageID, features):
    # Convert numpy arrays to bytes for storage
    n0_usage, label = dataset[imageID]
    # print("Size of ColorMoments before storing:", features['ColorMoments'].shape)
    ColorMoments_bytes = features['ColorMoments'].astype(np.float32).tobytes()
    # print("Size of ColorMoments before storing:", len(ColorMoments_bytes))

    HOG_bytes = features['HOG'].astype(np.float32).tobytes()

    # Reduce dimensionality of ResNet-AvgPool and ResNet-Layer3 features
    ResNetAvgPool1024 = np.array(features['AvgPool'])
    AvgPool_bytes = ResNetAvgPool1024.tobytes()
    ResNetLayer31024 = np.array(features['Layer3'])
    Layer3_bytes = ResNetLayer31024.tobytes()
    FCLayer_bytes = features['FCLayer'].tobytes()
    ResNetOutput_bytes = features['RESNET'].tobytes()

    # Check if imageID already exists in the database
    cursor.execute("SELECT 1 FROM features WHERE imageID=?", (imageID,))
    exists = cursor.fetchone()

    if not exists:
        # Insert a new record if imageID doesn't exist
        cursor.execute('''
        INSERT INTO features (imageID, label, ColorMoments, HOG, AvgPool, Layer3, FCLayer, RESNET)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            imageID, label, ColorMoments_bytes, HOG_bytes, AvgPool_bytes, Layer3_bytes, FCLayer_bytes, ResNetOutput_bytes))

        conn.commit()


def load_features_from_database():
    cursor.execute("SELECT imageID, label, ColorMoments, HOG, AvgPool, Layer3, FCLayer, RESNET FROM features")
    rows = cursor.fetchall()
    for row in rows:
        database.append({
            "imageID": row[0],
            "label": row[1],
            "features": {
                "ColorMoments": np.frombuffer(row[2], dtype=np.float32),
                "HOG": np.frombuffer(row[3], dtype=np.float32),
                "AvgPool": np.frombuffer(row[4], dtype=np.float32),
                "Layer3": np.frombuffer(row[5], dtype=np.float32),
                "FCLayer": np.frombuffer(row[6], dtype=np.float32),
                "RESNET": np.frombuffer(row[7], dtype=np.float32)
            }
        })


# Load Caltech101 dataset
dataset = Caltech101(root="./data", download=True)

# Extract class names from the Caltech101 dataset
label_name_to_idx = {name: idx for idx, name in enumerate(dataset.categories)}

# Database simulation (in reality, you'd use an actual database)
database = []


def custom_transform(image):
    # Check if the image is grayscale
    if len(image.getbands()) == 1:
        # Convert grayscale to RGB by repeating the channel 3 times
        image = Image.merge("RGB", (image, image, image))
    return image


custom_transforms = transforms.Lambda(lambda x: custom_transform(x))

# Define transformations for the image
transform = transforms.Compose([
    custom_transforms,  # Use the custom transforms here
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def compute_cell_moments(image_np, start_row, end_row, start_col, end_col):
    cell_values = image_np[start_row:end_row, start_col:end_col]
    total_pixels = cell_values.shape[0] * cell_values.shape[1]

    # Compute mean for each channel
    mean = np.sum(cell_values, axis=(0, 1)) / total_pixels

    # Compute variance for each channel
    variance = np.sum((cell_values - mean) ** 2, axis=(0, 1)) / total_pixels

    # Compute standard deviation for each channel
    std_dev = np.sqrt(variance)

    return mean, std_dev, variance


def compute_color_moments_matrix(image_np):
    if len(image_np.shape) == 2:  # Grayscale image
        image_np = np.stack([image_np, image_np, image_np], axis=-1)  # Convert to RGB by repeating the channel 3 times

    height, width, channels = image_np.shape
    cell_height, cell_width = height // 10, width // 10

    mean_matrix = np.zeros((10, 10, channels))
    std_dev_matrix = np.zeros((10, 10, channels))
    variance_matrix = np.zeros((10, 10, channels))

    for i in range(10):
        for j in range(10):
            start_row, end_row = i * cell_height, (i + 1) * cell_height
            start_col, end_col = j * cell_width, (j + 1) * cell_width
            mean, std_dev, variance = compute_cell_moments(image_np, start_row, end_row, start_col, end_col)
            mean_matrix[i, j] = mean
            std_dev_matrix[i, j] = std_dev
            variance_matrix[i, j] = variance

    return mean_matrix, std_dev_matrix, variance_matrix


def extract_features(image):
    # Extract color moments manually
    image_np_resized = np.array(image.resize((100, 100)))
    mean_matrix, std_dev_matrix, variance_matrix = compute_color_moments_matrix(image_np_resized)
    color_moments = np.concatenate([mean_matrix, std_dev_matrix, variance_matrix], axis=2).flatten()
    # print("ColorMoments shape:", color_moments.shape)

    # Convert image to grayscale for HOG
    image_tensor = transform(image)  # Use the main transform here
    gray_image = rgb2gray(image_tensor.permute(1, 2, 0).numpy())

    # Extract HOG features
    hog_features = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(4, 4), visualize=False)
    hog_features = hog_features.flatten()[:900]

    outputs = {}

    def hook(module, input, output):
        outputs[module._get_name()] = output

    # Attach hooks to the desired layers
    hook_handles = []
    hook_handles.append(resnet50.avgpool.register_forward_hook(hook))
    hook_handles.append(resnet50.layer3.register_forward_hook(hook))
    hook_handles.append(resnet50.fc.register_forward_hook(hook))

    # Extract features using RESNET50
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        resnet_output = resnet50(image_tensor)

    # Remove hooks
    for handle in hook_handles:
        handle.remove()

    avgpool_output = outputs['AdaptiveAvgPool2d'].squeeze().numpy()
    avgpool_1024 = (avgpool_output[::2] + avgpool_output[1::2]) / 2

    layer3_output = outputs['Sequential'].squeeze().numpy()
    layer3_output_flattened = layer3_output.reshape(-1)
    stride = len(layer3_output_flattened) // 1024
    layer3_1024 = [np.mean(layer3_output_flattened[i:i + stride]) for i in
                   range(0, len(layer3_output_flattened), stride)]

    fc_1000 = outputs['Linear'].squeeze().numpy()
    resnet = resnet_output.squeeze().numpy()

    return {
        "ColorMoments": color_moments,
        "HOG": hog_features,
        "AvgPool": avgpool_1024,
        "Layer3": layer3_1024,
        "FCLayer": fc_1000,
        "RESNET": resnet
    }


def task_0a():
    for i, (image, label) in enumerate(dataset):
        # Only process even-numbered images
        if i % 2 == 0:
            features = extract_features(image)

            # Store features in the database
            store_in_database(i, features)


def task_0b(imageID_or_file, feature_space, k):
    load_features_from_database()
    # Retrieve the image based on the imageID or file
    query_image = retrieve_image(imageID_or_file)

    # Extract features for the query image
    query_features = extract_features(query_image)[feature_space]

    # Determine if the input is an image ID or a file path
    if isinstance(imageID_or_file, int):
        similar_images = get_most_similar_images(query_features, feature_space, k, query_imageID=imageID_or_file)
    else:
        similar_images = get_most_similar_images(query_features, feature_space, k, query_image=query_image)

    print("\nSimilar Images: ")
    for idx, (imageID, score) in enumerate(similar_images, start=1):
        print(f"{idx}) Image_ID - {imageID}, Score - {score:.2f}")

    # Create a new figure
    plt.figure(figsize=(15, 5))

    # Display the query image
    plt.subplot(1, k + 1, 1)
    plt.imshow(query_image)
    plt.title("Query Image")
    plt.axis('off')

    # Display the similar images
    for idx, (imageID, score) in enumerate(similar_images, start=2):
        image = dataset[imageID][0]  # Retrieve the image using the imageID
        plt.subplot(1, k + 1, idx)
        plt.imshow(image)
        plt.title(f"ID: {imageID}\nScore: {score:.2f}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def get_image_ids_for_label(label_name):
    if label_name not in label_name_to_idx:
        print(f"Label name '{label_name}' not found in the dataset.")
        return []

    label_idx = label_name_to_idx[label_name]
    return [i for i, (img, lbl) in enumerate(dataset) if lbl == label_idx]


def task_1(query_label_idx, feature_space, k):
    load_features_from_database()
    idx_to_label_name = {idx: name for name, idx in label_name_to_idx.items()}

    query_label_name = idx_to_label_name[int(query_label_idx)]

    # Get all image IDs for the given label name
    relevant_image_ids = get_image_ids_for_label(query_label_name)

    # Filter the database entries for the given label
    relevant_entries = [entry for entry in database if entry["imageID"] in relevant_image_ids]

    if not relevant_entries:
        print(f"No images found for label {query_label_name}.")
        return

    # Compute the average feature vector for the given label
    avg_feature_vector = np.mean([entry["features"][feature_space] for entry in relevant_entries], axis=0)

    # Extract the features for each entry and compute their scores
    scores = []
    for entry in relevant_entries:
        score = compare_features(avg_feature_vector, entry["features"][feature_space])
        scores.append((entry["imageID"], score))

    # Sort the scores in descending order and get the top k scores
    top_k_images = sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    print(f"\nMost Relevant Images for Label {query_label_name}:\n")
    for idx, (imageID, score) in enumerate(top_k_images, start=1):
        print(f"{idx}) Image_ID - {imageID}, Score - {score:.2f}")

    plt.figure(figsize=(15, 5))

    # Display the top k images
    for idx, (imageID, score) in enumerate(top_k_images, start=1):
        image = dataset[imageID][0]
        plt.subplot(1, k, idx)
        plt.imshow(image)
        plt.title(f"ID: {imageID}\nScore: {score:.2f}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def task_2a(imageID_or_file, feature_space, k):
    load_features_from_database()
    query_image = retrieve_image(imageID_or_file)
    query_features = extract_features(query_image)[feature_space]
    idx_to_label_name = {idx: name for name, idx in label_name_to_idx.items()}

    label_scores = {}
    for entry in database:
        label_idx = entry["label"]
        label_name = idx_to_label_name[int(label_idx)]
        score = compare_features(query_features, entry["features"][feature_space])
        if label_name in label_scores:
            label_scores[label_name].append(score)
        else:
            label_scores[label_name] = [score]

    # Average the scores for each label
    avg_label_scores = {label: np.mean(scores) for label, scores in label_scores.items()}

    # Get the top k labels
    top_k_labels = sorted(avg_label_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    print("\nTop Matching Labels:")
    for idx, (label, score) in enumerate(top_k_labels, start=1):
        print(f"{idx}) Label - {label}, Score - {score:.2f}")


def extract_resnet50_features(img):
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet50(img_tensor)
    return features.squeeze().numpy()


def task_2b(imageID, k):
    # Extract features for all even-numbered images in the database and compute average features for each label
    label_features = defaultdict(list)
    for i, (img, label) in enumerate(dataset):
        if i % 2 == 0:
            features = extract_resnet50_features(img)
            label_features[label].append(features)

    avg_label_features = {label: np.mean(features, axis=0) for label, features in label_features.items()}

    # Extract features for the query image
    query_img, _ = dataset[imageID]
    query_features = extract_resnet50_features(query_img)

    # Compute similarity scores with average features of each label
    scores = {}
    for label, features in avg_label_features.items():
        similarity_score = np.dot(query_features, features) / (np.linalg.norm(query_features) * np.linalg.norm(features))
        scores[label] = similarity_score

    # Get the top k labels based on similarity scores
    top_k_labels = sorted(scores, key=scores.get, reverse=True)[:k]

    print(f"Top {k} similar labels for imageID {imageID}:")
    for label in top_k_labels:
        print(f"Label: {dataset.categories[label]}, Similarity Score: {scores[label]:.4f}")


def get_top_k_weights(image_weights, k):
    sorted_indices = np.argsort(image_weights)[::-1][:k]
    top_k_weights = image_weights[sorted_indices]
    output = f"ImageID: {sorted_indices[0]}, Weights: {top_k_weights.tolist()}"

    return output


def truncated_svd(A, k):
    U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)

    # Truncate matrices
    U_k = U[:, :k]
    Sigma_k = np.diag(Sigma[:k])
    Vt_k = Vt[:k, :]

    return U_k, Sigma_k, Vt_k


def nnmf(V, k, max_iter=100, tol=1e-4):
    # Initialize W and H with non-negative random values
    W = np.abs(np.random.randn(V.shape[0], k))
    H = np.abs(np.random.randn(k, V.shape[1]))

    for n in range(max_iter):
        # Update H
        WH = np.dot(W, H) + 1e-10  # Add a small value to avoid division by zero
        H *= (np.dot(W.T, V) / np.dot(W.T, WH))

        # Update W
        WH = np.dot(W, H) + 1e-10
        W *= (np.dot(V, H.T) / np.dot(WH, H.T))

        # Check for convergence
        if np.linalg.norm(V - np.dot(W, H)) < tol:
            break

    return W, H


def kmeans_clustering(X, k, max_iters=100, tol=1e-4):
    # Randomly initialize cluster centers
    random_idx = np.random.choice(X.shape[0], k, replace=False)
    cluster_centers = X[random_idx]

    prev_centers = np.zeros(cluster_centers.shape)
    cluster_labels = np.zeros(X.shape[0], dtype=int)

    for _ in range(max_iters):
        # Assign each data point to the nearest cluster center
        for i in range(X.shape[0]):
            distances = np.linalg.norm(X[i] - cluster_centers, axis=1)
            cluster_labels[i] = np.argmin(distances)

        # Update cluster centers
        prev_centers = cluster_centers.copy()
        for j in range(k):
            cluster_points = X[cluster_labels == j]
            if cluster_points.shape[0] > 0:  # Check if cluster has points assigned
                cluster_centers[j] = np.mean(cluster_points, axis=0)

        # Check for convergence
        center_shift = np.linalg.norm(cluster_centers - prev_centers)
        if center_shift < tol:
            break

    return cluster_centers, cluster_labels


def task_3(feature_space, k, reduction_technique):
    load_features_from_database()

    # Extract features for all images in the database
    all_features = [entry["features"][feature_space] for entry in database]
    all_features_matrix = np.array(all_features)

    # Apply the selected dimensionality reduction technique
    if reduction_technique == "SVD":
        svd = TruncatedSVD(n_components=k)
        latent_semantics = svd.fit_transform(all_features_matrix)
    elif reduction_technique == "NNMF":
        min_val = np.min(all_features_matrix)
        if min_val < 0:
            all_features_matrix -= min_val
        nmf = NMF(n_components=k)
        latent_semantics = nmf.fit_transform(all_features_matrix)
    elif reduction_technique == "LDA":
        min_val = np.min(all_features_matrix)
        if min_val < 0:
            all_features_matrix -= min_val
        lda = LatentDirichletAllocation(n_components=k)
        latent_semantics = lda.fit_transform(all_features_matrix)
    elif reduction_technique == "k-means":
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(all_features_matrix)
        cluster_labels = kmeans.predict(all_features_matrix)
        latent_semantics = kmeans.cluster_centers_

    # Store the latent semantics in an output file
    output_filename = f"T3-{feature_space}-{k}-{reduction_technique}.json"
    latent_semantics_data = []

    for i, entry in enumerate(database):
        imageID = entry["imageID"]
        data_entry = {"ImageID": imageID}

        if reduction_technique == "k-means":
            distances_to_centers = [np.linalg.norm(entry["features"][feature_space] - center) for center in
                                    latent_semantics]
            data_entry["Weights"] = distances_to_centers
        else:
            weights = latent_semantics[i]
            formatted_weights = [float(w) for w in weights]
            data_entry["Weights"] = formatted_weights

        latent_semantics_data.append(data_entry)

    # print(latent_semantics_data)
    max_latent = []
    for i in range(k):
        max = 0
        for j in latent_semantics_data:
            if j["Weights"][i] > max:
                max = j["Weights"][i]
                image = j["ImageID"]
        max_latent.append([image, max])

    for i in range(k):
        print(f"Latent Semantic {i + 1} - ImageID - {max_latent[i][0]} - {max_latent[i][1]}")

    def default_serialize(o):
        if isinstance(o, np.float32):
            return float(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(output_filename, 'w') as f:
        json.dump(latent_semantics_data, f, indent=4, default=default_serialize)

    print(f"Latent semantics stored in {output_filename}")


def normalize_factors(factors):
    normalized_factors = []
    for factor in factors:
        # If the factor is a list of matrices
        if isinstance(factor, list):
            normalized_subfactors = []
            for subfactor in factor:
                norms = np.linalg.norm(subfactor, axis=0, keepdims=True)
                normalized_subfactor = subfactor / norms
                normalized_subfactors.append(normalized_subfactor)
            normalized_factors.append(normalized_subfactors)
        # If the factor is a single matrix
        else:
            norms = np.linalg.norm(factor, axis=0, keepdims=True)
            normalized_factor = factor / norms
            normalized_factors.append(normalized_factor)
    return normalized_factors


def extract_latent_semantics(tensor, k):
    weights, factors = parafac(tensor, rank=k, n_iter_max=100, normalize_factors=True)
    return weights, factors[2]  # Third mode is 'label'


def structure_output(weights, label_factors, label_dict):
    structured_output = []

    for i in range(label_factors.shape[1]):
        label_weights = weights[i] * label_factors[:, i]  # Multiply the weights with the label factors
        sorted_indices = np.argsort(label_weights)[::-1]
        sorted_labels = [label_dict[idx] for idx in sorted_indices]
        sorted_weights = label_weights[sorted_indices]
        label_dict_output = {label: weight for label, weight in zip(sorted_labels, sorted_weights)}

        structured_output.append({
            "LatentSemantic": i + 1,
            "Labels": label_dict_output
        })

    return structured_output


def task_4(feature_space, k):
    load_features_from_database()
    # Create a tensor of shape (number of images, number of features, number of labels)
    num_images = len(database)
    num_features = len(database[0]["features"][feature_space])
    num_labels = len(label_name_to_idx)

    tensor = np.zeros((num_images, num_features, num_labels))

    for idx, entry in enumerate(database):
        label_idx = int(entry["label"])
        tensor[idx, :, label_idx] = entry["features"][feature_space]

    label_dict = {idx: name for name, idx in label_name_to_idx.items()}

    # Perform CP decomposition
    weights, label_factors = extract_latent_semantics(tensor, k)
    structured_results = structure_output(weights, label_factors, label_dict)

    print(structured_results)

    # Store the latent semantics in an output file
    output_filename = f"T4-{feature_space}-{k}.json"
    # Save to JSON
    with open(output_filename, 'w') as f:
        json.dump(structured_results, f, indent=4)

    print(f"Latent semantics stored in {output_filename}")


def task_5(feature_space, k, reduction_technique):
    load_features_from_database()

    # Create a label-label similarity matrix
    num_labels = len(label_name_to_idx)
    label_similarity_matrix = np.zeros((num_labels, num_labels))

    for i in range(num_labels):
        for j in range(num_labels):
            label_i_images = [entry["features"][feature_space] for entry in database if int(entry["label"]) == i]
            label_j_images = [entry["features"][feature_space] for entry in database if int(entry["label"]) == j]

            avg_label_i = np.mean(label_i_images, axis=0)
            avg_label_j = np.mean(label_j_images, axis=0)

            similarity = compare_features(avg_label_i, avg_label_j)
            label_similarity_matrix[i, j] = similarity
    np.savetxt(f"label__{feature_space}_{reduction_technique}_similarity_matrix.txt", label_similarity_matrix,
               fmt="%.4f")

    # Shift the matrix values to be non-negative before applying LDA
    if reduction_technique == "LDA":
        min_val = np.min(label_similarity_matrix)
        if min_val < 0:
            label_similarity_matrix -= min_val

    # Apply the selected dimensionality reduction technique
    if reduction_technique == "SVD":
        svd = TruncatedSVD(n_components=k)
        latent_semantics = svd.fit_transform(label_similarity_matrix)
    elif reduction_technique == "NNMF":
        min_val = np.min(label_similarity_matrix)
        if min_val < 0:
            label_similarity_matrix -= min_val
        nmf = NMF(n_components=k)
        latent_semantics = nmf.fit_transform(label_similarity_matrix)
    elif reduction_technique == "LDA":
        min_val = np.min(label_similarity_matrix)
        if min_val < 0:
            label_similarity_matrix -= min_val
        lda = LatentDirichletAllocation(n_components=k)
        latent_semantics = lda.fit_transform(label_similarity_matrix)
    elif reduction_technique == "k-means":
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(label_similarity_matrix)
        cluster_labels = kmeans.predict(label_similarity_matrix)
        latent_semantics = kmeans.cluster_centers_

    # Store the latent semantics in an output file
    output_filename = f"T5-{feature_space}-{k}-{reduction_technique}.json"
    latent_semantics_data = []

    for i in range(k):
        data_entry = {"LatentSemantic": i + 1, "Labels": {}}
        if reduction_technique == "k-means":
            weights = latent_semantics[i]
        else:
            weights = latent_semantics[:, i]
        sorted_indices = np.argsort(weights)[::-1]
        print(f"\nLatent Semantic {i+1}:")
        for idx in sorted_indices:
            label_name = list(label_name_to_idx.keys())[list(label_name_to_idx.values()).index(idx)]
            data_entry["Labels"][label_name] = float(weights[idx])
            print(f"{label_name} : {weights[idx]}")
        latent_semantics_data.append(data_entry)

    def default_serialize(o):
        if isinstance(o, np.float32):
            return float(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(output_filename, 'w') as f:
        json.dump(latent_semantics_data, f, indent=4, default=default_serialize)

    print(f"Latent semantics stored in {output_filename}")


def task_6(feature_space, k, reduction_technique):
    load_features_from_database()

    # Create an image-image similarity matrix
    num_images = len(database)
    image_similarity_matrix = np.zeros((num_images, num_images))

    for i in range(num_images):
        for j in range(num_images):
            similarity = compare_features(database[i]["features"][feature_space],
                                          database[j]["features"][feature_space])
            image_similarity_matrix[i, j] = similarity

    # Save the image-image similarity matrix
    np.savetxt(f"image_{feature_space}_{reduction_technique}_similarity_matrix.txt", image_similarity_matrix,
               fmt="%.4f")

    # Shift the matrix values to be non-negative before applying LDA
    if reduction_technique == "LDA":
        min_val = np.min(image_similarity_matrix)
        if min_val < 0:
            image_similarity_matrix -= min_val

    # Apply the selected dimensionality reduction technique
    if reduction_technique == "SVD":
        svd = TruncatedSVD(n_components=k)
        latent_semantics = svd.fit_transform(image_similarity_matrix)
    elif reduction_technique == "NNMF":
        min_val = np.min(image_similarity_matrix)
        if min_val < 0:
            image_similarity_matrix -= min_val
        nmf = NMF(n_components=k)
        latent_semantics = nmf.fit_transform(image_similarity_matrix)
    elif reduction_technique == "LDA":
        min_val = np.min(image_similarity_matrix)
        if min_val < 0:
            image_similarity_matrix -= min_val
        lda = LatentDirichletAllocation(n_components=k)
        latent_semantics = lda.fit_transform(image_similarity_matrix)
    elif reduction_technique == "k-means":
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(image_similarity_matrix)
        cluster_labels = kmeans.predict(image_similarity_matrix)
        latent_semantics = kmeans.cluster_centers_

    # Store the latent semantics in an output file
    output_filename = f"T6-{feature_space}-{k}-{reduction_technique}.json"
    latent_semantics_data = []

    for latent_idx in range(k):
        # Collect ImageID-Weight pairs for the current latent semantic
        image_weight_pairs = []
        for i, entry in enumerate(database):
            imageID = entry["imageID"]
            if reduction_technique == "k-means":
                distances_to_centers = [np.linalg.norm(image_similarity_matrix[i] - center) for center in
                                        latent_semantics]
                weight = distances_to_centers[latent_idx]
            else:
                weight = latent_semantics[i][latent_idx]

            image_weight_pairs.append({"ImageID": imageID, "Weight": float(weight)})

        # Sort the pairs in decreasing order of weights
        sorted_image_weight_pairs = sorted(image_weight_pairs, key=lambda x: x["Weight"], reverse=True)

        # Print out the sorted pairs for the current latent semantic
        print(f"\nLatent Semantic {latent_idx + 1}:")
        for pair in sorted_image_weight_pairs:
            print(f"ImageID: {pair['ImageID']}, Weight: {pair['Weight']}")

        # Add the sorted pairs to the latent_semantics_data for saving to a file
        latent_semantics_data.append(sorted_image_weight_pairs)

    def default_serialize(o):
        if isinstance(o, np.float32):
            return float(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(output_filename, 'w') as f:
        json.dump(latent_semantics_data, f, indent=4, default=default_serialize)

    print(f"Latent semantics stored in {output_filename}")


def task_7(imageID, latent_semantics_file, k):
    if "T3" in latent_semantics_file or "T6" in latent_semantics_file:
        load_features_from_database()

        # Load the latent semantics from the provided file
        with open(f"{latent_semantics_file}.json", 'r') as f:
            latent_semantics_data = json.load(f)
        latent_semantics_dict = {int(entry["ImageID"]): entry["Weights"] for entry in latent_semantics_data}

        # Check if the imageID is present in the latent semantics dictionary
        if imageID not in latent_semantics_dict:
            print(f"Latent semantics for ImageID {imageID} not found in the provided file!")
            return

        # Retrieve the latent semantics for the given imageID
        query_latent_semantics = np.array(latent_semantics_dict[imageID])

        # Compute similarities between the query latent semantics and the latent semantics of all images
        similarities = []
        for other_image_id, weights in latent_semantics_dict.items():
            # Skip the query imageID
            if other_image_id == imageID:
                continue
            # print(query_latent_semantics, np.array(weights))
            similarity = compare_features(query_latent_semantics, np.array(weights))
            # print(similarity)
            similarities.append((other_image_id, similarity))

        # Sort images based on similarity
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

        print("\nSimilar Images under the selected latent space: ")
        for idx, (matched_imageID, score) in enumerate(sorted_similarities):
            print(f"{idx}) Image_ID - {matched_imageID}, Score - {score:.4f}")

    if "T4" in latent_semantics_file or "T5" in latent_semantics_file:
        load_features_from_database()
        with open(f"T3-RESNET-5-k-means.json", 'r') as f:
            latent_semantics_data = json.load(f)
        latent_semantics_dict = {int(entry["ImageID"]): entry["Weights"] for entry in latent_semantics_data}

        # Check if the imageID is present in the latent semantics dictionary
        if imageID not in latent_semantics_dict:
            print(f"Latent semantics for ImageID {imageID} not found in the provided file!")
            return

        # Retrieve the latent semantics for the given imageID
        query_latent_semantics = np.array(latent_semantics_dict[imageID])

        # Compute similarities between the query latent semantics and the latent semantics of all images
        similarities = []
        for other_image_id, weights in latent_semantics_dict.items():
            # Skip the query imageID
            if other_image_id == imageID:
                continue
            # print(query_latent_semantics, np.array(weights))
            similarity = compare_features(query_latent_semantics, np.array(weights))
            # print(similarity)
            similarities.append((other_image_id, similarity))

        # Sort images based on similarity
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

        print("\nSimilar Images under the selected latent space: ")
        for idx, (matched_imageID, score) in enumerate(sorted_similarities):
            print(f"{idx}) Image_ID - {matched_imageID}, Score - {score:}")

    # Create a new figure
    plt.figure(figsize=(15, 5))

    # Display the query image
    plt.subplot(1, k + 1, 1)  # +1 for the query image
    plt.imshow(dataset[imageID][0])
    plt.title("Query Image")
    plt.axis('off')

    # Display the similar images
    for idx, (matched_imageID, score) in enumerate(sorted_similarities, start=2):
        image = dataset[matched_imageID][0]  # Retrieve the image using the matched_imageID
        plt.subplot(1, k + 1, idx)
        plt.imshow(image)
        plt.title(f"ID: {matched_imageID}\nScore: {score:.2f}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def get_label_for_image(image_id):
    for entry in database:
        if entry["imageID"] == image_id:
            label_idx = entry.get("label", None)
    for label, idx in label_name_to_idx.items():
        if int(idx) == int(label_idx):
            return label
    return None


def task_8(imageID, latent_semantics_file, k):
    load_features_from_database()
    image_label = get_label_for_image(imageID)

    # Load the latent semantics from the provided JSON file
    with open(f"{latent_semantics_file}.json", 'r') as f:
        latent_semantics_data = json.load(f)

    # Extract the latent semantic corresponding to the given image label
    image_latent_semantic = None

    # Compute similarities between the image label latent semantic and the latent semantics of all labels
    similarities = []
    if "T5" in latent_semantics_file:
        for entry in latent_semantics_data:
            if entry["Labels"].get(image_label):
                image_latent_semantic = entry["Labels"][image_label]
                break

        for entry in latent_semantics_data:
            for label, score in entry["Labels"].items():
                similarity = abs(image_latent_semantic - score)
                similarities.append((label, similarity))

        # Sort labels based on similarity
        sorted_similarities = sorted(similarities, key=lambda x: x[1])[:k]

        print("\nTop k matching labels under the selected latent space: ")
        for label, score in sorted_similarities:
            print(f"Label: {label}, Score: {score:.4f}")

    elif ("T6" in latent_semantics_file) or ("T3" in latent_semantics_file):
        for entry in latent_semantics_data:
            if entry["ImageID"] == imageID:
                target_weights = entry["Weights"]
                break

        weight_differences = {}
        for image in latent_semantics_data:
            if image["ImageID"] != imageID:
                differences = [abs(a - b) for a, b in zip(target_weights, image["Weights"])]
                mean_difference = sum(differences) / len(differences)  # Calculate the mean of the differences
                weight_differences[image["ImageID"]] = mean_difference

        sorted_images = sorted(weight_differences.items(), key=lambda x: x[1])
        extended_k = k * 20
        top_k_images = [{"ImageID": image[0], "Score": image[1]} for image in sorted_images[:extended_k]]

        # Get labels for the top k images
        labels = [get_label_for_image(image["ImageID"]) for image in top_k_images]
        scores = [image["Score"] for image in top_k_images]
        scores.sort()

        print("\n")
        # Sort labels based on the image's score and select the top 5 unique labels
        sorted_labels = sorted(set(labels), key=lambda x: labels.count(x), reverse=True)[:k]
        for i in range(len(sorted_labels)):
            print(f"Label: {sorted_labels[i]}, Score: {scores[i+1]:.7f}")


def task_9(label_id, latent_semantics_file, k):
    # Load the latent semantics from the provided file
    if 'T3' in latent_semantics_file or 'T6' in latent_semantics_file:
        idx_to_label_name = {idx: name for name, idx in label_name_to_idx.items()}
        label = idx_to_label_name[int(label_id)]
        with open("T5-RESNET-5-k-means.json", 'r') as f:
            latent_semantics_data = json.load(f)

        # Extract the weights for the given label
        for entry in latent_semantics_data:
            label_weights = entry["Labels"].get(label)
            break

        # Calculate the difference between the given label's weights and all other labels' weights
        differences = {}
        for other_label, other_weights in entry["Labels"].items():
            diff = abs(label_weights - other_weights)
            differences[other_label] = diff

        # Sort labels based on the difference
        sorted_differences = sorted(differences.items(), key=lambda x: x[1])

        # Display the top k similar labels
        print(f"\nTop {k} similar labels to {label} under the selected latent space:")
        for i, (label_name, diff) in enumerate(sorted_differences[:k], start=1):
            print(f"{i}. {label_name} - Score: {diff:.4f}")

    elif "T4" in latent_semantics_file or "T5" in latent_semantics_file:
        idx_to_label_name = {idx: name for name, idx in label_name_to_idx.items()}
        label = idx_to_label_name[int(label_id)]

        with open(f"{latent_semantics_file}.json", 'r') as f:
            latent_semantics_data = json.load(f)

        # Extract the weights for the given label
        for entry in latent_semantics_data:
            label_weights = entry["Labels"].get(label)
            break

        # Calculate the difference between the given label's weights and all other labels' weights
        differences = {}
        for other_label, other_weights in entry["Labels"].items():
            diff = abs(label_weights - other_weights)
            differences[other_label] = diff

        # Sort labels based on the difference
        sorted_differences = sorted(differences.items(), key=lambda x: x[1])

        # Display the top k similar labels
        print(f"\nTop {k} similar labels to {label} under the selected latent space:")
        for i, (label_name, diff) in enumerate(sorted_differences[:k], start=1):
            print(f"{i}. {label_name} - Score: {diff:.4f}")


def task_10(label_id, latent_semantics_file, k):
    # Load the latent semantics from the provided file
    idx_to_label_name = {idx: name for name, idx in label_name_to_idx.items()}
    label = idx_to_label_name[int(label_id)]

    if "T4" in latent_semantics_file or "T5" in latent_semantics_file:
        with open(f"{latent_semantics_file}.json", 'r') as f:
            latent_semantics_data = json.load(f)

        # Extract the weights for the given label
        for entry in latent_semantics_data:
            label_weights = entry["Labels"].get(label)
            break

        for name, idx in label_name_to_idx.items():
            if name == label:
                label_idx = idx

        # Get the image IDs for the given label from the database
        cursor.execute("SELECT imageID FROM features WHERE label=?", (label_idx,))
        image_ids_for_label = [row[0] for row in cursor.fetchall()]

        # Calculate the difference between the given label's weights and all images' weights
        differences = {}
        for image_id in image_ids_for_label:
            cursor.execute("SELECT FCLayer FROM features WHERE imageID=?", (image_id,))
            image_weights = np.frombuffer(cursor.fetchone()[0], dtype=np.float32)
            diff = np.linalg.norm(label_weights - image_weights)
            differences[image_id] = diff

        # Sort images based on the difference
        sorted_differences = sorted(differences.items(), key=lambda x: x[1], reverse= True)[:k]
    elif "T3" or "T6" in latent_semantics_file:
        with open("T5-RESNET-5-SVD.json", 'r') as f:
            latent_semantics_data = json.load(f)

        # Extract the weights for the given label
        for entry in latent_semantics_data:
            label_weights = entry["Labels"].get(label)
            break

        for name, idx in label_name_to_idx.items():
            if name == label:
                label_idx = idx

        # Get the image IDs for the given label from the database
        cursor.execute("SELECT imageID FROM features WHERE label=?", (label_idx,))
        image_ids_for_label = [row[0] for row in cursor.fetchall()]

        # Calculate the difference between the given label's weights and all images' weights
        differences = {}
        for image_id in image_ids_for_label:
            cursor.execute("SELECT FCLayer FROM features WHERE imageID=?", (image_id,))
            image_weights = np.frombuffer(cursor.fetchone()[0], dtype=np.float32)
            diff = np.linalg.norm(label_weights - image_weights)
            differences[image_id] = diff

        # Sort images based on the difference
        sorted_differences = sorted(differences.items(), key=lambda x: x[1], reverse=True)[:k]

    print("\nSimilar Images under the selected latent space: ")
    for idx, (matched_imageID, score) in enumerate(sorted_differences):
        print(f"{idx}) Image_ID - {matched_imageID}, Score - {score:.4f}")

    plt.figure(figsize=(15, 5))

    for idx, (matched_imageID, score) in enumerate(sorted_differences, start=1):
        # Assuming you have a function to get the image path by its ID
        image = dataset[matched_imageID][0]
        plt.subplot(1, k, idx)
        plt.imshow(image)
        plt.title(f"ID: {matched_imageID}\nScore: {score:.4f}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Longchao Taks11:

def descriptor_HOG(image_id):
    img = imread(image_id)
    image_gray = rgb2gray(img)
    resized_img = resize(image_gray, (300, 100))

    #  channel_axis: If None, the image is assumed to be a grayscale (single channel) image.
    #         Otherwise, this parameter indicates which axis of the array corresponds
    #         to channels.
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(26, 9),
                        cells_per_block=(2, 2), visualize=True, channel_axis=None, feature_vector=False)

    fd_lower = np.transpose(fd, (0, 1, 4, 2, 3))
    fd_lower = np.mean(fd_lower, axis=(3, 4))

    # print(fd_lower.shape)
    return fd_lower


def descriptor_moments(image_id):
    img = imread(image_id)
    resized_img = resize(img, (300, 100))
    ## partition the image into 10*10

    # Define the number of cells in the grid
    num_cells_x = 10
    num_cells_y = 10

    # Compute the dimensions of each cell
    cell_height = resized_img.shape[0] // num_cells_y
    cell_width = resized_img.shape[1] // num_cells_x

    # Create a copy of the resized image for visualization
    visualized_img = resized_img.copy()

    # Draw grid lines on the image
    for i in range(1, num_cells_x):
        x = i * cell_width
        visualized_img[:, x, :] = 0  # Set vertical lines to black

    for j in range(1, num_cells_y):
        y = j * cell_height
        visualized_img[y, :, :] = 0  # Set horizontal lines to black

    y, x, d = visualized_img.shape

    x_len = 10
    y_len = 10

    y_block = y // y_len
    x_block = x // x_len

    sub_figs = []
    for j in range(0, y-1, y_block):
        for i in range(0, x-1, x_block):
            x_start = i
            x_end = i+x_block

            y_start = j
            y_end = j+y_block

            sub_figs.append(resized_img[y_start:y_end,x_start:x_end,])


    # print(sub_figs.__len__())

    read_group = [[] for i in range (3)] #[[], [], []]: mean, std, stew
    blue_group = [[] for i in range (3)]
    green_group = [[] for i in range (3)]

    # print(value_pack[0].shape)
    for item in sub_figs: # (30, 10, 3)
        r_group = item[:, :, 0].reshape(-1) # (30, 10)
        g_group = item[:, :, 1].reshape(-1) # (30, 10)
        b_group = item[:, :, 2].reshape(-1) # (30, 10)


        for i in range(3):
            if i == 0 :
                read_group[i].append(np.mean(r_group))
                blue_group[i].append(np.mean(g_group))
                green_group[i].append(np.mean(b_group))
            if i == 1 :
                read_group[i].append(np.std(r_group))
                blue_group[i].append(np.std(g_group))
                green_group[i].append(np.std(b_group))
            if i == 2 :
                read_group[i].append(skew(r_group, axis=0, bias=True))
                blue_group[i].append(skew(r_group, axis=0, bias=True))
                green_group[i].append(skew(r_group, axis=0, bias=True))
        # print(r_group.shape)

    # print(np.array(read_group).shape)

    descriptor = []
    descriptor.append(np.reshape((r_group), (10, 10, -1)))
    descriptor.append(np.reshape((g_group), (10, 10, -1)))
    descriptor.append(np.reshape((b_group), (10, 10, -1)))

    return descriptor

def Euclidean_Distance(query, candidate):
    var = query - candidate
    distances = np.sqrt(np.sum(var ** 2))

    # mean_again = np.mean(distances, axis=(0, 1))
    return distances

def Euclidean_Distance_for_moments(query, candidate):
    var = query - candidate
    distances = np.sqrt(np.sum(var ** 2, axis=(-1, -2)))

    mean_again = np.mean(distances, axis=(0, 1))
    return mean_again



def task_11(label_id = 0, m = 5, n = 10, in_mode = "T3-CM-5-SVD", use_current_graph = True):
    mode = in_mode # HOGs, moments, avg_pool, FC, layer3, T3-CM-5-SVD
    # ******************************* formal query input **************************************
    value_n = n # n most similar images in the database in the given space
    value_m = m # m most significant images
    # *****************************************************************************************
    # Self Judge first, then follow the rules:
    if use_current_graph and (label_id in [0, 20, 55, 100]) and m == 5 and n == 10 and (in_mode in ["T3-CM-5-SVD", "FC"]):
        try:
            if mode == "T3-CM-5-SVD":
                if label_id == 0:
                    exist_graph_path = "./graph_files/T3-CM-5-SVD-Simi-Graph-0.gexf" 
                elif label_id == 20:
                    exist_graph_path = "./graph_files/T3-CM-5-SVD-Simi-Graph-20.gexf"
                elif label_id == 55:
                    exist_graph_path = "./graph_files/T3-CM-5-SVD-Simi-Graph-55.gexf"
                elif label_id == 100:
                    exist_graph_path = "./graph_files/T3-CM-5-SVD-Simi-Graph-100.gexf"
            
            if mode == "FC":
                if label_id == 0:
                    exist_graph_path = "./graph_files/FC-Simi-Graph-0.gexf" 
                elif label_id == 20:
                    exist_graph_path = "./graph_files/FC-Simi-Graph-20.gexf"
                elif label_id == 55:
                    exist_graph_path = "./graph_files/FC-Simi-Graph-55.gexf"
                elif label_id == 100:
                    exist_graph_path = "./graph_files/FC-Simi-Graph-100.gexf"
        except Exception as e:
            print("Existing graph files not found, constructing new one...")
            use_current_graph = False
    else:
        use_current_graph = False

    save_graph_name = "./graph_files/"+ mode + "-Simi"+ "-Graph-"+str(label_id)+".gexf"
    # FC / T3-CM-5-SVD

    if mode in ["HOGs", "moments"]:

        image_id = str(label_id)
        mapping = {"0":"8394", "880":"4402", "2500":"1134", "5122":"", "8676": "3912"}
        base_path = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/origin/"
        label = mapping[image_id] # query label
        input_path = base_path+label+".png"
        img = imread(input_path)

    else:

        image_id = label_id # 100, 55, 20, 0
        
        download_dir = "./data"
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the images to 224x224 pixels (adjust as needed)
            transforms.ToTensor(),          # Convert images to PyTorch tensors
        ])

        caltech101_dataset = torchvision.datasets.Caltech101(
            root=download_dir,
            download=True,        # Set to True to download the dataset
            transform=transform   # Apply the defined transform to the images
        )

        data_loader = DataLoader(
            dataset=caltech101_dataset,
            batch_size=4,
            shuffle=True,
            num_workers = 8
        )
        selected_image, label = caltech101_dataset[image_id]

        print(selected_image.shape)
        img = np.transpose(selected_image, (1, 2, 0))

    # Show the image that is being requested:

    plt.axis("off")
    plt.imshow(img)
    print(img.shape)
    plt.show()
    
    if use_current_graph:
        G = nx.read_gexf(exist_graph_path)
    else:
        # Create a graph.
        G = nx.Graph()
        # target_description = descriptor_fun(input_path)
        support_latent_list = ["T3-CM-5-SVD", "T3-HOG-5-LDA", "T3-HOG-5-SVD" , "T3-CM-10-LDA", "T3-CM-10-SVD",
                     "T3-HOG-10-LDA", "T3-HOG-1-LDA", "T3-HOG-1-SVD"]
        
        if mode == "HOGs":
            data = torch.load("/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/rgb_data_HOGs.pt")
            descriptor_fun = descriptor_HOG
            
        elif mode == "moments":
            data = torch.load("/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/rgb_data_moments.pt")[-500:]
            descriptor_fun = descriptor_moments

        elif mode == "avg_pool":
            data = torch.load("/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/rgb_data_avgpool_1024.pt")

        elif mode == "layer3":
            data = torch.load("/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/rgb_data_layer3_1024.pt")

        elif mode == "FC":
            # ('0', tensor([[6.1887e+00,...937e-04]]))
            data = torch.load("/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/rgb_data_fclayer_1000.pt")[0:500]
            image_data = data[image_id][1]
            # descriptor_fun = descriptor_moments
        
        elif mode in support_latent_list:
            
            file_name = "./"+mode+".csv"
            csv_data = pd.read_csv(file_name, sep=",", header=0)
            length = csv_data.shape[0]
            # transfer from origin structure to unified level:
            data = []
            for i in range(length):
                data.append((str(i), csv_data.iloc[i].values))
            data = data[:500]
            image_data = data[image_id][1]
        
        
        print("Starting constructing the graph, please wait...")
        for node_id, descrip in tqdm(data):
            # add note
            G.add_node(node_id)
            # renew list for every new node
            evaluate_list = []
            if mode in ["HOGs", "moments"] :
            # current descriptor from node
                node_now = descriptor_fun(base_path+str(node_id+".png"))

            elif mode =="moments":
                node_now = np.transpose(node_now, (1, 2, 3, 0))
            
            elif mode == "FC":
                # the data set directly contains the FC output (1, 1000), so we take [0] ==> (1000,)
                node_now = descrip.detach().numpy()[0]

            elif mode in support_latent_list:
                node_now = descrip[0]

            for id_now, description in data:
                if mode == "HOGs":
                    evaluate_list.append((id_now, Euclidean_Distance(node_now, description)))
                elif mode == "moments":
                    description_new = np.transpose(description, (1, 2, 3, 0))
                    evaluate_list.append((id_now, Euclidean_Distance_for_moments(node_now, description_new)))
                elif mode == "FC":
                    # evaluate_list.append((id, Euclidean_Distance(image_data.detach().numpy(), description_new.detach().numpy())))
                    evaluate_list.append((id_now, np.corrcoef(image_data.detach().numpy()[0], description.detach().numpy()[0])[0, 1]))
                elif mode in support_latent_list:
                    evaluate_list.append((id_now, np.corrcoef(image_data, description)[0, 1]))

            if mode == "HOGs": # distance = Euclidean distance
                sorted_data = sorted(evaluate_list, key=lambda x:x[1])
            else: # mode in ["FC"] + support_latent_list
                sorted_data = sorted(evaluate_list, key=lambda x:x[1], reverse=True)
            
            selected_result = sorted_data[:value_n]
            # print(selected_result)
            # the id is the data[0]
            for i in range(1, len(selected_result)): # ignore the first
                # add edge:
                G.add_edge(node_id, selected_result[i][0], weight=selected_result[i][1])
            
        nx.write_gexf(G, save_graph_name)
    
    # alpha (damping factor) is set to 0.85, 
    # means that there is a 15% chance that the random walker will teleport to any node in the graph with equal probability
    # and an 85% chance that the walker will follow outgoing links to other nodes in the graph.

    # pagerank_scores = nx.pagerank(G, personalization=personalized_teleport_vector)
    pagerank_scores = nx.pagerank(G)

    sorted_images = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True) # rank pageranke score by larege-smaller

    key_list = []

    for i in range(1, value_m+1):
        id, pagescore = sorted_images[i+1]
        if not isinstance(id, types.BuiltinFunctionType):
            key_list.append(str(id))
            print(sorted_images[i]) # 0(itself), so we start with 12345

    print("finished!")


    plt.figure(figsize=(10, 3))

    counter = 1

    if not value_m % 2 == 0:
        for j in range(value_m):
            plt.subplot(1, value_m, counter) # 1 row
            if mode in ["HOGs" or "moments"] :
                    plt.imshow(imread(base_path+str(key_list[counter-1]+".png")))
            elif mode in support_latent_list:
                selected_image, selected_label = caltech101_dataset[int(key_list[counter-1])]
                plt.imshow(np.transpose(selected_image, (1, 2, 0)))
            counter += 1 
            
            plt.axis('off')
            plt.tight_layout()
    else:
        for i in range(2):
            for j in range(int(value_m/2)):
                plt.subplot(2, int(value_m/2), counter)
                print(counter)
                if mode in ["HOGs" or "moments"] :
                    plt.imshow(imread(base_path+str(key_list[counter-1]+".png")))
                elif mode in support_latent_list:
                    selected_image, selected_label = caltech101_dataset[int(key_list[counter-1])]
                    plt.imshow(np.transpose(selected_image, (1, 2, 0)))

                counter += 1 
                plt.axis('off')
        plt.tight_layout()

    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if not os.path.exists("./data/output/"):
        os.makedirs("./data/output/")
    plt.savefig("./data/output/mode"+str(mode)+"_"+"l"+str(image_id)+"_"+"n"+str(value_n-1)+"_"+"m"+str(value_m)+current_time+".png")
    # Show the entire figure with subplots
    plt.show()

def resnetfc(img):
    #dictionary storing layer outputs
    layer_outputs = {
        'layer3': [],
        'avgpool': [],
        'fc': []
    }
    def get_intermediate_outputs(layer):
        def hook(model, input, output):
            layer_outputs[layer] = output.detach()
        return hook
    #resize image
    imOpen = np.asarray(img)
    resizedImg = resize(imOpen, (224,224))
    npDataset = torch.tensor([np.transpose(resizedImg)])
    #print("Shape:" + str(npDataset.shape))
    rn_model =  torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    rn_model.eval()
    rn_model.fc.register_forward_hook(get_intermediate_outputs('fc'))
    rn_model(npDataset.float())

    return layer_outputs['fc']


def task_11_extra(label_src = "", m = 5, n = 10, in_mode = "T3-CM-5-SVD", use_current_graph = False):
    # label_src = "./data/extraImg/Dwayne_Johnson.jpeg"
    mode = in_mode # HOGs, moments, avg_pool, FC, layer3, T3-CM-5-SVD
    # ******************************* formal query input **************************************
    value_n = n # n most similar images in the database in the given space
    value_m = m # m most significant images
    # *****************************************************************************************
    save_graph_name = mode + "-Simi"+ "-Graph-"+str(label_src.split("/")[-1].replace(".", ""))+".gexf"
    # read in the current file
    img = imread(label_src)
    # Show the image that is being requested:

    plt.axis("off")
    plt.imshow(img)
    print(img.shape)
    plt.show()
    # Create a graph.
    G = nx.Graph()
    
    if mode == "HOGs":
        pass
    elif mode == "moments":
        pass
    elif mode == "avg_pool":
        pass
    elif mode == "T3-CM-5-SVD":
        pass
    elif mode == "layer3":
        pass
    elif mode == "FC":
        # ('0', tensor([[6.1887e+00,...937e-04]]))
        data = torch.load("/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/rgb_data_fclayer_1000.pt")[0:500]
        image_data = resnetfc(img)
        data.insert(0, ("new", image_data))
        # descriptor_fun = descriptor_moments

    print("Starting constructing the graph, please wait...")
    for node_id, descrip in tqdm(data):
        # add note
        G.add_node(node_id)
        # renew list for every new node
        evaluate_list = []
        if mode in ["HOGs", "moments"] :
            pass
        # current descriptor from node
            # node_now = descriptor_fun("base_path"+str(node_id+".png"))

        elif mode =="moments":
            node_now = np.transpose(node_now, (1, 2, 3, 0))
        
        elif mode == "FC":
            # the data set directly contains the FC output (1, 1000), so we take [0] ==> (1000,)
            node_now = descrip.detach().numpy()[0]

        elif mode == "T3-CM-5-SVD":
            node_now = descrip[0]

        for id_now, description in data:
            if mode == "HOGs":
                evaluate_list.append((id_now, Euclidean_Distance(node_now, description)))
            elif mode == "moments":
                description_new = np.transpose(description, (1, 2, 3, 0))
                evaluate_list.append((id_now, Euclidean_Distance_for_moments(node_now, description_new)))
            elif mode == "FC":
                # evaluate_list.append((id, Euclidean_Distance(image_data.detach().numpy(), description_new.detach().numpy())))
                evaluate_list.append((id_now, np.corrcoef(image_data.detach().numpy()[0], description.detach().numpy()[0])[0, 1]))
            elif mode == "T3-CM-5-SVD":
                evaluate_list.append((id_now, np.corrcoef(image_data, description)[0, 1]))

        if mode == "HOGs": # distance = Euclidean distance
            sorted_data = sorted(evaluate_list, key=lambda x:x[1])
        elif mode in ["FC", "T3-CM-5-SVD"]:
            sorted_data = sorted(evaluate_list, key=lambda x:x[1], reverse=True)
        
        selected_result = sorted_data[:value_n]
        # print(selected_result)
        # the id is the data[0]
        for i in range(1, len(selected_result)): # ignore the first
            # add edge:
            G.add_edge(node_id, selected_result[i][0], weight=selected_result[i][1])
        
    nx.write_gexf(G, save_graph_name)
    
    # alpha (damping factor) is set to 0.85, 
    # means that there is a 15% chance that the random walker will teleport to any node in the graph with equal probability
    # and an 85% chance that the walker will follow outgoing links to other nodes in the graph.

    # pagerank_scores = nx.pagerank(G, personalization=personalized_teleport_vector)
    pagerank_scores = nx.pagerank(G)

    sorted_images = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True) # rank pageranke score by larege-smaller

    key_list = []

    for i in range(1, value_m+1):
        id, pagescore = sorted_images[i+1]
        if not isinstance(id, types.BuiltinFunctionType):
            key_list.append(str(id))
            print(sorted_images[i]) # 0(itself), so we start with 12345

    print("finished!")

    plt.figure(figsize=(10, 3))

    download_dir = "./data"
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the images to 224x224 pixels (adjust as needed)
        transforms.ToTensor(),          # Convert images to PyTorch tensors
    ])

    caltech101_dataset = torchvision.datasets.Caltech101(
        root=download_dir,
        download=True,        # Set to True to download the dataset
        transform=transform   # Apply the defined transform to the images
    )

    counter = 1

    if not value_m % 2 == 0:
        for j in range(value_m):
            plt.subplot(1, value_m, counter) # 1 row
            if mode in ["HOGs" or "moments"] :
                print("Please choose other mods...")
                    # plt.imshow(imread(base_path+str(key_list[counter-1]+".png")))
            elif mode in ["FC", "T3-CM-5-SVD"]:
                selected_image, selected_label = caltech101_dataset[int(key_list[counter-1])]
                plt.imshow(np.transpose(selected_image, (1, 2, 0)))
            counter += 1 
            
            plt.axis('off')
            plt.tight_layout()
    else:
        for i in range(2):
            for j in range(int(value_m/2)):
                plt.subplot(2, int(value_m/2), counter)
                print(counter)
                if mode in ["HOGs" or "moments"] :
                    print("Please choose other mods...")
                    # plt.imshow(imread(base_path+str(key_list[counter-1]+".png")))
                elif mode in ["FC", "T3-CM-5-SVD"]:
                    selected_image, selected_label = caltech101_dataset[int(key_list[counter-1])]
                    plt.imshow(np.transpose(selected_image, (1, 2, 0)))

                counter += 1 
                plt.axis('off')
        plt.tight_layout()

    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if not os.path.exists("./data/output/"):
        os.makedirs("./data/output/")
    plt.savefig("./data/output/mode"+str(mode)+"_"+"l"+str(label_src.split("/")[-1].replace(".", ""))+"_"+"n"+str(value_n-1)+"_"+"m"+str(value_m)+current_time+".png")
    # Show the entire figure with subplots
    plt.show()


def retrieve_image(imageID_or_file):
    if isinstance(imageID_or_file, int):
        return dataset[imageID_or_file][0]
    else:
        return Image.open(imageID_or_file)


def get_most_similar_images(query_features, feature_space, k, query_imageID=None, query_image=None):
    scores = []

    for entry in database:
        # Skip the query image based on imageID or features
        if query_imageID is not None and query_imageID == entry["imageID"]:
            continue
        elif query_image is not None:
            similarity = compare_features(query_features, entry["features"][feature_space])
            if np.isclose(similarity, 1, atol=1e-09):
                continue

        score = compare_features(query_features, entry["features"][feature_space])
        scores.append((entry["imageID"], score))

    top_k_images = sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    return top_k_images


def compare_features(query_features, database_features):
    dot_product = np.dot(query_features, database_features)
    norm_a = np.linalg.norm(query_features)
    norm_b = np.linalg.norm(database_features)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

# try:
while True:
    choice = str(input("Please enter the task you want to execute (0a/0b/1/2a/2b/3/4/5/6/7/8/9/10/11): "))
    feature_space = None
    if choice == "0a":
        task_0a()
    elif choice == "0b":
        imageID = input("Enter the ImageId or provide Image Path: ")
        if imageID.isnumeric():
            imageID = int(imageID)
        feature_in = int(input("1) ColorMoments\n2) HOG\n3) AvgPool\n4) Layer3\n5) FCLayer\n\nEnter feature space: "))
        if feature_in == 1:
            feature_space = "ColorMoments"
        elif feature_in == 2:
            feature_space = "HOG"
        elif feature_in == 3:
            feature_space = "AvgPool"
        elif feature_in == 4:
            feature_space = "Layer3"
        elif feature_in == 5:
            feature_space = "FCLayer"
        else:
            print("Wrong feature space selected!")
        k = int(input("Enter the number of relevant images you want: "))
        task_0b(imageID, feature_space, k)

    elif choice == "1":
        label_name = input("Enter the label id: ")
        feature_in = int(input("1) ColorMoments\n2) HOG\n3) AvgPool\n4) Layer3\n5) FCLayer\n\nEnter feature space: "))
        if feature_in == 1:
            feature_space = "ColorMoments"
        elif feature_in == 2:
            feature_space = "HOG"
        elif feature_in == 3:
            feature_space = "AvgPool"
        elif feature_in == 4:
            feature_space = "Layer3"
        elif feature_in == 5:
            feature_space = "FCLayer"
        else:
            print("Wrong feature space selected!")
        k = int(input("Enter the number of relevant images you want: "))
        task_1(label_name, feature_space, k)

    elif choice == "2a":
        imageID = input("Enter the ImageId or provide Image Path: ")
        if imageID.isnumeric():
            imageID = int(imageID)
        feature_in = int(input("1) ColorMoments\n2) HOG\n3) AvgPool\n4) Layer3\n5) FCLayer\n\nEnter feature space: "))
        if feature_in == 1:
            feature_space = "ColorMoments"
        elif feature_in == 2:
            feature_space = "HOG"
        elif feature_in == 3:
            feature_space = "AvgPool"
        elif feature_in == 4:
            feature_space = "Layer3"
        elif feature_in == 5:
            feature_space = "FCLayer"
        else:
            print("Wrong feature space selected!")
        k = int(input("Enter the number of top labels you want: "))
        task_2a(imageID, feature_space, k)

    elif choice == "2b":
        imageID = input("Enter the ImageId or provide Image Path: ")
        if imageID.isnumeric():
            imageID = int(imageID)
        k = int(input("Enter the number of top labels you want: "))
        task_2b(imageID, k)

    elif choice == "3":
        feature_in = int(input("1) ColorMoments\n2) HOG\n3) AvgPool\n4) Layer3\n5) FCLayer\n6) RESNET\n\nEnter feature space: "))
        if feature_in == 1:
            feature_space = "ColorMoments"
        elif feature_in == 2:
            feature_space = "HOG"
        elif feature_in == 3:
            feature_space = "AvgPool"
        elif feature_in == 4:
            feature_space = "Layer3"
        elif feature_in == 5:
            feature_space = "FCLayer"
        elif feature_in == 6:
            feature_space = "RESNET"
        else:
            print("Wrong feature space selected!")
            exit()

        k = int(input("Enter the number of latent semantics you want: "))
        reduction_in = int(input(
            "Choose a dimensionality reduction technique:\n1) SVD\n2) NNMF\n3) LDA\n4) k-means\n\nEnter your choice: "))
        if reduction_in == 1:
            reduction_technique = "SVD"
        elif reduction_in == 2:
            reduction_technique = "NNMF"
        elif reduction_in == 3:
            reduction_technique = "LDA"
        elif reduction_in == 4:
            reduction_technique = "k-means"
        else:
            print("Invalid choice for reduction technique!")
            exit()

        task_3(feature_space, k, reduction_technique)


    elif choice == "4":
        feature_in = int(input("1) ColorMoments\n2) HOG\n3) AvgPool\n4) Layer3\n5) FCLayer\n6) RESNET\n\nEnter feature space: "))
        if feature_in == 1:
            feature_space = "ColorMoments"
        elif feature_in == 2:
            feature_space = "HOG"
        elif feature_in == 3:
            feature_space = "AvgPool"
        elif feature_in == 4:
            feature_space = "Layer3"
        elif feature_in == 5:
            feature_space = "FCLayer"
        elif feature_in == 6:
            feature_space = "RESNET"
        else:
            print("Wrong feature space selected!")
            exit()

        k = int(input("Enter the number of latent semantics you want: "))
        task_4(feature_space, k)

    elif choice == "5":
        feature_in = int(input("1) ColorMoments\n2) HOG\n3) AvgPool\n4) Layer3\n5) FCLayer\n6) RESNET\n\nEnter feature space: "))
        if feature_in == 1:
            feature_space = "ColorMoments"
        elif feature_in == 2:
            feature_space = "HOG"
        elif feature_in == 3:
            feature_space = "AvgPool"
        elif feature_in == 4:
            feature_space = "Layer3"
        elif feature_in == 5:
            feature_space = "FCLayer"
        elif feature_in == 6:
            feature_space = "RESNET"
        else:
            print("Wrong feature space selected!")
            exit()

        k = int(input("Enter the number of latent semantics you want: "))
        reduction_in = int(input(
            "Choose a dimensionality reduction technique:\n1) SVD\n2) NNMF\n3) LDA\n4) k-means\n\nEnter your choice: "))
        if reduction_in == 1:
            reduction_technique = "SVD"
        elif reduction_in == 2:
            reduction_technique = "NNMF"
        elif reduction_in == 3:
            reduction_technique = "LDA"
        elif reduction_in == 4:
            reduction_technique = "k-means"
        elif feature_in == 6:
            feature_space = "RESNET"
        else:
            print("Invalid choice for reduction technique!")
            exit()

        task_5(feature_space, k, reduction_technique)

    elif choice == "6":
        feature_in = int(input("1) ColorMoments\n2) HOG\n3) AvgPool\n4) Layer3\n5) FCLayer\n6) RESNET\n\nEnter feature space: "))
        if feature_in == 1:
            feature_space = "ColorMoments"
        elif feature_in == 2:
            feature_space = "HOG"
        elif feature_in == 3:
            feature_space = "AvgPool"
        elif feature_in == 4:
            feature_space = "Layer3"
        elif feature_in == 5:
            feature_space = "FCLayer"
        elif feature_in == 6:
            feature_space = "RESNET"
        else:
            print("Wrong feature space selected!")
            exit()

        k = int(input("Enter the number of latent semantics you want: "))
        reduction_in = int(input(
            "Choose a dimensionality reduction technique:\n1) SVD\n2) NNMF\n3) LDA\n4) k-means\n\nEnter your choice: "))
        if reduction_in == 1:
            reduction_technique = "SVD"
        elif reduction_in == 2:
            reduction_technique = "NNMF"
        elif reduction_in == 3:
            reduction_technique = "LDA"
        elif reduction_in == 4:
            reduction_technique = "k-means"
        else:
            print("Invalid choice for reduction technique!")
            exit()

        task_6(feature_space, k, reduction_technique)

    elif choice == "7":
        imageID = input("Enter the ImageId or provide Image Path: ")
        if imageID.isnumeric():
            imageID = int(imageID)
        latent_semantic = input("Enter the latent semantic file you want to use: ")
        # query_image = retrieve_image(imageID)
        k = int(input("Enter the number of similar images you want: "))
        task_7(imageID, latent_semantic, k)

    elif choice == "8":
        imageID = input("Enter the ImageId or provide Image Path: ")
        if imageID.isnumeric():
            imageID = int(imageID)
        latent_semantic = input("Enter the latent semantic file you want to use: ")
        k = int(input("Enter the number of similar labels you want: "))

        task_8(imageID, latent_semantic, k)

    elif choice == "9":
        label = input("Enter the label id: ")
        label_latent_semantic = input("Enter the Label's latent semantic file you want to use: ")
        k = int(input("Enter the number of similar labels you want: "))

        task_9(label, label_latent_semantic, k)

    elif choice == "10":
        label = input("Enter the label id: ")
        label_latent_semantic = input("Enter the Label's latent semantic you want to use: ")
        k = int(input("Enter the number of relevant images you want: "))

        task_10(label, label_latent_semantic, k)

    elif choice == "11":
        style = int(input("You want to use image from Database (1) or External (2): "))
        while style not in [1, 2]:
            style = int(input("You want to use image from Database (1) or External (2): "))
        
        if style == 1:
            label= input("Enter the label id (0, 20, 55, 100): ")
            get_m = input("Enter m significant images you want to get (5): ") # m = 5
            get_n = input("Enter n similar images you want in PageRank (10): ") # n = 10
            sup_list = ["T3-CM-5-SVD", "T3-HOG-5-LDA", "T3-HOG-5-SVD" , "T3-CM-10-LDA", "T3-CM-10-SVD",
                     "T3-HOG-10-LDA", "T3-HOG-1-LDA", "T3-HOG-1-SVD", "FC", "..."]
            print("Current support latent list: "+ str(sup_list))
            get_mode = input("Select mode: T3-CM-5-SVD / FC ... :")
            current_graph = input("Use pre-constructed graph? True/False:")

            if current_graph == "True":
                choice = True
            else:
                choice = False

            task_11(label_id = int(label), m = int(get_m), n = int(get_n), in_mode = get_mode, use_current_graph = choice)
        
        elif style == 2:
            src = input("Enter the image absolute path: ")
            get_m = input("Enter m significant images you want to get (5): ") # m = 5
            get_n = input("Enter n similar images you want in PageRank (10): ") # n = 10
            # get_mode = input("Select mode: T3-CM-5-SVD / FC ... :")
            print("Using default mode: FC...")
            choice = False # constructe new one
            
            task_11_extra(label_src=src, m = int(get_m), n = int(get_n), in_mode = "FC", use_current_graph = choice)
            # task_11_extra(label_src="/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase2/CSE515-Project/data/extraImg/Dwayne_Johnson.jpeg", m = 5, n = 10, in_mode = "FC", use_current_graph = choice)

    conn.close()

# except Exception as e:
#     print(e)