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
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
import sqlite3
import pandas as pd

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
    FCLayer BLOB
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

    # Check if imageID already exists in the database
    cursor.execute("SELECT 1 FROM features WHERE imageID=?", (imageID,))
    exists = cursor.fetchone()

    if not exists:
        # Insert a new record if imageID doesn't exist
        cursor.execute('''
        INSERT INTO features (imageID, label, ColorMoments, HOG, AvgPool, Layer3, FCLayer)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            imageID, label, ColorMoments_bytes, HOG_bytes, AvgPool_bytes, Layer3_bytes, FCLayer_bytes))

        conn.commit()


def load_features_from_database():
    cursor.execute("SELECT imageID, label, ColorMoments, HOG, AvgPool, Layer3, FCLayer FROM features")
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
                "FCLayer": np.frombuffer(row[6], dtype=np.float32)
            }
        })


# Load Caltech101 dataset
dataset = Caltech101(root="/Users/abhinav/Desktop/CSE515-Project", download=True)

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
    """Compute the color moments for a specific cell in the image grid."""
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
        resnet50(image_tensor)

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

    return {
        "ColorMoments": color_moments,
        "HOG": hog_features,
        "AvgPool": avgpool_1024,
        "Layer3": layer3_1024,
        "FCLayer": fc_1000
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
    plt.subplot(1, k + 1, 1)  # +1 for the query image
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


def task_1(query_label_name, feature_space, k):
    load_features_from_database()

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
    min_val = np.min(all_features_matrix)
    if min_val < 0:
        all_features_matrix -= min_val

    # Apply the selected dimensionality reduction technique
    if reduction_technique == "SVD":
        U_k, Sigma_k, Vt_k = truncated_svd(all_features_matrix, k)
        latent_semantics = np.dot(all_features_matrix, Vt_k.T)
    elif reduction_technique == "NNMF":
        W, H = nnmf(all_features_matrix, k)
        latent_semantics = W
    elif reduction_technique == "LDA":
        lda = LatentDirichletAllocation(n_components=k)
        latent_semantics = lda.fit_transform(all_features_matrix)
    elif reduction_technique == "k-means":
        latent_semantics, cluster_labels = kmeans_clustering(all_features_matrix, k)

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

    # Perform CP decomposition
    weights, factors = parafac(tensor, rank=k)

    label_weights = factors[2]

    # Sort the weights and associate with labels
    idx_to_label_name = {idx: name for name, idx in label_name_to_idx.items()}
    df = pd.DataFrame(label_weights)
    df.index = df.index.map(idx_to_label_name)

    # Save to csv
    k = df.shape[1]
    filename = f"latent_semantics_{k}_components.csv"
    df.to_csv(filename)
    print(f"Data saved to {filename}")

    # Compute label-weight pairs
    label_weights = df.sum(axis=1)
    sorted_labels = label_weights.sort_values(ascending=False).index.tolist()
    sorted_weights = label_weights.sort_values(ascending=False).values.tolist()
    label_weight_pairs = [{'label': label, 'weight': weight} for label, weight in zip(sorted_labels, sorted_weights)]

    # Display the label-weight pairs
    for pair in label_weight_pairs:
        print(f"Label: {pair['label']}, Weight: {pair['weight']}")

    # Store the latent semantics in an output file
    output_filename = f"T4-{feature_space}-{k}.json"

    # Save to JSON
    with open(output_filename, 'w') as f:
        json.dump(label_weight_pairs, f, indent=4)

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
        nmf = NMF(n_components=k)
        latent_semantics = nmf.fit_transform(label_similarity_matrix)
    elif reduction_technique == "LDA":
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
        for idx in sorted_indices:
            label_name = list(label_name_to_idx.keys())[list(label_name_to_idx.values()).index(idx)]
            data_entry["Labels"][label_name] = float(weights[idx])
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
        U_k, Sigma_k, Vt_k = truncated_svd(image_similarity_matrix, k)
        latent_semantics = np.dot(image_similarity_matrix, Vt_k.T)
    elif reduction_technique == "NNMF":
        W, H = nnmf(image_similarity_matrix, k)
        latent_semantics = W
    elif reduction_technique == "LDA":
        lda = LatentDirichletAllocation(n_components=k)
        latent_semantics = lda.fit_transform(image_similarity_matrix)
    elif reduction_technique == "k-means":
        latent_semantics, cluster_labels = kmeans_clustering(image_similarity_matrix, k)

    # Store the latent semantics in an output file
    output_filename = f"T6-{feature_space}-{k}-{reduction_technique}.json"
    latent_semantics_list = []

    for i in range(k):
        if reduction_technique == "k-means":
            weights = latent_semantics[i]
        else:
            weights = latent_semantics[:, i]
        sorted_indices = np.argsort(weights)[::-1]

        semantic_entry = {"LatentSemantic": i + 1, "Images": []}
        for idx in sorted_indices:
            image_data = {
                "ImageID": database[idx]['imageID'],
                "Weight": f"{weights[idx]:.4f}"
            }
            semantic_entry["Images"].append(image_data)

        latent_semantics_list.append(semantic_entry)

    # Save to JSON
    with open(output_filename, 'w') as f:
        json.dump(latent_semantics_list, f, indent=4)

    print(f"Latent semantics stored in {output_filename}")


def task_7(imageID, latent_semantics_file, k):
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
        print(f"{idx}) Image_ID - {matched_imageID}, Score - {score}")

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
        plt.title(f"ID: {matched_imageID}\nScore: {score}")
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

    # Assuming you have a function `get_label_for_image` to retrieve the label for a given imageID or filename
    image_label = get_label_for_image(imageID)

    # Load the latent semantics from the provided JSON file
    with open(f"{latent_semantics_file}.json", 'r') as f:
        latent_semantics_data = json.load(f)

    # Extract the latent semantic corresponding to the given image label
    image_latent_semantic = None
    for entry in latent_semantics_data:
        if entry["Labels"].get(image_label):
            image_latent_semantic = entry["Labels"][image_label]
            break

    if not image_latent_semantic:
        print(f"Latent semantic for label {image_label} not found in the provided file!")
        return

    # Compute similarities between the image label latent semantic and the latent semantics of all labels
    similarities = {}
    count = {}
    for entry in latent_semantics_data:
        for label, score in entry["Labels"].items():
            similarity = abs(image_latent_semantic - score)
            if label in similarities:
                similarities[label] += similarity
                count[label] += 1
            else:
                similarities[label] = similarity
                count[label] = 1
            # similarities[label] = similarity

    average_scores = {label: similarities[label] / count[label] for label in similarities}

    print(average_scores)
    # Sort labels based on similarity
    sorted_similarities = sorted(average_scores, key=lambda x: x[1])[:k]
    print(sorted_similarities)

    # print("\nMost likely matching labels under the selected latent space: ")
    # for label, score in average_scores:
    #     print(f"Label - {label}, Score - {score:.4f}")


def task_9(label, latent_semantics_file, k):
    # Load the latent semantics from the provided file
    with open(f"{latent_semantics_file}.json", 'r') as f:
        latent_semantics_data = json.load(f)

    # Extract the weights for the given label
    for entry in latent_semantics_data:
        if entry["LatentSemantic"] == latent_semantics_file:
            label_weights = entry["Labels"].get(label)
            break
    else:
        print(f"Label {label} not found in the provided latent semantics file!")
        return

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

    return sorted_differences[:k]


def task_10(label, latent_semantics_file, k):
    # Load the latent semantics from the provided file
    with open(f"{latent_semantics_file}.json", 'r') as f:
        latent_semantics_data = json.load(f)

    # Extract the weights for the given label
    for entry in latent_semantics_data:
        if entry["LatentSemantic"] == latent_semantics_file:
            label_weights = entry["Labels"].get(label)
            break
    else:
        print(f"Label {label} not found in the provided latent semantics file!")
        return

    for name,idx in label_name_to_idx.items():
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


choice = str(input(
    "Please enter the task you want to execute:\n"
    "0a\n"
    "0b\n"
    "1\n"
    "2a\n"
    "2b\n"
    "3\n"
    "4\n"
    "5\n"
    "6\n"
    "7\n"
    "8\n"
    "9\n"
    "10\n"
    "11\n"
    ": "
))
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
    label_name = input("Enter the label name (Ex- 'Motorbikes', 'Faces'): ")
    if label_name not in label_name_to_idx:
        print(f"Label name '{label_name}' not found. Please enter a valid label name.")
    else:
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
        exit()

    k = int(input("Enter the number of latent semantics you want: "))
    task_4(feature_space, k)

elif choice == "5":
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

    task_5(feature_space, k, reduction_technique)

elif choice == "6":
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
    label = input("Enter the label: ")
    label_latent_semantic = input("Enter the Label's latent semantic file you want to use: ")
    k = int(input("Enter the number of similar labels you want: "))

    task_9(label, label_latent_semantic, k)

elif choice == "10":
    label = input("Enter the label: ")
    label_latent_semantic = input("Enter the Label's latent semantic you want to use: ")
    k = int(input("Enter the number of relevant images you want: "))

    task_10(label, label_latent_semantic, k)

conn.close()
