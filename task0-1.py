import warnings
import torch
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
import sqlite3

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

    HOG_bytes = np.array(features['HOG'].tobytes())

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
    """Compute the 10x10 color moments matrix for the image."""
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
        image = dataset[imageID][0]  # Retrieve the image using the imageID
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


def task_2b(imageID_or_file, k):
    task_2a(imageID_or_file, "FCLayer", k)


def get_top_k_weights(image_weights, k):
    sorted_indices = np.argsort(image_weights)[::-1][:k]
    top_k_weights = image_weights[sorted_indices]
    output = f"ImageID: {sorted_indices[0]}, Weights: {top_k_weights.tolist()}"

    return output


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
        svd = TruncatedSVD(n_components=k)
        latent_semantics = svd.fit_transform(all_features_matrix)
    elif reduction_technique == "NNMF":
        nmf = NMF(n_components=k)
        latent_semantics = nmf.fit_transform(all_features_matrix)
    elif reduction_technique == "LDA":
        lda = LatentDirichletAllocation(n_components=k)
        latent_semantics = lda.fit_transform(all_features_matrix)
    elif reduction_technique == "k-means":
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(all_features_matrix)
        cluster_labels = kmeans.predict(all_features_matrix)
        latent_semantics = kmeans.cluster_centers_

    # Store the latent semantics in an output file
    output_filename = f"{feature_space}_{reduction_technique}_latent_semantics.txt"
    with open(output_filename, 'w') as f:
        for i, entry in enumerate(database):
            imageID = entry["imageID"]
            if reduction_technique == "k-means":
                distances_to_centers = [np.linalg.norm(entry["features"][feature_space] - center) for center in latent_semantics]
                f.write(f"ImageID: {imageID}, Weights: {distances_to_centers}\n")
            else:
                weights = latent_semantics[i]
                formatted_weights = [float(w) for w in weights]
                f.write(f"ImageID: {imageID}, Weights: {formatted_weights}\n")

    print(f"Latent semantics stored in {output_filename}")


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


choice = str(input("Please enter the task you want to execute (0a/0b/1/2a/2b/3/4): "))
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

conn.close()
