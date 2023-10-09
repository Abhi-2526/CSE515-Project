import numpy as np
import cv2
import os
from skimage.io import imread
from skimage.transform import resize

# input_path = '/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase2/code/data/query/0.jpg' # 8394
# input_path = '/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase2/code/data/query/880.jpg' # 4402
# input_path = '/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase2/code/data/query/2500.jpg' # 1134
# input_path = '/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase2/code/data/query/5122.jpg' # cafe cat: 
input_path = '/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase2/code/data/query/8676.jpg' # 3912

base_path = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase2/code/data/origin"

file_list = []
for root, dirs, files in os.walk(base_path):
    for file_name in files:
        if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            file_path = os.path.join(root, file_name)
            file_list.append(file_path)


def calculate_mse(image1, image2):
    """
    Calculate the Mean Squared Error (MSE) between two image arrays.

    Args:
    image1 (numpy.ndarray): The first image as a NumPy array.
    image2 (numpy.ndarray): The second image as a NumPy array.

    Returns:
    float: The calculated MSE value.
    """
    # Ensure both images have the same shape
    if image1.shape != image2.shape:
        image1 = resize(image1, (300, 100))
        image2 = resize(image2, (300, 100))
        # raise ValueError("Both images must have the same dimensions.")

    # Calculate the MSE
    mse = np.mean((image1 - image2) ** 2)

    return mse

image1 = imread(input_path)

similar_data = {image2.split("/")[-1]: calculate_mse(image1, imread(image2)) for image2 in file_list}

sorted_mse_dict = dict(sorted(similar_data.items(), key=lambda item: item[1]), reversed=True)

print(next(iter(sorted_mse_dict.items())))

# np.save(sorted_mse_dict[:10], "data.csv")