from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew
from skimage.color import rgb2gray
import torch
import os, sys
from tqdm import tqdm
import networkx as nx


mode = "HOGs" # HOGs, moments, avg_pool, fc_layer, layer3

# image_id = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/testset/query/image_0001.jpg"

# image_id = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/testset/query/880.jpg"

# image_id = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/testset/query/plane.png"

# image_id = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/testset/query/cafeeecat.png"

# image_id = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/testset/query/taiji.png"

image_id = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/testset/query/8676.jpg"

# image_id = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/testset/query/5122.jpg"



# ******************************* formal query input **************************************
feature_model = mode
value_n = 3 # n most similar images in the database in the given space
value_m = 5 # m most significant images
label = image_id # query label
# *****************************************************************************************



img = imread(image_id)
plt.axis("off")
plt.imshow(img)
print(img.shape)
plt.show()

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


target_description = descriptor_HOG(image_id)

def Euclidean_Distance(query, candidate):
    var = query - candidate
    distances = np.sqrt(np.sum(var ** 2))

    # mean_again = np.mean(distances, axis=(0, 1))
    return distances




if mode == "HOGs":
    data = torch.load("/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/rgb_data_HOGs.pt")
    
elif mode == "moments":
    data = torch.load("/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/rgb_data_moments.pt")


elif mode == "avg_pool":
    data = torch.load("/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/rgb_data_avgpool_1024.pt")

elif mode == "fc_layer":
    data = torch.load("/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/rgb_data_fclayer_1000.pt")

elif mode == "layer3":
    data = torch.load("/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/rgb_data_layer3_1024.pt")
    
base_path = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/origin/"

# Create a graph.
G = nx.Graph()

save_edge = []



for node_id, _ in tqdm(data):
    # renew list for every node center
    evaluate_list = []
    node_now = descriptor_HOG(base_path+str(node_id+".png"))
    for id_now, description in data:
            evaluate_list.append((id_now, Euclidean_Distance(node_now, description)))
    sorted_data = sorted(evaluate_list, key=lambda x:x[1])
    K = 11
    selected_result = sorted_data[:K]

    # the id is the data[0]
    key_list = []
    for i in range(len(selected_result)):
        key_list.append(selected_result[i][0])
    print(key_list)
    mapping = [(node_id, s) for s in key_list[1:]]
    save_edge.append(mapping)

save_path = "Graph_data2.pt"
torch.save(save_edge, save_path)

print("saved!")



# plt.figure(figsize=(10, 3))

# counter = 1
# for i in range(2):
#     for j in range(5):
#         plt.subplot(2, 5, counter)
#         counter += 1 
#         plt.imshow(imread(base_path+str(key_list[counter]+".png")))
#         plt.axis('off')


# plt.tight_layout()

# # Show the entire figure with subplots
# plt.show()


