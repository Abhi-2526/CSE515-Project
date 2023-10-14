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
import datetime
import types
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


mode = "fc_layer" # HOGs, moments, avg_pool, fc_layer, layer3, T3-CM-5-SVD
# ******************************* formal query input **************************************
feature_model = mode
value_n = 10 # n most similar images in the database in the given space
value_m = 5 # m most significant images
# *****************************************************************************************
use_current_graph = False
# exist_graph_path = "FC-Simi-Graph-20.gexf"
save_graph_name = "FC-Simi-Graph-100.gexf"


# image_id = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/testset/query/image_0001.jpg"

# image_id = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/testset/query/880.jpg"

# image_id = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/testset/query/plane.png"

# image_id = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/testset/query/cafeeecat.png"

# image_id = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/testset/query/taiji.png"

# image_id = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/testset/query/8676.jpg"

# image_id = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/testset/query/5122.jpg"

# two sets of data loading styles:

if mode == ("HOGs" or "moments"):

    image_id = '0'

    mapping = {"0":"8394", "880":"4402", "2500":"1134", "5122":"", "8676":"3912"}
    base_path = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/origin/"
    label = mapping[image_id] # query label
    input_path = base_path+label+".png"
    img = imread(input_path)

elif mode == ("fc_layer"):

    # image_id = 0
    # image_id = 20
    # image_id = 55
    image_id = 100
    # image_id = 880
    # image_id = 2500
    # image_id = 5122
    # image_id = 8676

    download_dir = "/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/tv_download/"
    
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


if mode == "HOGs":
    data = torch.load("/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/rgb_data_HOGs.pt")
    descriptor_fun = descriptor_HOG
    
elif mode == "moments":
    data = torch.load("/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/rgb_data_moments.pt")[-500:]
    descriptor_fun = descriptor_moments

elif mode == "avg_pool":
    data = torch.load("/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/rgb_data_avgpool_1024.pt")

elif mode == "fc_layer":
    # ('0', tensor([[6.1887e+00,...937e-04]]))
    data = torch.load("/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/rgb_data_fclayer_1000.pt")[0:500]
    image_data = data[image_id][1]
    # descriptor_fun = descriptor_moments

elif mode == "layer3":
    data = torch.load("/Users/danielsmith/Documents/1-RL/ASU/courses/23Fall/CSE515/project/phase1/caltech-101/dataset/rgb_data_layer3_1024.pt")
    


if use_current_graph:
    G = nx.read_gexf(exist_graph_path)
else:
    # Create a graph.
    G = nx.Graph()
    # target_description = descriptor_fun(input_path)

    personalized_teleport_vector = {image: 0.0 for image, _ in data}
    personalized_teleport_vector[label] = 1.0

    for node_id, descrip in tqdm(data):
        # add note
        G.add_node(node_id)
        # renew list for every new node
        evaluate_list = []
        if mode == ("HOGs" or "moments") :
        # current descriptor from node
            node_now = descriptor_fun(base_path+str(node_id+".png"))

        elif mode =="moments":
            node_now = np.transpose(node_now, (1, 2, 3, 0))
        
        elif mode == "fc_layer":
            # the data set directly contains the fc_layer output (1, 1000), so we take [0] ==> (1000,)
            node_now = descrip.detach().numpy()[0]

        # if mode == "avg_pool":
            

        for id_now, description in data:
            if mode == "HOGs":
                evaluate_list.append((id_now, Euclidean_Distance(node_now, description)))
            elif mode == "moments":
                description_new = np.transpose(description, (1, 2, 3, 0))
                evaluate_list.append((id_now, Euclidean_Distance_for_moments(node_now, description_new)))
            elif mode == "fc_layer":
                # evaluate_list.append((id, Euclidean_Distance(image_data.detach().numpy(), description_new.detach().numpy())))
                evaluate_list.append((id_now, np.corrcoef(image_data.detach().numpy()[0], description.detach().numpy()[0])[0, 1]))

        if mode == "HOGs": # distance = Euclidean distance
            sorted_data = sorted(evaluate_list, key=lambda x:x[1])
        elif mode == "fc_layer":
            sorted_data = sorted(evaluate_list, key=lambda x:x[1], reverse=True)
        
        selected_result = sorted_data[:value_n]
        print(selected_result)
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

sorted_images = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

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
        if mode == ("HOGs" or "moments") :
                plt.imshow(imread(base_path+str(key_list[counter-1]+".png")))
        elif mode == "fc_layer":
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
            if mode == ("HOGs" or "moments") :
                plt.imshow(imread(base_path+str(key_list[counter-1]+".png")))
            elif mode == "fc_layer":
                selected_image, selected_label = caltech101_dataset[int(key_list[counter-1])]
                plt.imshow(np.transpose(selected_image, (1, 2, 0)))

            counter += 1 
            plt.axis('off')
    plt.tight_layout()

current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
plt.savefig("./data/output/mode"+str(mode)+"_"+"l"+str(image_id)+"_"+"n"+str(value_n-1)+"_"+"m"+str(value_m)+current_time+".png")
# Show the entire figure with subplots
plt.show()


