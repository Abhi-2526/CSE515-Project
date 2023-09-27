#Function to extract HOG Features 
import numpy as np 
import pandas as pd 
import torch 
from torchvision import transforms, models 
from torch.utils.data import DataLoader 
from  torchvision.datasets import ImageFolder 
import torchvision 
from PIL import image
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
from tqdm import tqdm 
resnet_model = models.resnet50(pretrained = True)

def extract_hog_features(image):
    # Convert PIL Image to NumPy array
    image_np = np.array(image)
    
    # Check if the image is grayscale
    if len(image_np.shape) == 2:
        gray_image = image_np
    else:
        gray_image = rgb2gray(image_np)
    
    # Resize the image
    resized_image = resize(gray_image, (300, 100))
    
    # Compute the HOG features
    features, hog_image = hog(resized_image, orientations=9, pixels_per_cell=(10, 10),
                              cells_per_block=(1, 1), visualize=True)
    
    return features, hog_image 
#Function to calculate the Resnet Layer Output 
from torchvision import models, transforms
from PIL import Image
import torch
import numpy as np


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize([0.53994344, 0.52009986, 0.49254049], 
                         [0.31415099, 0.30712622, 0.31878401]),  # Normalize the images
])

def resnet_computations(hook_layer, dataset):
    
    # List to store the output tensors for each image along with their ImageID
    outputs_with_ids = []
    
    # List to temporarily capture the output tensor from the hook
    captured_output = [None]

    # Hook function to capture the output tensor of a specified layer
    def capture_output(module, input, output):
        captured_output[0] = output

    # Register the hook function to the specified layer
    if hook_layer == 'avgpool':
        hook = resnet_model.avgpool.register_forward_hook(capture_output)
    elif hook_layer == 'layer3':
        hook = resnet_model.layer3.register_forward_hook(capture_output)
    elif hook_layer == 'fc':   
        hook = resnet_model.fc.register_forward_hook(capture_output)

    # Loop through the dataset
    for i in tqdm(range(len(dataset))):
        img, label = dataset[i]
        #skipping grayscale images 
        if img.mode == 'L' or img.mode == '1':
            continue
        # Apply transformations and prepare image batch
        img_tensor = transform(img)
        img_batch = img_tensor.unsqueeze(0)  # Add a batch dimension

        # Forward pass (disable gradient computation to save memory)
        with torch.no_grad():
            resnet_model(img_batch)
        
        # Retrieve the captured output tensor
        resnet_output = captured_output[0]
        if resnet_output is None:
            print("Warning: Hook Not Triggered")
            continue

        # Process the output tensor depending on the specified layer and store it in a dictionary
        output_dict = {"ImageID": i}
        if hook_layer == 'avgpool':
            avgpool_output = resnet_output.flatten().cpu().numpy()
            averaged_values = [(avgpool_output[i] + avgpool_output[i+1]) / 2.0 for i in range(0, len(avgpool_output), 2)]
            output_dict["Output"] = np.array(averaged_values)
        elif hook_layer == 'layer3':
            avg_vector = resnet_output.mean(dim=[2, 3]).cpu().numpy().squeeze()
            output_dict["Output"] = avg_vector
        elif hook_layer == 'fc':
            output_dict["Output"] = resnet_output.cpu().numpy().squeeze()
        
        # Append the dictionary to the list
        outputs_with_ids.append(output_dict)
    
    # Remove the hook to free resources
    hook.remove()
    
    return outputs_with_ids
