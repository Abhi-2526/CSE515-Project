# CSE515-Project
Introduction:

This is the Phase 2 repo for Multimedia and Web Databases. At its core, the project seeks to extract underlying latent semantics from a database of images, by employing various dimensionality reduction techniques, such as Singular Value Decomposition (SVD), Non-Negative Matrix Factorization (NNMF), Latent Dirichlet Allocation (LDA), k-means clustering, and CP Decomposition.Those reduced dimensional vectors are used to see the relevant images and labels.

Datasets:
This repository requires two kinds of data:
1. Data for task 0 - 10 is stored in Sqlite Database, database file please find in: 
https://drive.google.com/drive/folders/18ZQ1xyfsRyFmPVvK46mS1qBGINVZlb6f?usp=sharing


2. Data for pre-constructed graphs is stored in the ./graph_files

If the data is not in the ./graph_files, task 11 will automatically start the constructing process and store is inside the folder.


Installation:
  Source:
  To install our environment, please follow these instructions:  
  1)	mkdir phase2
  2)	cd phase2
  3)	git clone git@github.com:LongchaoDa/CSE515-Project.git
     
  Dependencies:
  To install dependencies run the following command: pip install PIL sqlite3 sklearn numpy matplotlib torch torchvision pillow scikit-image scipy tensorly networkx

System Run:
To run the system: “python3 main.py”

# Usage Instructions
1. When started the algorithm, you will be able to input the command in the terminal.
2. Please follow the instructions in the command line for each task.
3. For task selection:   
Please select from below list:
`0a/0b/1/2a/2b/3/4/5/6/7/8/9/10/11`
4. Note: This program is not stopping until you stop manually, so you can try options in a better experience.
5. Stop the code: Just Ctrl+C
