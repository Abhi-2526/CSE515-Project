# CSE515-Project

## Task 11:
Implement a program 

input:

(a) given a feature model or latent space, (b) a value n, (c) a value m, and (d) a label l

process:

1. creates a similarity graph, G(V, E), where V corresponds to the images in the database and E contains node pairs vi , vj such that, for each subject vi , vj is one of the n most similar images in the database in the given space

2. identifies the most significant m images (relative to the given label l) using personalized PageRank measure.

