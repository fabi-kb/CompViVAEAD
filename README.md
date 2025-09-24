# Computer Vision: VAE for Anomaly Detection
Final project for the Computer Vision course implementing a VAE for Anomaly Detection in industrial parts

# Processed dataset:
I uploaded the processed dataset [here](https://heibox.uni-heidelberg.de/d/82f8dc366d504a49b989/)

The images are normalized to 0-1 and have a image size of 128x128


# Structure:
- In the dataset.py is the code is the dataclass for loading the MVTEC dataset from the H5PY format. 
- In the eval.py is only a short code to see if the reconstruction of the base line implementation is working. 
- In the processing.py is the code for preprocessing the dataset and is only needed for that. 
- The training_loop.py contains the code for training a VAE model
- training.py contains the training loop and a code for detecting anomalies, but this is not tested yet
- VAE.py is the baseline VAE implementation

