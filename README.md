# Lane Line Detection

This project uses a neural network to detect lane lines on road images. 
The model employs a UNet architecture for image segmentation to identify road markings.


## Dataset: TUSimpleDataset

### Description
The TUSimpleDataset contains road images with corresponding annotations for lane markings. 
Each image has a mask indicating the position of lane lines on the road.

### Parameters:
- **data_path**: Path to the root directory where images and annotations are stored.
- **json_mapping**: A dictionary that maps folders to their corresponding JSON files.
- **folders**: List of folders containing images.
- **target_height**: The height of the image after scaling (default is 360).

### How It Works:
1. **Annotation Loading**: JSON files contain coordinates of the lane lines and their vertical positions.
2. **Image Scaling**: Images are resized based on the target height, maintaining the aspect ratio.
3. **Mask Creation**: Masks are created for each image, where lane lines are drawn as white pixels on a black background.
4. **Tensor Conversion**: Images and masks are converted to PyTorch tensors for use in the model.


## Model: UNet

### Description
The UNet model uses a popular architecture for image segmentation tasks. 
It consists of several DoubleConv blocks to learn complex patterns, as well as downsampling and upsampling blocks to preserve and recover spatial information.

### Architecture:
- **DoubleConv**: Each block consists of two convolutions to help the network learn complex patterns.
- **Downsampling**: Reduces the image size to capture important features.
- **Upsampling**: Restores the image size while preserving key details.


## Training

### Description
The model is trained using the BCEWithLogitsLoss loss function, which is suitable for binary segmentation tasks. 
The Adam optimizer is used to update the model’s parameters.

### Training Process:
1. **Data Loading**: DataLoader is used for both training and validation datasets.
2. **Loss Calculation**: For each image, loss (prediction error) is calculated.
3. **Parameter Updates**: Model parameters are updated during training.
4. **Saving the Best Model**: The best model is saved after each epoch if its performance improves on the validation set.


## Visualization

### Description
The visualize_predictions function allows us to visualize the model’s predictions and compare them with the ground truth (true lane line masks) on input images. 
This helps in evaluating the model’s performance and understanding how well the model detects lane lines.

### How It Works:
1. **Prediction Generation**: The model generates predicted lane masks for each input image.
2. **Thresholding**: The model's output probabilities are passed through a sigmoid activation and then thresholded to create binary predictions (1 for lane pixels, 0 for non-lane pixels).
3. **Visualization**: A batch of images is displayed along with:
    - The input image.
    - The true lane line mask (ground truth).
    - The predicted lane line mask.

### Visualization process
The function uses Matplotlib to display the images in a side-by-side comparison. Each row contains:
- **Original Image**: The input image with road markings.
- **True Mask**: The actual lane markings (ground truth) for the corresponding image.
- **Predicted Mask**: The predicted lane markings from the model.
