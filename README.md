# Face Recognition with Triplet Loss and ResNet50

## Overview

This project implements a face recognition pipeline using a Siamese network with a ResNet50 backbone and **standard triplet loss**. The model learns embeddings such that faces of the same identity are close in embedding space, while faces of different identities are far apart.

Key features:

* Face cropping and alignment script (`crop_faces.py`)
* Custom `tf.data` pipeline (`dataset.py`) for loading triplets
* Light augmentations: random horizontal flip, rotation, brightness jitter
* ResNet50-based embedding model (`model.py`)
* Standard **TripletLoss** for training
* Inference script to compare face embeddings (`inference.py`)

## Repository Structure

```
├── config.py         # Hyperparameters and paths
├── crop_faces.py     # Utility to detect and crop faces from raw images
├── dataset.py        # Dataset class and MapFunction for loading triplets
├── model.py          # Defines the embedding model (ResNet50 backbone)
├── train.py          # Training script using TripletLoss
├── inference.py      # Inference example to compare two face images
└── README.md         # Project documentation
```

## Requirements

* Python 3.8+
* TensorFlow 2.x
* OpenCV (for face detection)
* tqdm (progress bars)

Install dependencies:

```bash
pip install tensorflow opencv-python tqdm
```

## Data Preparation

This implementation is designed to train on the **CASIA-WebFace** dataset. Follow these steps:

1. **Download CASIA-WebFace**: Obtain the dataset from its official source`.
2. **DataSet Directory Structure**:

   ```
   dataset/
   ├── 000001/
   │   ├── img1.jpg
   │   └── img2.jpg
   ├── 000002/
   │   ├── img1.jpg
   │   └── img2.jpg
   └── ...
   ```
3. **Crop Faces**: Run the cropping script to detect and save aligned face crops:

   ```bash
   python crop_faces.py --input_dir dataset --output_dir cropped_data
   ```
4. **Dataset Structure**: After cropping, ensure structure:

   ```
   cropped_data/
   ├── train/
   │   ├── 000001/
   │   │   ├── crop1.jpg
   │   │   └── crop2.jpg
   │   └── ...
   ```

## Configuration (`config.py`)

Edit paths and hyperparameters:

```python
# Paths
TRAIN_DIR = "path/to/cropped_data/train"
TEST_TRIPLETS = "path/to/test_triplets.txt"
# Training
BATCH_SIZE = 256
STEPS_PER_EPOCH = 50
VALIDATION_STEPS = 10
EPOCHS = 10
IMAGE_SIZE = (224, 224)
MARGIN = 0.2
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = "path/to/checkpoints"
TENSORBOARD_LOGDIR = "path/to/logs"
```

## Augmentation

Light augmentations are applied in `MapFunction`:

* Random horizontal flip
* Random rotation (±10° via `tf.keras.layers.RandomRotation`)
* Random brightness adjustment

Adjust or add more augmentations (contrast, cutout) in `dataset.py` as needed.

## Model Architecture (`model.py`)

* Base: Pretrained ResNet50 without top classifier
* Global average pooling
* Dense projection layer to 128-D embeddings
* L2 normalization

Customize layers or embedding size by editing `build_embedding_model()`.

## Training (`train.py`)

Train with standard Triplet Loss:

python train.py

Outputs the distance between embeddings; lower distance indicates higher similarity.

## Training & Testing Metrics

```
Train Loss: 0.1958
Train Precision: 0.9356
Train Recall: 0.9254
Test Loss: 0.2168
Test Precision: 0.9124
Test Recall: 0.9078
```

# Pros & Cons, and Future Improvements

**Pros**:

* Leverages a large-scale dataset (CASIA-WebFace) for robust feature learning.
* Simple pipeline with standard triplet loss—easy to understand and extend.
* Light augmentations improve generalization without heavy processing overhead.

**Cons**:

* No semi-hard or hard negative mining by default; relies on pre-generated triplets or random sampling, which may slow convergence.
* Limited augmentation variety; complex variations (occlusions, motion blur) are not covered.
* Fixed hyperparameters may not generalize to other datasets without tuning.

**Future Improvements**:

* Integrate **semi-hard** or **hard** negative mining (e.g., `tf.keras.losses.TripletSemiHardLoss` or TFA `TripletHardLoss`) to accelerate and stabilize training.
* Add advanced augmentations: occlusion, cutout, color jitter, Gaussian blur.
* Experiment with alternative backbones (EfficientNet, MobileNet) for efficiency gains.
* Implement an online memory bank for mining hard negatives across batches.
* Introduce face-quality filtering and alignment refinement to reduce noisy inputs.

