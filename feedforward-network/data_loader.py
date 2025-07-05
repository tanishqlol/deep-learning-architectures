import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def load_mnist_data(validation_split: float = 0.2) -> Tuple[Tuple, Tuple, Tuple]:
    """
    Load MNIST data and split into train/validation/test sets.
    
    Why we return tuples of tuples:
    - First tuple: (x_train, y_train) 
    - Second tuple: (x_val, y_val)
    - Third tuple: (x_test, y_test)
    
    Tuples are also immutable so they are a good fit for storing fixed data/datasets.
    """
    
    # Load MNIST - TensorFlow downloads it automatically first time
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Why do we need validation data?
        # Train: Learns patterns
        # Validation: Tunes hyperparameters and provides checks for underfitting/overfitting
        # Test: Final unbiased evaluation
    
    # Calculate split indices
    num_train = len(x_train_full)
    num_val = int(num_train * validation_split)
    
    # Split the training data
    x_val = x_train_full[:num_val]
    y_val = y_train_full[:num_val]
    x_train = x_train_full[num_val:]
    y_train = y_train_full[num_val:]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def preprocess_data(x_train, y_train, x_val, y_val, x_test, y_test, normalize: bool = True, flatten: bool = True) -> Tuple:
    """
    Preprocess the data for neural network training.
    
    Reason for normalizing:
        Neural networks work best when inputs are roughly 0-1 range. This is because...

    Reason for flattening:  
        Fully connected layers expect 1D input, MNIST has is 2D 28x28 images, so flattening reshapes the data into 1D
    """
    
    if flatten:
        # Reshape from (samples, 28, 28) to (samples, 784), here samples = 60000 (full training images)
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_val = x_val.reshape(x_val.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    
    if normalize:
        # Convert to float32 and normalize to [0,1] range
        # Why float32? It's the standard for deep learning (good speed/memory balance)
        x_train = x_train.astype('float32') / 255.0
        x_val = x_val.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical: **(one-hot encoding)**
    # Why? Neural networks output probabilities for each class
    # So, instead of predicting "7", we predict [0,0,0,0,0,0,0,1,0,0] to mean the number 7 then match with the one-hot encoded label
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def visualize_data_samples(x_data, y_data, num_samples: int = 10, title: str = "Data Samples"):
    """
    Visualize sample images from the dataset.
    
    Looking at the data helps:
    - Spot data quality issues
    - Understand what you're working with
    - Verify preprocessing worked correctly
    """
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle(title)
    
    for i in range(num_samples):
        row = i // 5
        col = i % 5
        
        # Handle both flattened and image format
        if len(x_data[i].shape) == 1:  # Flattened
            image = x_data[i].reshape(28, 28)
        else:  # Already 28x28
            image = x_data[i]
        
        axes[row, col].imshow(image, cmap='Greys')
        
        # Show the label
        if len(y_data[i].shape) > 0:  # One-hot encoded
            label = np.argmax(y_data[i])
        else:  # Regular label
            label = y_data[i]
            
        axes[row, col].set_title(f'Label: {label}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def get_data_statistics(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Print comprehensive statistics about your dataset.
    
    From data statistics you understand:
    - Class balance (are some digits more common?)
    - Data shapes and types
    - Value ranges
    """
    
    print("====== MNIST Dataset Statistics ======")
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Input shape: {x_train.shape[1:]}")
    print(f"Number of classes: {y_train.shape[1] if len(y_train.shape) > 1 else len(np.unique(y_train))}")
    print(f"Data type: {x_train.dtype}")
    print(f"Value range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    
    # Class distribution
    if len(y_train.shape) > 1:  # One-hot encoded
        train_counts = np.sum(y_train, axis=0)
    else:  # Regular labels
        train_counts = np.bincount(y_train)
    
    print("\nClass distribution in training set:")
    for i, count in enumerate(train_counts):
        print(f"  Digit {i}: {count} samples ({count/len(y_train)*100:.1f}%)")

def prepare_mnist_data(validation_split: float = 0.2, 
                      normalize: bool = True, 
                      flatten: bool = True,
                      show_samples: bool = True,
                      show_stats: bool = True):
    """
    The main function to load and prepare MNIST data.
    """
    
    print("Loading MNIST data...")
    train_data, val_data, test_data = load_mnist_data(validation_split)
    
    print("Preprocessing data...")
    train_data, val_data, test_data = preprocess_data(*train_data, *val_data, *test_data, normalize=normalize, flatten=flatten)
    
    if show_stats:
        get_data_statistics(*train_data, *val_data, *test_data)
    
    if show_samples:
        visualize_data_samples(train_data[0], train_data[1], title="Training Data Samples")
    
    return train_data, val_data, test_data