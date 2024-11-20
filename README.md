# Human Activity Recognition Using LRCN

## Overview
This implementation focuses exclusively on **Human Activity Recognition (HAR)** using **Long Short-Term Memory Recurrent Convolutional Networks (LRCN)**. By leveraging the UCF50 dataset, this model achieves state-of-the-art results in classifying human activities from video data with a remarkable accuracy of **96%**.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Dataset
The UCF50 dataset is a rich video dataset containing 50 action categories such as walking, running, playing basketball, etc.

- **Instructions**: Download the dataset from the [UCF50 dataset page](https://www.crcv.ucf.edu/data/UCF50.php) and place it in the `data/` folder.
```

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow, Keras, OpenCV, Matplotlib, NumPy, and Pandas

Install required packages:
```bash
pip install -r requirements.txt
```

### Running on Google Colab
1. Open the `HAR_LRCN_Model.ipynb` notebook in Google Colab.
2. Upload the dataset to Google Drive or Colab's file system.
3. Follow the notebook instructions for data preprocessing and training.

## Preprocessing
Video preprocessing includes:
- **Frame extraction**: Capture key frames from videos.
- **Resizing**: Frames are resized to 64x64 pixels.
- **Normalization**: Pixel values are scaled to [0, 1].
- **Sequence preparation**: Videos are segmented into sequences for LSTM input.

Run the preprocessing script in `data_preprocessing.py` or within the notebook.

## Model Architecture
The **LRCN** model combines Convolutional Neural Networks (CNN) for spatial feature extraction and Long Short-Term Memory (LSTM) layers for temporal sequence learning:

- **CNN Backbone**:
  - Input: Resized video frames (e.g., 64x64x3).
  - Layers: Convolutional layers with ReLU activation and max-pooling for spatial feature extraction.
- **LSTM Head**:
  - Input: Sequential features from CNN.
  - Layers: LSTM layers to capture temporal dependencies in video sequences.
- **Output**:
  - Fully connected (Dense) layer with softmax activation for multi-class classification.

### Hyperparameters
- Optimizer: **Adam**
- Loss: **Categorical Cross-Entropy**
- Batch size: **32**
- Epochs: **50**
- Regularization: Dropout layers to mitigate overfitting.

## Training and Evaluation
1. Load preprocessed data.
2. Train the LRCN model using `HAR_LRCN_Model.ipynb`.
3. Monitor training using accuracy and loss plots.
4. Evaluate the model using metrics like:
   - **Accuracy**: 96% achieved on the test set.
   - **Confusion Matrix**: Visualize classification performance.

## Results
| Metric       | Value       |
|--------------|-------------|
| **Accuracy** | **96%**     |
| Loss         | Minimal     |

The LRCN model significantly outperforms traditional ConvLSTM-based methods in both accuracy and speed, making it suitable for real-time applications.

### Visualizations
Training progress, confusion matrix, and misclassification examples are available in the notebook.

## Future Work
- **Multi-person Activity Recognition**: Extend the model for simultaneous detection of multiple actions.
- **Real-time Inference**: Optimize for low-latency predictions on edge devices.
- **Dataset Expansion**: Include more diverse datasets to generalize better.

## References
1. Aboo, Adeeba Kh, and Laheeb M. Ibrahim. "Human Activity Recognition Using A Hybrid CNN-LSTM Deep Neural Network." *Webology* (2022).
2. Xia, Kun, Jianguang Huang, and Hanyu Wang. "LSTM-CNN architecture for human activity recognition." *IEEE Access* 8 (2020): 56855-56866.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
