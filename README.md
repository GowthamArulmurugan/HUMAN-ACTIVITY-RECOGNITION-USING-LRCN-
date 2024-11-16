Here's a comprehensive README file for your GitHub repository that includes Google Colab code for the "Human Activity Recognition Using Deep Learning Techniques" project:

---

# Human Activity Recognition Using Deep Learning Techniques

## Overview
This repository contains the implementation of **Human Activity Recognition (HAR)** using **Long Short-Term Memory Recurrent Convolutional Networks (LRCN)** in Python. The project is focused on accurately classifying human activities using video data, leveraging the UCF50 dataset. The developed model demonstrates significant performance improvements compared to traditional approaches like ConvLSTM, achieving an impressive accuracy rate of 96%.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Dataset
We utilize the [**UCF50 Dataset**](https://www.crcv.ucf.edu/data/UCF50.php) for this project. It includes a diverse collection of video clips depicting 50 distinct human actions such as walking, running, playing tennis, etc.

- **Download Instructions**: The dataset is not included in this repository. You can download it directly from the [UCF website](https://www.crcv.ucf.edu/data/UCF50.php). Extract the dataset into the `data/` directory before running the code.

## Project Structure
```
├── data/                    # Directory to store UCF50 dataset
├── notebooks/
│   ├── HAR_LRCN_Model.ipynb # Google Colab notebook for training LRCN model
│   ├── HAR_ConvLSTM_Model.ipynb # Google Colab notebook for training ConvLSTM model
├── utils/
│   ├── data_preprocessing.py # Code for preprocessing video data
│   ├── model_training.py     # Code for training models
├── requirements.txt          # List of required Python packages
├── README.md                 # Project documentation
└── LICENSE                   # License file
```

## Getting Started

### Prerequisites
- Python 3.x
- Google Colab
- TensorFlow, Keras, OpenCV, Matplotlib, NumPy, and Pandas

You can install the necessary libraries using:
```bash
pip install -r requirements.txt
```

### Running the Code on Google Colab
To run the project on Google Colab:
1. Open the `HAR_LRCN_Model.ipynb` or `HAR_ConvLSTM_Model.ipynb` notebook in Google Colab.
2. Ensure the dataset is uploaded to your Colab environment or Google Drive.
3. Follow the instructions provided in the notebooks to run the cells and train the models.

## Preprocessing
Data preprocessing is a crucial step to ensure high-quality input for the model. It involves:
- **Reading video files** from the dataset.
- **Resizing frames** to a fixed size (e.g., 64x64 pixels).
- **Normalizing pixel values** to a range between 0 and 1.
- **Data augmentation** (optional) to improve model robustness.

These steps are implemented in the `data_preprocessing.py` script.

## Model Training
We have implemented two models:
1. **ConvLSTM**: Combines Convolutional and LSTM layers to capture spatial and temporal features.
2. **LRCN (Long-term Recurrent Convolutional Networks)**: Extracts spatial features using CNN layers and temporal dependencies using LSTM layers.

Both models are trained using:
- **Adam optimizer**
- **Categorical cross-entropy loss**
- **Early stopping** and **dropout regularization** to prevent overfitting.

### Training on Google Colab
Simply run the code in the notebooks:
- For **ConvLSTM**: `HAR_ConvLSTM_Model.ipynb`
- For **LRCN**: `HAR_LRCN_Model.ipynb`

## Results
The performance of the models on the test set is summarized below:

| Model       | Accuracy | Processing Speed | Adaptability |
|-------------|----------|------------------|--------------|
| ConvLSTM    | 74%     | Slower           | Limited      |
| **LRCN**    | **96%** | Faster           | Yes          |

- **LRCN Model** achieved an accuracy of **96%**, outperforming the ConvLSTM model which achieved **74%**.
- The LRCN model demonstrated faster training and real-time detection capabilities due to reduced layers and optimized preprocessing.

### Evaluation Metrics
- **Accuracy**: Proportion of correct predictions.
- **Loss**: Indicates model performance over epochs.
- **Confusion Matrix**: Assesses classification errors.

Visualizations of training loss, accuracy, and confusion matrices are provided in the Colab notebooks.

## Future Work
- **Multi-person Activity Recognition**: Extend the model to recognize actions performed by multiple individuals simultaneously.
- **Real-time Deployment**: Optimize the model for deployment on edge devices with limited computational resources.
- **Improved Annotations**: Incorporate detailed annotations, such as bounding box coordinates for better activity recognition.

## References
1. Aboo, Adeeba Kh, and Laheeb M. Ibrahim. "Human Activity Recognition Using A Hybrid CNN-LSTM Deep Neural Network." *Webology* (2022).
2. Xia, Kun, Jianguang Huang, and Hanyu Wang. "LSTM-CNN architecture for human activity recognition." *IEEE Access* 8 (2020): 56855-56866.
3. Perez-Gamboa, Sonia, et al. "Improved sensor-based human activity recognition via hybrid CNN-RNN models." *IEEE Symposium on Inertial Sensors* (2021).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to adjust the README as needed to better align with your project specifics!
