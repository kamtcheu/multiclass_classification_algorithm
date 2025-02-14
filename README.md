# Muticlass Image Classification

This project focuses on image classification using various Convolutional Neural Networks (CNNs) and classical machine learning models on the CIFAR-10 dataset.

## Project Structure




## Files Description

- **cnn_architectures.py**: Contains the implementation and training of various CNN architectures like ResNet18, EfficientNet B0, and ViT B16.
- **non_cnn_methods.py**: Implements classical machine learning models like SVM and Random Forest for CIFAR-10 classification.
- **demo.py**: Demonstrates the usage of trained models to classify a single image.
- **cnn.py**: Contains the implementation of custom CNN models and training/testing routines.
- **non_cnn_methods_with_plots.py**: Similar to `non_cnn_methods.py` but includes plotting of results.
- **data/**: Directory containing the CIFAR-10 and FashionMNIST datasets.
- **cifar-10-batches-py/**: Directory containing the CIFAR-10 dataset batches.
- **fine_tuned_cnn_vit_b_16.pth**: Pre-trained ViT B16 model weights.
- **random_forest_model.pkl**: Trained Random Forest model.
- **fine_tuned_cnn_resnet18.pth**: Pre-trained ResNet18 model weights.
- **fine_tuned_cnn_efficientnet_b0.pth**: Pre-trained EfficientNet B0 model weights.
- **dog.jpg**: Sample image used for demonstration in `demo.py`.

## Setup and Usage

1. **Clone the repository**:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Download CIFAR-10 dataset**:
    The dataset will be automatically downloaded when running the scripts.

4. **Train and evaluate models**:
    - To train and evaluate CNN models, run:
        ```sh
        python cnn_architectures.py
        ```
    - To train and evaluate classical ML models, run:
        ```sh
        python non_cnn_methods.py
        ```

5. **Run the demo**:
    ```sh
    python demo.py
    ```

## Results

- Confusion matrice of ViT_B16
![cnn_confusion_matrix_vitb16](https://github.com/user-attachments/assets/9b64e15a-ce08-4d29-a5fd-186afa1b5bd5)

- Confusion matrice of ResNet18
![cnn_confusion_matrix_efficientnet_b0_cuda](https://github.com/user-attachments/assets/805dead0-afa3-42c9-9e54-cd45c75e1961)

- Confusion matrice of EfficientNetB0
![cnn_confusion_matrix_restnet18_cuda](https://github.com/user-attachments/assets/756472c8-a511-4fae-b451-f62b8c3e278a)

- Inference Time for each algorihtm
![output](https://github.com/user-attachments/assets/2cef2775-21b8-4ec5-8b94-a3820a6e7ac6)

- Precision of the Algorihtms
![output2](https://github.com/user-attachments/assets/e2c1fe90-ca90-4554-8ef7-11a66966a1af)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- The project is inspired by various tutorials and resources from the PyTorch community.
- CIFAR-10 dataset is provided by the Canadian Institute For Advanced Research.

For any questions or issues, please open an issue in the repository.
