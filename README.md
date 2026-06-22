<img width="940" height="604" alt="image" src="https://github.com/user-attachments/assets/5c003555-6505-40cb-bdc1-12a2d72586b4" /># Cats vs. Dogs Image Classification using Deep Learning

This project implements an end-to-end Deep Learning pipeline using Convolutional Neural Networks (CNNs) and Transfer Learning to automate the classification of digital pet images (Cats vs. Dogs). It compares four state-of-the-art architectures and deploys the champion model via an interactive Streamlit web application.

## Project Specifications & Requirements
* **Dataset:** A balanced sample of 8,000 images (4,000 cats / 4,000 dogs) sourced from Kaggle to prevent structural bias.
* **Data Split:** Programmatic 3-way split consisting of 80% Training (6,400 images), 10% Validation (800 images), and 10% Testing (800 images).
* **Preprocessing:** Automated programmatic image resizing to 224x224 pixels, pixel intensity normalization ([0, 255] scaled to [0, 1]), and duplicate/corrupt file filtering using MD5 hashes.
* **Framework:** Python, PyTorch, and Streamlit for web interface deployment.

## Architectural Comparison & Hyperparameter Tuning
Hyperparameter tuning was systematically executed across four pre-trained architectures using 3 distinct configurations per model, optimizing against Cross-Entropy Loss with GPU acceleration:

| Architecture | Best Configuration | Test Accuracy | Precision | Recall | F1-Score | Storage Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50** | **lr=0.001, Adam, 5 Epochs** | **99.00%** | **99.01%** | **99.01%** | **99.01%** | **90.00 MB** |
| VGG16 | lr=0.01, SGD, 10 Epochs | 98.75% | 98.29% | 99.26% | 98.77% | 512.20 MB |
| MobileNetV2 | lr=0.001, Adam, 5 Epochs | 98.62% | 98.05% | 99.26% | 98.65% | 8.73 MB |
| InceptionV3 | lr=0.01, SGD, 10 Epochs | 85.25% | 91.86% | 77.83% | 84.27% | 96.15% |

## Key Insights & Deployment Trade-offs
* **The Champion Model:** **ResNet50** yielded the highest overall predictive performance (99.01% F1-Score) due to its residual identity mappings that effectively mitigate vanishing gradient issues during fine-tuning. It serves as the premium candidate for server-side architectures.
* **Edge Compute Alternative:** **MobileNetV2** maintained strict parity with giant models, achieving a remarkable 98.62% accuracy with a minuscule memory footprint of only 8.73 MB. This efficiency, achieved via Depthwise Separable Convolutions, makes it the optimal choice for resource-constrained environments.
* **InceptionV3 Limitation:** Performance dropped to 85.25% on the test set because InceptionV3 natively expects 299x299 inputs; resizing to 224x224 altered its multi-scale parallel convolutional grids, causing spatial feature degradation.

## User Interface & Real-Time Inference
The optimal ResNet50 model was deployed using Streamlit. Users can upload raw images (JPEG/PNG), which are preprocessed in the backend, and the network immediately outputs the predicted class along with its confidence score percentage.

### Interface Screenshot:
<img width="940" height="654" alt="image" src="https://github.com/user-attachments/assets/3a48eb80-c41b-49fe-bc28-f207f523bb8d" />


<img width="940" height="654" alt="image" src="https://github.com/user-attachments/assets/ce07dc7a-72ea-4b30-a83d-fad55e686825" />
