# Brain Tumor Multi-Class Classification 

This project is a deep learning solution for classifying brain MRI images into four categories: **Glioma Tumor**, **Meningioma Tumor**, **Pituitary Tumor**, and **Healthy (No Tumor)**.

The notebook uses an advanced approach by first addressing class imbalance with a Conditional GAN (CGAN) to generate synthetic data, and then training an ensemble of three different powerful models (ConvNeXt, CoAtNet, and a ViT-Hybrid) to achieve robust classification.

## Project Overview

The core task is to build a classifier for brain tumor MRIs. A significant challenge in this dataset is the imbalance between the "healthy" class and the various tumor classes. This project implements the following pipeline:

1.  **Exploratory Data Analysis (EDA):** The dataset is loaded and the class distribution is visualized, confirming the data imbalance.
2.  **Data Augmentation (CGAN):** A Conditional GAN is trained to generate new, synthetic images for the minority (tumor) classes to create a balanced dataset.
3.  **Model Training:** Three different modern computer vision models are trained on this balanced dataset.
4.  **Evaluation:** Each model is evaluated individually on a test set.
5.  **Ensemble & Final Evaluation:** The predictions from all three models are combined (ensembled) to produce a final, more accurate prediction. Performance is measured using a classification report, confusion matrix, and multi-class ROC/AUC curves.

## Dataset

This project uses the **Brain Tumor MRI Multi-Class Dataset** from Kaggle.
The dataset is automatically downloaded in the notebook using the `opendatasets` library.

The four classes for classification are:
* Glioma Tumor
* Meningioma Tumor
* Pituitary Tumor
* Healthy (No Tumor)

## Key Features

* **Model Ensemble:** Uses an ensemble of three high-performing models to improve accuracy and robustness:
    * `convnext_tiny`
    * `coatnet_0_rw_224`
    * `vit_base_r50_s16_224` (Vision Transformer-ResNet Hybrid)
* **Class Imbalance Handling:** Implements a Conditional GAN (CGAN) to generate synthetic data for the under-represented classes, leading to a more balanced training dataset.
* **Modern Tooling:** Built with PyTorch, `timm` (PyTorch Image Models), and `albumentations` for data augmentation.
* **Comprehensive Evaluation:** Provides detailed metrics including confusion matrices and ROC/AUC curves for both individual models and the final ensemble.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_GITHUB_REPO_URL]
    cd [YOUR_REPO_NAME]
    ```
2.  **Install Dependencies:**
    The notebook installs its own dependencies in the first cell. The key libraries are:
    ```bash
    pip install torch torchvision timm albumentations opendatasets scikit-learn pandas matplotlib
    ```
3.  **Run the Notebook:**
    * Open `Brain_Tumor_Detection.ipynb` in Jupyter, VS Code, or Google Colab.
    * Run the cells sequentially. The dataset will be downloaded automatically.
    * **Note:** The notebook is configured to use a GPU (`DEVICE='cuda'`) for training the GAN and the classifiers. Running this on a CPU will be very slow.

4.  **Configuration:**
    You can modify key parameters in the **Configuration** cell (cell [1]):
    * `USE_CGAN`: Set to `False` to skip the (slow) GAN training and run the classifiers on the original, imbalanced dataset for comparison.
    * `EPOCHS`: Adjust the number of training epochs for the classifiers.
    * `CGAN_EPOCHS`: Adjust the number of epochs for training the GAN.

## Results

The notebook generates the following evaluation metrics after training:
* **Classification Report:** Precision, Recall, and F1-score for each class.
* **Confusion Matrix:** A plot to visualize where the models are making errors.
* **ROC/AUC Curves:** A multi-class ROC curve to show the performance of each model and the final ensemble.
