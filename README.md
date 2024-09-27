# Arabic Handwriting Classification

This project implements a machine learning model for classifying Arabic handwritten characters. It uses deep learning techniques to recognize and categorize Arabic letters and numerals from images.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/arabic-handwriting-classification.git
   cd arabic-handwriting-classification
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages

The following packages are necessary for this project:

- torch
- torchvision
- torchaudio
- torchmetrics
- torchmetrics.classification
- torchvision
- torchvision.models
- torchvision.transforms
- torchvision.io
- torchvision.io.image
- torchvision.io.video
- torchvision.io.video.read
- torchvision.io.video.write
- pandas
- numpy
- matplotlib
- opencv-python
- tensorflow

You can install these packages manually using pip:

```bash
pip install torch torchvision torchaudio torchmetrics torchmetrics.classification torchvision opencv-python pandas numpy matplotlib
```

Or simply use the `requirements.txt` file as mentioned in step 3 of the installation process.

## Usage

To run the classification model:

1. Ensure you have the dataset in the correct directory (see [Dataset](#dataset) section).
2. Run the training script:
   ```bash
   python train.py --data_dir <path_to_data_directory> --model_name <model_name> --epochs <number_of_epochs> --batch_size <batch_size> --learning_rate <learning_rate>
   ```
3. For evaluation, use:
   ```bash
   python evaluate.py --data_dir <path_to_data_directory> --model_name <model_name> --batch_size <batch_size>
   ```

## Dataset

This project uses the [Arabic Handwritten Characters Dataset](https://www.kaggle.com/datasets/mloey1/ahcd1) from Kaggle. Please download the dataset and place it in the `data/` directory before running the scripts.

## Model Architecture

The model uses a Convolutional Neural Network (CNN) architecture. Details of the layers and parameters can be found in `model.py`.

## Training

The training process involves data preprocessing, augmentation, and model fitting. See `train.py` for the implementation details.

## Evaluation

The model's performance is evaluated using accuracy and confusion matrix. The evaluation script is in `evaluate.py`.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
