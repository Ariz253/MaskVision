# 😷 Face Mask Detector

An AI-powered web application that detects whether a person is wearing a face mask using Convolutional Neural Networks (CNN). Built with Streamlit and TensorFlow for real-time mask detection.

## 🌟 Features

- **Real-time Detection**: Instantly analyze uploaded images for mask presence
- **High Accuracy**: CNN-based binary classification model with 99.03% accuracy
- **User-Friendly Interface**: Modern dark-themed UI with intuitive design
- **Confidence Scoring**: Visual confidence metrics for each prediction

## 🚀 Live Demo

[MaskVision](https://maskvision.streamlit.app/) 

## 📋 Prerequisites

- Python 3.8 or higher
- TensorFlow 2.15.0
- Streamlit 1.28.0

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Ariz253/MaskVision.git
cd face-mask-detector
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Ensure your trained model file `face_mask_detector.h5` is in the project directory.

## 💻 Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

### How to Use:
1. Upload a face image (JPG, JPEG, or PNG format)
2. Click the "Analyze Image" button
3. View the prediction result and confidence score

## 📁 Project Structure

```
face-mask-detector/
│
├── app.py                      # Main Streamlit application
├── face_mask_detector.h5       # Trained CNN model
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🧠 Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 128x128 pixels (RGB)
- **Classification**: Binary (With Mask / Without Mask)
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Binary Crossentropy

## 🎨 Technologies Used

- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow/Keras
- **Image Processing**: PIL (Pillow)
- **Numerical Operations**: NumPy

## 📊 Model Performance

The model is trained to classify images into two categories:
- ✅ **With Mask**: Person wearing a face mask
- ⚠️ **Without Mask**: Person not wearing a face mask

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

**Ariz Ejaz khan**
- GitHub: [Ariz253](https://github.com/Ariz253)
- LinkedIn: [Ariz Ejaz Khan](www.linkedin.com/in/arizejazkhan)

---

⭐ If you found this project helpful, please consider giving it a star!