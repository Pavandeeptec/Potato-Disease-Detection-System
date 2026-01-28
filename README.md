# 🌱 Potato Disease Detection System

A comprehensive AI-powered web application for detecting potato leaf diseases using computer vision and deep learning. The system can identify healthy leaves, early blight, and late blight with high accuracy and generate professional reports.

![Potato Disease Detection](https://img.shields.io/badge/AI-Computer%20Vision-green)
![Flask](https://img.shields.io/badge/Flask-Web%20App-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)

## 🎯 Features

- **AI-Powered Detection**: Uses Vision Transformer (ViT) for accurate disease classification
- **Multiple Input Methods**: Upload files or capture images with camera
- **Batch Processing**: Analyze multiple images simultaneously
- **Professional Reports**: Export results as PDF or CSV
- **Mobile Friendly**: Responsive design works on all devices
- **Real-time Analysis**: Fast processing with confidence scores

## 🔬 Disease Classes

- **Healthy Leaves**: No disease detected
- **Early Blight**: Alternaria solani infection
- **Late Blight**: Phytophthora infestans infection

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/pavandeeptec/potato-disease-detection.git
cd potato-disease-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Get the trained model:
   - **Option 1**: Train your own model using `train.py` (recommended)
   - **Option 2**: Download pre-trained model from [Google Drive](https://drive.google.com/file/d/1xP7g79pBvqIqFlQiOGTvwlIuGIqIjysO/view?usp=drive_link))
   - Place the model file as `potato_disease_model.pth` in the root directory

4. Run the application:
```bash
python app.py
```

5. Open your browser and go to `http://localhost:5000`

## 📁 Project Structure

```
potato-disease-detection/
├── app.py                    # Main Flask application
├── train.py                  # Model training script
├── requirements.txt          # Python dependencies
├── potato_disease_model.pth  # Trained model (not included - see setup)
├── templates/
│   └── index.html           # Web interface
├── static/
│   └── style.css            # Styling
├── dataset/                 # Training dataset (not included)
└── README.md               # This file
```

> **Note**: The trained model file (`potato_disease_model.pth`) is not included in this repository due to GitHub's file size limitations (982MB). Please follow the setup instructions to obtain the model.

## 🎨 Screenshots

### Main Interface
- Modern drag & drop upload interface
- Camera capture functionality
- Real-time disease analysis

### Results Display
- Detailed disease information
- Confidence scores
- Treatment recommendations

## 🔧 Training Your Own Model

1. Prepare your dataset in the following structure:
```
dataset/
├── train/
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   └── Potato___healthy/
└── val/
    ├── Potato___Early_blight/
    ├── Potato___Late_blight/
    └── Potato___healthy/
```

2. Run the training script:
```bash
python train.py
```

## 📊 Model Performance

- **Architecture**: Vision Transformer (ViT-Base-Patch16-224)
- **Accuracy**: 95%+ on validation set
- **Processing Time**: 2-3 seconds per image
- **Supported Formats**: JPG, PNG, JPEG

## 🌐 API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Image analysis endpoint
- `POST /export` - Report generation endpoint

## 🔒 Privacy & Security

- **No Data Storage**: Images are processed in memory only
- **Local Processing**: AI runs on your server, not in the cloud
- **Secure Uploads**: File validation and sanitization
- **HTTPS Ready**: Camera access requires secure connection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Vision Transformer model from Hugging Face
- Dataset from [PlantVillage](https://www.plantvillage.org/)
- Flask web framework
- PyTorch deep learning library

## 📞 Support

If you have any questions or issues, please open an issue on GitHub or contact [pavandeeptech@gmail.com].

---

**Made for agricultural innovation**
