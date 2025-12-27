# NepAgNeuroVision v1.0

**CNN-Based Crop Abnormality Classification for Nepali Agriculture**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)  
[![Flask](https://img.shields.io/badge/Flask-2.0+-black)](https://flask.palletsprojects.com/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)  
<!-- Replace with your actual DOI after Zenodo archiving -->
[![DOI](https://doi.org/10.5281/zenodo.18069308)](https://doi.org/10.5281/zenodo.18069307)

**NepAgNeuroVision** is a bilingual (English/नेपाली) Flask web application that employs Convolutional Neural Networks (CNNs) and custom hybrid architectures to classify crop abnormalities across **11 major crops** grown in Nepal, covering a total of **36 classes** (including healthy states, diseases, and nutrient deficiencies). The system provides instant AI-powered diagnosis, confidence breakdown, and research-backed, Nepal-specific management recommendations to support farmers.

**Developer**: Bikas Basnet – Agricultural Scientist & AI Researcher  
**Portfolio**: [www.bikasbasnet.com.np](https://www.bikasbasnet.com.np/)

## Key Features

- Crop verification using a dedicated hybrid CNN classifier (confidence threshold >60%)
- Crop-specific deep learning models (Hybrid ShuffleNet-SqueezeNet, MobileNetV2, ResNet18, SqueezeNet)
- Top-3 prediction confidence breakdown with interactive doughnut chart
- Localized recommendations including priority, actions, details, and follow-up tailored to Nepali conditions
- Full bilingual interface (English/Nepali) with Devanagari script support
- Modern responsive design with glassmorphism, dark mode, drag-and-drop upload, and smooth animations
- Optimized for mobile use in field conditions

**Supported Crops and Classes** (Total: 36)
- Acid-Lime (2 classes)
- Cauliflower (4 classes)
- Chilli (3 classes)
- Kidneybean (2 classes)
- Large Cardamom (2 classes)
- Maize (5 classes)
- Mungbean (2 classes)
- Onion (2 classes)
- Rice (9 classes)
- Sesame (3 classes)
- Strawberry (2 classes)

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Local Installation
```bash
git clone https://github.com/bikas-basnet/NepAgNeuroVision.git
cd NepAgNeuroVision

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
