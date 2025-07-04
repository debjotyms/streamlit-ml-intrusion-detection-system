<div align="center">

# Network Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A machine learning-powered web application for real-time network intrusion detection. Built with Streamlit and Random Forest classifier.
</div>

## Features

- **Real-time Detection**: Instant classification of network traffic as normal or anomalous
- **User-friendly Interface**: Clean web interface with separate input fields and real-time validation
- **Visual Results**: Confidence scores, probability breakdown, and color-coded alerts
- **Easy Setup**: One-click model retraining and preprocessing included

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone and navigate:**
   ```bash
   git clone https://github.com/yourusername/streamlit-ml-intrusion-detection-system
   cd streamlit-ml-intrusion-detection-system
   ```

2. **Setup environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model:**
   ```bash
   python train_model.py
   ```

5. **Run the application:**
   ```bash
   streamlit run main.py
   ```

6. **Open browser:** http://localhost:8501


## Model Performance

- **Accuracy**: 99.68%
- **Algorithm**: Random Forest (100 estimators)
- **Features**: 10 carefully selected network traffic features
- **Preprocessing**: SMOTE balancing, StandardScaler normalization
- **Training Data**: NSL-KDD intrusion detection dataset

### Why High Accuracy?
- Synthetic dataset 
- Strong discriminative features
- Effective ensemble learning approach

### Model Files
The application uses 3 essential model files:
- `rf_new.sav` - Trained Random Forest classifier (2.7MB)
- `scaler.sav` - Feature scaler for normalization (1.3KB)
- `label_encoders.sav` - Categorical feature encoders (1.6KB)


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>Star this repo if you found it helpful!</p>
</div>
