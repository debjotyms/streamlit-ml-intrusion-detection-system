<div align="center">

# Network Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33.0-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458?style=flat-square&logo=pandas)

A machine learning-powered web application for detecting network intrusions and anomalies in real-time. Built with Streamlit and powered by Random Forest classifier, this system helps identify potential security threats in network traffic.

</div>

## Features

### Core Functionality
- **Real-time Intrusion Detection**: Input network traffic features and get instant classification
- **Binary Classification**: Distinguishes between 'normal' and 'anomaly' network traffic
- **User-friendly Interface**: Clean, intuitive web interface built with Streamlit
- **Pre-trained Model**: Uses a Random Forest classifier trained on network intrusion data

### Technical Features
- **Feature Engineering**: Utilizes 10 most important network traffic features
- **Data Preprocessing**: Includes correlation analysis, feature selection, and SMOTE resampling
- **Model Performance**: Optimized using LazyClassifier for best algorithm selection
- **Scalable Architecture**: Easy to extend with additional features or models

## Prerequisites

Before setting up the project, ensure you have the following installed:

### 1. Install Python 3.8+

**Windows:**
- Download Python from [https://www.python.org/downloads/](https://www.python.org/downloads/)
- Run the installer and make sure to check "Add Python to PATH"
- Verify installation: Open Command Prompt and run `python --version`

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**macOS:**
```bash
# Using Homebrew
brew install python
```

### 2. Install Git

**Windows:**
- Download Git from [https://git-scm.com/download/win](https://git-scm.com/download/win)
- Run the installer and follow the setup wizard

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install git
```

**macOS:**
```bash
# Using Homebrew
brew install git
```

### 3. Verify Installation

```bash
python --version
pip --version
git --version
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/streamlit-ml-intrusion-detection-system
   cd streamlit-ml-intrusion-detection-system
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **Linux/macOS:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application:**
   ```bash
   streamlit run main.py
   ```

6. **Access the application:**
   - Open your browser and go to [http://localhost:8501](http://localhost:8501)

## Usage

### Input Features

The system requires the following 10 network traffic features as comma-separated values:

1. **service** - Network service type (encoded as integer)
2. **flag** - Connection status flag (encoded as integer)
3. **src_bytes** - Number of bytes from source to destination
4. **dst_bytes** - Number of bytes from destination to source
5. **count** - Number of connections to same host as current connection
6. **same_srv_rate** - % of connections to same service
7. **diff_srv_rate** - % of connections to different services
8. **dst_host_srv_count** - Number of connections having same destination host and service
9. **dst_host_same_srv_rate** - % of connections having same destination host and service
10. **dst_host_same_src_port_rate** - % of connections having same destination host and source port

### Example Usage

**Input:** `1,2,100,200,5,0.5,0.2,3,0.8,0.1`

**Output:** `normal` or `anomaly`

## Model Training

The project includes a comprehensive Jupyter notebook (`rf.ipynb`) that demonstrates the entire machine learning pipeline:

### Data Preprocessing
- **Dataset Loading**: Uses network intrusion detection dataset
- **Duplicate Removal**: Eliminates redundant records
- **Feature Encoding**: Converts categorical variables to numerical
- **Correlation Analysis**: Removes highly correlated features (>70%)

### Feature Selection
- **CatBoost Feature Importance**: Identifies most relevant features
- **Random Forest Importance**: Validates feature selection
- **Top 10 Features**: Selects most predictive features for model

### Model Training
- **Data Resampling**: Uses SMOTE to handle class imbalance
- **Train-Test Split**: 70-30 split for model validation
- **Scaling**: StandardScaler for feature normalization
- **Model Comparison**: LazyClassifier for algorithm selection
- **Final Model**: Random Forest with 100 estimators

### Model Evaluation
- **Cross-validation**: 5-fold CV with ROC-AUC scoring
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **ROC Curve**: Visual performance assessment

## Model Accuracy & Future Improvements

### Understanding High Accuracy Results

The Random Forest model achieves exceptionally high accuracy (often >99%) on the intrusion detection dataset. While this seems impressive, it's important to understand the underlying factors:

#### Why Such High Accuracy?

1. **Dataset Characteristics**:
   - The NSL-KDD dataset contains well-separated attack patterns
   - Network intrusion signatures are often distinct and identifiable
   - Binary classification (normal vs. anomaly) simplifies the problem

2. **Feature Engineering**:
   - Selected features have strong discriminative power
   - Categorical encoding creates clear decision boundaries
   - Correlation analysis removes redundant information

3. **Model Architecture**:
   - Random Forest's ensemble approach reduces overfitting
   - Multiple decision trees capture different aspects of intrusion patterns
   - Bootstrap aggregating provides robust predictions

#### Potential Limitations

- **Dataset Bias**: Training data may not reflect real-world network diversity
- **Temporal Drift**: Network attack patterns evolve over time
- **False Sense of Security**: High accuracy in testing doesn't guarantee real-world performance

### Future Work & Improvements

#### Enhanced Model Development

1. **Advanced Algorithms**:
   - Implement Deep Learning models (CNN, LSTM) for temporal pattern recognition
   - Explore XGBoost and LightGBM for better gradient boosting
   - Develop ensemble methods combining multiple algorithms

2. **Feature Engineering**:
   - Add time-series features for temporal attack detection
   - Implement automated feature selection using genetic algorithms
   - Include network topology and behavioral features

3. **Model Robustness**:
   - Implement adversarial training to handle evasion attacks
   - Add uncertainty quantification for prediction confidence
   - Develop online learning capabilities for real-time adaptation

#### Real-World Deployment Considerations

1. **Data Validation**:
   - Implement data drift detection mechanisms
   - Add input validation and sanitization
   - Create feedback loops for model improvement

2. **Performance Optimization**:
   - Model quantization for faster inference
   - Edge computing deployment for low-latency detection
   - Distributed processing for high-throughput scenarios

3. **Evaluation Metrics**:
   - Focus on precision to minimize false positives
   - Implement time-based evaluation windows
   - Add business-impact metrics beyond accuracy

#### Addressing Overfitting Concerns

1. **Cross-Validation**:
   - Implement time-series cross-validation
   - Use stratified sampling for better generalization
   - Add holdout test sets from different time periods

2. **Regularization**:
   - Apply dropout in neural network architectures
   - Implement early stopping mechanisms
   - Use L1/L2 regularization for feature selection

3. **Data Augmentation**:
   - Generate synthetic attack variations
   - Include adversarial examples in training
   - Collect real-world network traffic data

### Recommended Next Steps

1. **Baseline Establishment**: Create simpler models (logistic regression, SVM) for comparison
2. **Real-World Testing**: Deploy in controlled network environments
3. **Continuous Learning**: Implement systems for model updates with new attack patterns
4. **Explainability**: Add SHAP or LIME for model interpretation
5. **Security Hardening**: Protect the model itself from adversarial attacks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>⭐ Star this repo if you found it helpful!</p>
</div>
