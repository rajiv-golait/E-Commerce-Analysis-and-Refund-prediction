# 🛒 E-Commerce Refund Prediction Project

A comprehensive machine learning project that analyzes e-commerce delivery data and predicts whether customers will request refunds for their orders. The project includes data analysis, visualization, multiple ML model comparisons, and an interactive Streamlit web application.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project analyzes e-commerce delivery data from platforms like Swiggy Instamart, Blinkit, and JioMart to predict refund requests. The project includes:

- **Data Analysis**: Comprehensive exploratory data analysis with visualizations
- **Model Training**: Comparison of 11 different machine learning models
- **Best Model Selection**: AdaBoost Classifier selected as the best performing model
- **Web Application**: Interactive Streamlit app for real-time refund predictions

## ✨ Features

### Data Analysis & Visualization
- Distribution analysis of refund requests
- Platform-wise refund comparison
- Delivery time analysis with outlier removal
- Service rating patterns
- Correlation heatmaps
- Order value distributions
- Pairwise relationship analysis

### Machine Learning Models
The project compares 11 different models:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. AdaBoost Classifier
7. **AdaBoost Classifier** (Best Model)
8. Gradient Boosting Classifier
9. Bagging Classifier
10. Voting Classifier
11. Stacking Classifier

### Streamlit Web Application
- Interactive prediction interface
- Real-time refund probability calculation
- Visual probability distribution
- Risk assessment insights
- Order summary display
- Scenario presets, sticky sidebar KPIs, and premium prediction cards for a dashboard-like UX

## 📊 Dataset

The dataset contains **100,000 e-commerce orders** with the following features:

- **Order ID**: Unique order identifier
- **Customer ID**: Unique customer identifier
- **Platform**: Delivery platform (Swiggy Instamart, Blinkit, JioMart)
- **Order Date & Time**: Timestamp of the order
- **Delivery Time (Minutes)**: Time taken for delivery
- **Product Category**: Category of the product
- **Order Value (INR)**: Order value in Indian Rupees
- **Customer Feedback**: Customer feedback text
- **Service Rating**: Rating given by the customer
- **Delivery Delay**: Whether delivery was delayed
- **Refund Requested**: Target variable (Yes/No)

### Features Used for Modeling
After data leakage analysis, the following features were used:
- Platform (Categorical)
- Delivery Time (Minutes) (Numerical)
- Product Category (Categorical)
- Order Value (INR) (Numerical)

## 📁 Project Structure

```
CI Project/
│
├── E-Commerce Analysis and refund prediction.ipynb  # Main Jupyter notebook
├── streamlit_app.py                                 # Streamlit web application
├── requirements.txt                                  # Python dependencies
├── README.md                                        # This file
├── README_STREAMLIT.md                              # Streamlit-specific README
├── .gitignore                                       # Git ignore file
│
├── models/                                          # Model files (generated)
│   ├── adaboost_model.pkl                          # Trained model
│   ├── preprocessor.pkl                            # Data preprocessor
│   └── feature_info.pkl                            # Feature information
│
└── e-commerce-analytics-swiggy-zomato-blinkit/     # Dataset directory
    └── Ecommerce_Delivery_Analytics_New.csv        # Dataset file
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- pip (Python package manager)

### Step 1: Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd "CI Project"

# Or download and extract the project folder
```

### Step 2: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset
The dataset is automatically downloaded from Kaggle when you run Cell 2 in the notebook. Alternatively, you can manually place the CSV file in the appropriate directory.

## 📖 Usage

### Running the Jupyter Notebook

1. **Open the Notebook**:
   ```bash
   jupyter notebook "E-Commerce Analysis and refund prediction.ipynb"
   ```

2. **Run Cells in Order**:
   - Cells 1-8: Data loading and preprocessing
   - Cells 9-27: Data visualization
   - Cells 29-34: Model setup and data preparation
   - Cells 38-60: Model training (11 different models)
   - Cell 61: Model comparison
   - Cell 62: Display results
   - **Cell 53**: Save model files (IMPORTANT: Run this before using Streamlit app)

### Running the Streamlit Application

1. **Ensure Model Files Exist**:
   - Run Cell 53 in the notebook to generate model files
   - Verify that `models/` directory contains the `.pkl` files

2. **Start the Streamlit App**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Use the Application**:
   - The app will open in your browser at `http://localhost:8501`
   - Enter order details in the form
   - Click "Predict Refund Probability"
   - View predictions and insights

## 📈 Model Performance

### Best Model: AdaBoost Classifier

| Metric | Value |
|--------|-------|
| **Accuracy** | 54.18% |
| **Precision** | 50.00% |
| **Recall** | 0.02% |
| **F1 Score** | 0.04% |
| **ROC-AUC** | 50.00% |

### All Models Comparison

| Rank | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
| 1 | AdaBoost | 54.18% | 50.00% | 0.02% | 0.04% | 50.00% |
| 2 | Gradient Boost | 54.07% | 42.95% | 0.73% | 1.44% | 49.95% |
| 3 | Decision Tree | 51.71% | 45.37% | 26.44% | 33.41% | 49.76% |
| 4 | Stacking | 51.07% | 45.38% | 33.39% | 38.47% | 49.70% |
| 5 | KNN | 50.48% | 45.62% | 42.05% | 43.76% | 49.83% |
| 6 | Logistic Regression | 50.19% | 45.88% | 48.52% | 47.16% | 50.06% |
| 7 | SVM (Subset) | 50.17% | 46.31% | 54.86% | 50.22% | 50.53% |
| 8 | Bagging | 50.15% | 45.79% | 47.89% | 46.82% | 49.97% |
| 9 | Random Forest | 49.59% | 45.26% | 47.83% | 46.51% | 49.45% |
| 10 | Voting | 49.53% | 45.34% | 49.41% | 47.29% | 49.52% |
| 11 | XGBoost | 49.38% | 45.23% | 49.72% | 47.37% | 49.40% |

## 🧠 Streamlit Experience Highlights

- **One-Click Scenarios** – Instantly explore Fast Grocery, Premium Order, or Late Beverage presets to see how the risk shifts.
- **Always-On Insights** – Sidebar stays fixed with KPIs, playbooks, and an explainer of how the AdaBoost model works.
- **Premium Prediction Card** – Gradient card with dual probability tiles, deltas, and risk chips for high readability in dark mode.
- **Actionable Tabs** – Risk overview charts, structured order summary, recommendations, and JSON payload export for audits.

## 🛠️ Technologies Used

### Data Science & Machine Learning
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and preprocessing
- **xgboost**: Gradient boosting framework

### Visualization
- **matplotlib**: Static plotting
- **seaborn**: Statistical data visualization

### Web Application
- **streamlit**: Interactive web app framework

### Data Source
- **opendatasets**: Kaggle dataset download utility

## 📸 Screenshots

### Notebook Analysis
- Data distribution visualizations
- Correlation heatmaps
- Model performance comparisons

### Streamlit App
- Interactive prediction interface
- Real-time probability visualization
- Risk assessment dashboard

## 🔍 Key Insights

1. **Data Leakage Prevention**: Removed features like 'Service Rating', 'Delivery Delay', 'Customer Feedback', and 'Order Date & Time' to prevent data leakage.

2. **Model Selection**: AdaBoost Classifier achieved the highest accuracy, though all models show relatively balanced performance near 50%, suggesting the remaining features have limited predictive power.

3. **Feature Importance**: The four features used (Platform, Delivery Time, Product Category, Order Value) provide baseline predictions, but additional feature engineering could improve performance.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Some areas for improvement:

- Feature engineering
- Hyperparameter tuning
- Additional model architectures
- UI/UX improvements for the Streamlit app
- Performance optimization

## 📝 Notes

- The dataset is downloaded from Kaggle and requires appropriate credentials
- Model files are generated when running Cell 53 in the notebook
- The Streamlit app requires the model files to be present in the `models/` directory
- All paths in the notebook are configured for Windows; adjust if using other operating systems

## 📄 License

This project is open source and available for educational and research purposes.

## 👤 Author

Created as part of a CI (Continuous Integration) project for e-commerce analytics and refund prediction.

## 🙏 Acknowledgments

- Dataset: [E-commerce Analytics - Swiggy, Zomato, Blinkit](https://www.kaggle.com/datasets/logiccraftbyhimanshi/e-commerce-analytics-swiggy-zomato-blinkit)
- Libraries and frameworks used in this project

---

**Last Updated**: 2024

For detailed Streamlit app instructions, see [README_STREAMLIT.md](README_STREAMLIT.md)

