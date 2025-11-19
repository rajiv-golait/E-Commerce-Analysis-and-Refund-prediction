# E-Commerce Refund Prediction - Streamlit App

This Streamlit application provides an interactive interface for predicting refund requests in e-commerce orders.

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- Trained model files (generated from the notebook)

### Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Make sure you have run the notebook cell (Cell 53) to generate the model files:
   - `models/adaboost_model.pkl`
   - `models/preprocessor.pkl`
   - `models/feature_info.pkl`

### Running the App

Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## 📋 Features

- **Interactive Prediction Interface**: Enter order details and get instant refund predictions
- **Probability Visualization**: See the probability distribution for refund vs no-refund
- **Risk Assessment**: Get insights based on order characteristics
- **Model Performance Metrics**: View model accuracy and performance in the sidebar

## 🎯 How to Use

1. **Enter Order Details**:
   - Select the delivery platform (Swiggy Instamart, Blinkit, or JioMart)
   - Choose the product category
   - Enter delivery time in minutes
   - Enter order value in INR

2. **Get Prediction**:
   - Click the "Predict Refund Probability" button
   - View the prediction result and probability percentages
   - Check the insights section for risk assessment

3. **Review Summary**:
   - See a complete order summary table at the bottom

## 📊 Model Information

- **Model Type**: AdaBoost Classifier
- **Accuracy**: 54.18%
- **Features Used**: Platform, Delivery Time, Product Category, Order Value

## 📁 Project Structure

```
.
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── models/                   # Model files (generated from notebook)
│   ├── adaboost_model.pkl
│   ├── preprocessor.pkl
│   └── feature_info.pkl
└── E-Commerce Analysis and refund prediction.ipynb  # Training notebook
```

## ⚠️ Important Notes

- Make sure to run Cell 53 in the notebook before using the Streamlit app
- The model files must be in the `models/` directory
- The app requires all dependencies from `requirements.txt`

## 🔧 Troubleshooting

**Error: Model files not found**
- Solution: Run Cell 53 in the notebook to generate the model files

**Error: Module not found**
- Solution: Install all requirements using `pip install -r requirements.txt`

**App not starting**
- Solution: Make sure Streamlit is installed and you're using the correct command: `streamlit run streamlit_app.py`

