import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="E-Commerce Refund Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    section[data-testid="stSidebar"] > div:first-child {
        position: sticky;
        top: 0;
        height: 100vh;
        overflow-y: hidden;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1f1f1f 0%, #262b36 100%);
        padding: 1.75rem;
        border-radius: 18px;
        margin: 1rem 0 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 20px 35px rgba(0,0,0,0.35);
        color: #f7f9fc;
        font-family: 'Inter', sans-serif;
    }
    .prediction-box .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.95rem;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        letter-spacing: 0.04em;
    }
    .prediction-box .status-pill.refund {
        background: rgba(229, 57, 53, 0.15);
        color: #ff8a80;
    }
    .prediction-box .status-pill.safe {
        background: rgba(76, 175, 80, 0.15);
        color: #69f0ae;
    }
    .prediction-box .prob-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(120px, 1fr));
        gap: 1.25rem;
        margin-top: 1.5rem;
    }
    .prediction-box .prob-card {
        background: rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        border: 1px solid rgba(255,255,255,0.07);
    }
    .prediction-box .prob-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.65);
    }
    .prediction-box .prob-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.25rem 0;
    }
    .prediction-box .delta {
        font-size: 0.85rem;
        color: #90caf9;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .info-card {
        background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .risk-chip {
        display: inline-block;
        padding: 0.1rem 0.65rem;
        border-radius: 999px;
        background-color: #e3f2fd;
        color: #0d47a1;
        font-size: 0.85rem;
        margin-left: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and preprocessor"""
    try:
        with open('models/adaboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('models/feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        return model, preprocessor, feature_info
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please run the notebook cell to save the model first. Error: {e}")
        return None, None, None

def predict_refund(model, preprocessor, input_data):
    """Make prediction using the model"""
    try:
        # Preprocess the input
        input_processed = preprocessor.transform(input_data)
        # Make prediction
        prediction = model.predict(input_processed)[0]
        prediction_proba = model.predict_proba(input_processed)[0]
        return prediction, prediction_proba
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def main():
    # Header
    st.markdown('<p class="main-header">🛒 E-Commerce Refund Prediction System</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model, preprocessor, feature_info = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application predicts whether a customer will request a refund for their e-commerce order.
        
        **Model:** AdaBoost Classifier
        
        **Features Used:**
        - Platform
        - Delivery Time (Minutes)
        - Product Category
        - Order Value (INR)
        """)
        
        st.markdown("---")
        st.header("📊 Model Performance")
        st.metric("Accuracy", "54.18%")
        st.metric("Precision", "50.00%")
        st.metric("Recall", "0.02%")
        st.metric("F1 Score", "0.04%")
        st.caption("Recall is intentionally low due to highly imbalanced responses.")

        st.markdown("---")
        st.subheader("🧪 Playbooks")
        st.markdown("""
        - Reduce delivery delays for top value orders
        - Monitor snacks & beverages category closely
        - Prioritize premium users on Swiggy Instamart
        """)

        st.markdown("---")
        with st.expander("How predictions are made"):
            st.write("""
            1. Inputs are scaled / one-hot encoded using the saved preprocessor.
            2. AdaBoost combines 100 weak trees to vote on refund risk.
            3. Probabilities are derived from the final estimator's confidence.
            """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Order Details")
        st.caption("Tip: Fill the form or click a quick preset to explore scenarios.")

        preset_col1, preset_col2, preset_col3 = st.columns(3)
        with preset_col1:
            if st.button("⚡ Fast Grocery"):
                st.session_state.update({
                    "platform_default": "Blinkit",
                    "category_default": "Grocery",
                    "delivery_default": 25,
                    "order_value_default": 650
                })
        with preset_col2:
            if st.button("🎯 Premium Order"):
                st.session_state.update({
                    "platform_default": "Swiggy Instamart",
                    "category_default": "Personal Care",
                    "delivery_default": 55,
                    "order_value_default": 3200
                })
        with preset_col3:
            if st.button("🥤 Late Beverage"):
                st.session_state.update({
                    "platform_default": "JioMart",
                    "category_default": "Beverages",
                    "delivery_default": 95,
                    "order_value_default": 780
                })

        platform_default = st.session_state.get("platform_default", "Swiggy Instamart")
        category_default = st.session_state.get("category_default", "Dairy")
        delivery_default = st.session_state.get("delivery_default", 30)
        order_value_default = st.session_state.get("order_value_default", 500)
        
        # Input form
        with st.form("prediction_form"):
            # Platform selection
            platform = st.selectbox(
                "Platform",
                options=["Swiggy Instamart", "Blinkit", "JioMart"],
                help="Select the delivery platform",
                index=["Swiggy Instamart", "Blinkit", "JioMart"].index(platform_default)
            )
            
            # Product Category
            product_category = st.selectbox(
                "Product Category",
                options=["Dairy", "Grocery", "Snacks", "Fruits & Vegetables", "Beverages", "Personal Care"],
                help="Select the product category",
                index=["Dairy", "Grocery", "Snacks", "Fruits & Vegetables", "Beverages", "Personal Care"].index(category_default)
            )
            
            # Delivery Time
            delivery_time = st.number_input(
                "Delivery Time (Minutes)",
                min_value=0,
                max_value=500,
                value=delivery_default,
                step=1,
                help="Enter the delivery time in minutes"
            )
            
            # Order Value
            order_value = st.number_input(
                "Order Value (INR)",
                min_value=0,
                max_value=100000,
                value=order_value_default,
                step=50,
                help="Enter the order value in Indian Rupees"
            )
            
            # Submit button
            submitted = st.form_submit_button("🔮 Predict Refund Probability", use_container_width=True)
    
    with col2:
        st.header("📈 Prediction Result")
        st.caption("Results update as soon as you submit the form.")

        if submitted:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Platform': [platform],
                'Delivery Time (Minutes)': [delivery_time],
                'Product Category': [product_category],
                'Order Value (INR)': [order_value]
            })
            
            # Make prediction
            prediction, prediction_proba = predict_refund(model, preprocessor, input_data)
            
            if prediction is not None:
                # Display prediction
                refund_prob = prediction_proba[1] * 100  # Probability of refund (Yes)
                no_refund_prob = prediction_proba[0] * 100  # Probability of no refund (No)
                risk_label = "High" if refund_prob > 50 else "Moderate" if refund_prob > 30 else "Low"
                
                delta_value = refund_prob - no_refund_prob if prediction == 1 else no_refund_prob - refund_prob
                status_class = "refund" if prediction == 1 else "safe"
                status_text = "Refund Likely" if prediction == 1 else "No Refund Expected"
                status_icon = "⚠️" if prediction == 1 else "✅"

                prediction_card = f"""
                <div class="prediction-box">
                    <div class="status-pill {status_class}">
                        <span>{status_icon}</span>
                        <strong>{status_text}</strong>
                    </div>
                    <div class="prob-grid">
                        <div class="prob-card">
                            <div class="prob-label">Refund Probability</div>
                            <div class="prob-value">{refund_prob:.2f}%</div>
                            <div class="delta">Δ {delta_value:+.2f}% vs no-refund</div>
                        </div>
                        <div class="prob-card">
                            <div class="prob-label">No Refund Probability</div>
                            <div class="prob-value">{no_refund_prob:.2f}%</div>
                            <div class="delta">Δ {-delta_value:+.2f}% vs refund</div>
                        </div>
                    </div>
                    <div style="margin-top:1.5rem;font-size:0.95rem;">
                        <strong>Risk Level:</strong> {risk_label}<span class='risk-chip'>{risk_label}</span>
                    </div>
                </div>
                """
                st.markdown(prediction_card, unsafe_allow_html=True)

                tabs = st.tabs(["Risk Overview", "Order Summary", "Recommendations"])
                with tabs[0]:
                    st.markdown("### Probability Distribution")
                    prob_df = pd.DataFrame(
                        {"Probability": [no_refund_prob, refund_prob]},
                        index=["No Refund", "Refund"]
                    )
                    st.bar_chart(prob_df)
                    st.caption(f"Refund: {refund_prob:.2f}% | No Refund: {no_refund_prob:.2f}%")

                with tabs[1]:
                    summary_data = {
                        "Platform": [platform],
                        "Product Category": [product_category],
                        "Delivery Time (Minutes)": [delivery_time],
                        "Order Value (INR)": [order_value],
                        "Predicted Refund": ["Yes" if prediction == 1 else "No"],
                        "Refund Probability": [f"{refund_prob:.2f}%"]
                    }
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

                with tabs[2]:
                    st.markdown("### 💡 Recommended Actions")
                    suggestions = []
                    if delivery_time > 70:
                        suggestions.append("Consider offering express delivery coupons for delayed orders.")
                    if order_value > 4000:
                        suggestions.append("Flag for manual review due to high order value exposure.")
                    if product_category in ["Beverages", "Snacks"]:
                        suggestions.append("Verify packaging quality; these categories have higher spill damage claims.")
                    if not suggestions:
                        suggestions.append("No critical follow-ups. Continue normal SLA.")
                    for item in suggestions:
                        st.write(f"- {item}")
    
    # Additional information section
    st.markdown("---")
    st.header("📋 Order Summary")
    
    if submitted and prediction is not None:
        with st.expander("Detailed Prediction Payload", expanded=False):
            st.json({
                "Inputs": input_data.iloc[0].to_dict(),
                "Probabilities": {"No Refund": f"{no_refund_prob:.2f}%", "Refund": f"{refund_prob:.2f}%"},
                "Risk Level": risk_label
            })
        st.info("Need to store or audit predictions? Export this section or connect to your ops system.")

if __name__ == "__main__":
    main()

