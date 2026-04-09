import streamlit as st
import json
import pandas as pd
import time

# Page Configuration
st.set_page_config(page_title="Llama-3 Fraud Guard (Demo Mode)", page_icon="🛡️", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #00ff7f;
        color: black;
        font-weight: bold;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🛡️ Fraud Guard")
    st.info("Demo Mode: Model Mocked")
    st.metric("Model", "Llama-3-8B")
    st.metric("Inference Engine", "Mock/Standard")
    st.write("---")
    st.caption("Developed by Shivam Nayak")

# Main Header
st.title("🛡️ Llama-3 Domain-Specific Fraud Detection")
st.write("Analyze transactions using fine-tuned Large Language Models.")

# Transaction Input
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Transaction Details")
    tx_input = st.text_area(
        "Paste transaction JSON or text description here:",
        height=200,
        placeholder="e.g. Transaction amount 500$, ID 1234, Device Mobile, Location NY..."
    )
    
    analyze_btn = st.button("🚀 Analyze Transaction")

with col2:
    st.subheader("System Status")
    st.success("API: Operational")
    st.success("GPU: Simulated")
    st.warning("Hardware: Laptop Standard")
    st.write("---")
    compare_mode = st.checkbox("🔄 Compare with Base Model")

# Analysis Logic (Simulated)
if analyze_btn:
    if tx_input:
        with st.spinner("Analyzing patterns with Ensemble Models..."):
            time.sleep(2) # Simulating LLM Latency
            
            # Mock Logic for Demo
            is_fraud = "fraud" in tx_input.lower() or "10000" in tx_input
            
            if compare_mode:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Base Llama-3-8B")
                    st.info("Prediction: Legitimate")
                    st.caption("Reason: Amount is within typical global limits.")
                with c2:
                    st.subheader("Fine-Tuned (Fraud Guard)")
                    if is_fraud:
                        st.error("Prediction: Fraud Alert!")
                        st.caption("Reason: High-stakes pattern detected for Nigerian IP branch.")
                    else:
                        st.success("Prediction: Legitimate")
                        st.caption("Reason: Verified Seattle Starbucks pattern.")
            else:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                if is_fraud:
                    st.error("🚨 ALERT: Fraudulent Activity Detected!")
                    st.write("**Confidence Score:** 98.4%")
                    st.write("**Reasoning:** High-value transaction from an unverified device coupled with suspicious geolocational patterns.")
                else:
                    st.success("✅ Transaction Verified: Legitimate")
                    st.write("**Confidence Score:** 94.2%")
                    st.write("**Reasoning:** Behavior matches historical patterns for this user ID.")
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter some transaction data first.")

# Footer Analytics
st.write("---")
st.subheader("Historical Batch Analytics")
mock_data = pd.DataFrame({
    'Category': ['Legitimate', 'Fraudulent', 'Review Needed'],
    'Count': [450, 45, 12]
})
st.bar_chart(mock_data.set_index('Category'))
