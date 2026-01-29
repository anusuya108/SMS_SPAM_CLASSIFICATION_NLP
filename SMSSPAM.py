import streamlit as st
import joblib

st.set_page_config(page_title="SMS Spam Classifier")

st.title(" SMS Spam Classifier")

#  Load PIPELINE model (TF-IDF + Classifier inside)
model = joblib.load("final_spam_model.pkl")

msg = st.text_area("Enter your SMS message:")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning(" Please enter a message.")
    else:
        #  Pass raw text directly (pipeline will vectorize internally)
        pred = model.predict([msg])[0]

        if pred == 1:
            st.error(" SPAM Detected")
        else:
            st.success(" HAM (Not Spam)")
