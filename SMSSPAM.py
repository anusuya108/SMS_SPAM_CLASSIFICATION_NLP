import streamlit as st
import joblib

# Load model + vectorizer
model = joblib.load("final_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")

st.title(" SMS Spam Classifier (NLP Project)")
st.write("Enter an SMS message and predict whether it is **SPAM** or **HAM**.")

msg = st.text_area("Enter your SMS message:")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning(" Please enter a message.")
    else:
        msg_vec = vectorizer.transform([msg])   
        pred = model.predict(msg_vec)[0]        
        if pred == 1:
            st.error(" SPAM Detected")
        else:
            st.success(" HAM (Not Spam)")
