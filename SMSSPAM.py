import streamlit as st
import joblib

st.set_page_config(page_title="SMS Spam Classifier")
st.title("📩 SMS Spam Classifier")

model = joblib.load("final_spam_model.pkl")           
vectorizer = joblib.load("tfidf_vectorizer.pkl")      

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
