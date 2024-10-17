import streamlit as st

# Embed the Flask front-end into Streamlit using an iframe
flask_url = "http://localhost:5000"  # Replace with your Flask app's URL
st.markdown(f'<iframe src="{flask_url}" width="100%" height="600"></iframe>', unsafe_allow_html=True)
