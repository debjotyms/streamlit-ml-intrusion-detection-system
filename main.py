import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('rf.sav')

# Define the list of features
features = ['service', 'flag', 'src_bytes', 'dst_bytes', 'count', 'same_srv_rate',
            'diff_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_same_src_port_rate']

# Function to predict class
def predict_class(input_values):
    input_dict = dict(zip(features, input_values))
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)
    return prediction[0]

# Main Streamlit app
def main():
    st.markdown("""
        <a href="https://github.com/debjotyms/Intrusion-Detection-Streamlit-CSE484" style="text-decoration: none; display: inline-block; padding: 8px 16px; background-color: #FF4B4B; color: white; font-size: 16px; cursor: pointer; text-align: center; border: none; border-radius: 7px; transition: background-color 0.3s ease, transform 0.3s ease;" onmouseover="this.style.backgroundColor='#FF2B2B'; this.style.transform='scale(1.1)';" onmouseout="this.style.backgroundColor='#FF4B4B'; this.style.transform='scale(1.0)';">
            Visit our GitHub Repo
        </a>
    """, unsafe_allow_html=True)
    


    st.title('Network Intrusion Detection')

    # Text input box for comma-separated values
    input_values = st.text_input("Enter comma-separated values for features")

    # Predict class
    if st.button('Predict') and input_values:
        input_values = input_values.split(',')
        prediction = predict_class(input_values)
        if prediction == 0:
            prediction = 'anomaly'
        else:
            prediction = 'normal'
        st.write(f'Predicted class: {prediction}')

if __name__ == "__main__":
    main()
