import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and preprocessing components
try:
    model = joblib.load('rf.sav')
    scaler = joblib.load('scaler.sav')
    label_encoders = joblib.load('label_encoders.sav')
    target_encoder = joblib.load('target_encoder.sav')
    st.success("✅ Model and encoders loaded successfully!")
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e}")
    st.info("Please run 'python retrain_model.py' first to create the model files.")
    st.stop()

# Define the list of features
features = ['service', 'flag', 'src_bytes', 'dst_bytes', 'count', 'same_srv_rate',
            'diff_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_same_src_port_rate']

# Function to predict class
def predict_class(input_values):
    try:
        # Create DataFrame with feature names
        input_dict = dict(zip(features, input_values))
        input_df = pd.DataFrame([input_dict])
        
        # Encode categorical features
        categorical_features = ['service', 'flag']
        for feature in categorical_features:
            if feature in input_df.columns and feature in label_encoders:
                # Convert to string to handle any data type
                input_df[feature] = input_df[feature].astype(str)
                # Transform using the saved encoder
                input_df[feature] = label_encoders[feature].transform(input_df[feature])
        
        # Convert all features to numeric
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        
        # Fill any NaN values with 0
        input_df.fillna(0, inplace=True)
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        return prediction[0], probability[0]
    
    except ValueError as e:
        st.error(f"❌ Invalid input: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None

# Main Streamlit app
def main():
    
    st.markdown("<h1 style='text-align: center; color: #1f77b4;'>Network Intrusion Detection System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 40px;">
        <p style="margin-bottom: 0; text-align: center;">Enter network traffic features below to detect potential intrusions. Default values are based on the most common patterns in the training dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for better layout with gap
    col1, spacer, col2 = st.columns([5, 1, 5])
    
    with col1:
        st.markdown("<h3 style='text-align: center; color: #1f77b4;'>Service & Connection Details</h3>", unsafe_allow_html=True)
        # st.subheader("Service & Connection Details")
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
        
        # Service dropdown
        service_options = list(label_encoders['service'].classes_)
        service = st.selectbox("Service", options=service_options, 
                              index=service_options.index('http') if 'http' in service_options else 0,
                              help="Network service type")
        
        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        
        # Flag dropdown
        flag_options = list(label_encoders['flag'].classes_)
        flag = st.selectbox("Flag", options=flag_options,
                           index=flag_options.index('SF') if 'SF' in flag_options else 0,
                           help="Connection status flag")
        
        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        
        # Bytes transferred
        src_bytes = st.number_input("Source Bytes", value=0, min_value=0, max_value=381709090, 
                                   help="Number of bytes from source to destination")
        
        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        
        dst_bytes = st.number_input("Destination Bytes", value=0, min_value=0, max_value=5151385,
                                   help="Number of bytes from destination to source")
        
        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        
        count = st.number_input("Count", value=1, min_value=1, max_value=511,
                               help="Number of connections to same host as current connection")
        
        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

        dst_host_srv_count = st.number_input("Destination Host Service Count", value=255, min_value=0, max_value=255,
                                    help="Number of connections having same destination host and service")

    with col2:
        st.markdown("<h3 style='text-align: center; color: #1f77b4;'>Traffic Rate Statistics</h3>", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 55px;'></div>", unsafe_allow_html=True)
        
        same_srv_rate = st.slider("Same Service Rate", min_value=0.0, max_value=1.0, value=1.0, step=0.01,
                                 help="% of connections to same service")
        
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
        
        diff_srv_rate = st.slider("Different Service Rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01,
                                 help="% of connections to different services")
        
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
        
        
        dst_host_same_srv_rate = st.slider("Destination Host Same Service Rate", min_value=0.0, max_value=1.0, value=1.0, step=0.01,
                                          help="% of connections having same destination host and service")
        
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
        
        dst_host_same_src_port_rate = st.slider("Destination Host Same Source Port Rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01,
                                               help="% of connections having same destination host and source port")
    
    # Add visual separator in the spacer column
    with spacer:
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='border-left: 2px solid #e0e0e0; height: 300px; margin-left: 50%;'></div>", unsafe_allow_html=True)
    
    # Prediction section
    st.markdown("---")
    
    # Create prediction button with better styling

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button('Analyze Network Traffic', type='primary', use_container_width=True)
    
    # Predict class
    if predict_button:
        # Prepare input values
        input_values = [service, flag, src_bytes, dst_bytes, count, same_srv_rate,
                       diff_srv_rate, dst_host_srv_count, dst_host_same_srv_rate,
                       dst_host_same_src_port_rate]
        
        result = predict_class(input_values)
        
        if result[0] is not None:
            prediction, probability = result
            
            # Convert prediction to human-readable format
            if prediction == 0:
                prediction_text = 'ANOMALY'
                confidence = probability[0] * 100
                st.markdown(f"""
                <div style="background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 5px solid #f44336;">
                    <h3 style="color: #c62828; margin-top: 0;">THREAT DETECTED</h3>
                    <p style="font-size: 18px; margin-bottom: 10px;"><strong>Classification:</strong> {prediction_text}</p>
                    <p style="font-size: 16px; margin-bottom: 0;"><strong>Confidence:</strong> {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                prediction_text = 'NORMAL'
                confidence = probability[1] * 100
                st.markdown(f"""
                <div style="background-color: #e8f5e8; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;">
                    <h3 style="color: #2e7d32; margin-top: 0;">TRAFFIC NORMAL</h3>
                    <p style="font-size: 18px; margin-bottom: 10px;"><strong>Classification:</strong> {prediction_text}</p>
                    <p style="font-size: 16px; margin-bottom: 0;"><strong>Confidence:</strong> {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show probability breakdown
            st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
            st.markdown("### Probability Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Anomaly Probability", f"{probability[0]:.4f}", f"{probability[0]*100:.2f}%")
            with col2:
                st.metric("Normal Probability", f"{probability[1]:.4f}", f"{probability[1]*100:.2f}%")
    
    # Add example section
    st.markdown("---")
    with st.expander("Example Values", expanded=False):
        st.markdown("""
        **Normal Traffic Example:**
        - Service: http, Flag: SF, Source Bytes: 200, Destination Bytes: 5000
        - Count: 5, Same Service Rate: 1.0, Different Service Rate: 0.0
        - Dest Host Service Count: 255, Dest Host Same Service Rate: 1.0, Dest Host Same Source Port Rate: 0.0
        
        **Potential Anomaly Example:**
        - Service: private, Flag: S0, Source Bytes: 0, Destination Bytes: 0
        - Count: 100, Same Service Rate: 0.05, Different Service Rate: 0.95
        - Dest Host Service Count: 10, Dest Host Same Service Rate: 0.1, Dest Host Same Source Port Rate: 0.9
        """)
    
    # Add feature descriptions
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    with st.expander("Feature Descriptions", expanded=False):
        st.markdown("""
        **Service**: Type of network service (http, ftp, ssh, etc.)
        
        **Flag**: Connection status (SF=Normal, S0=Connection attempt, REJ=Rejected, etc.)
        
        **Source/Destination Bytes**: Number of data bytes transferred
        
        **Count**: Number of connections to the same host
        
        **Same Service Rate**: Percentage of connections to the same service
        
        **Different Service Rate**: Percentage of connections to different services
        
        **Destination Host Service Count**: Number of connections with same destination host and service
        
        **Destination Host Same Service Rate**: Percentage of connections with same destination host and service
        
        **Destination Host Same Source Port Rate**: Percentage of connections with same destination host and source port
        """)

if __name__ == "__main__":
    main()
