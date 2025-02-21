import streamlit as st
import pickle
import nbformat
import numpy as np
# Load the model and scaler
with open("kmeans_model.pkl", "rb") as f:
    model = pickle.load(f)
    

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Segment mapping
segment_map = {
    0: 'Frequent Buyers',
    1: 'Low Engagement',
    2: 'New or Infrequent'
}

# Streamlit UI
st.title("Customer Segmentation using K-Means By Phinphin")
st.write("Enter the RFM values to predict the customer cluster.")

# Input fields for RFM values
recency = st.number_input("Recency", min_value=0.0, step=1.0)
frequency = st.number_input("Frequency", min_value=0.0, step=1.0)
monetary = st.number_input("Monetary", min_value=0.0, step=1.0)

if st.button("Predict Cluster"):
    try:
        # Prepare input for prediction
        rfm_values = np.array([[recency, frequency, monetary]])
        rfm_scaled = scaler.transform(rfm_values)
        
        # Make prediction
        cluster = model.predict(rfm_scaled)
        
        # Display result
        cluster_label = segment_map.get(int(cluster[0]), "Unknown Cluster")
        st.success(f"The predicted cluster is: {cluster_label}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
from nbconvert import HTMLExporter

def display_notebook(ipynb_path):
    # Read and load the notebook content using nbformat
    with open(ipynb_path, 'r') as f:
        notebook_content = nbformat.read(f, as_version=4)

    # Convert .ipynb to HTML
    html_exporter = HTMLExporter()
    body, resources = html_exporter.from_notebook_node(notebook_content)
    
    # Display in Streamlit
    st.markdown(body, unsafe_allow_html=True)

# Streamlit app
def app():
    st.title("Display Jupyter Notebook in Streamlit")
    ipynb_path = 'GROUP 7 REAL FINAL.ipynb'  # Replace with your file path
    display_notebook(ipynb_path)

if __name__ == "__main__":
    app()
