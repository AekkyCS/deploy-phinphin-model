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
import streamlit as st
import nbformat
from nbconvert import HTMLExporter

import streamlit as st
import nbformat



def display_code_only(ipynb_path):
    # อ่านและโหลดเนื้อหาจาก Notebook
    with open(ipynb_path, 'r') as f:
        notebook_content = nbformat.read(f, as_version=4)

    # ตรวจสอบว่า session_state มีตัวแปร show_code หรือไม่ ถ้ายังไม่มีให้กำหนดค่าเริ่มต้น
    if 'show_code' not in st.session_state:
        st.session_state.show_code = False  # กำหนดค่าเริ่มต้นเป็น False (ไม่แสดงโค้ด)

    # สร้างปุ่มเพื่อควบคุมการแสดงโค้ด
    button_label = "Hide Code" if st.session_state.show_code else "Show Code"
    
    if st.button(button_label):
        st.session_state.show_code = not st.session_state.show_code  # เปลี่ยนสถานะเมื่อกดปุ่ม

    # แสดงโค้ดถ้าสถานะ show_code เป็น True
    if st.session_state.show_code:
        for cell in notebook_content.cells:
            if cell.cell_type == 'code':
                code = cell.source  # ดึงโค้ดมาโดยไม่ต้องรวมบรรทัด
                st.code(code, language='python')  # แสดงโค้ดใน Streamlit

# Streamlit app
def app():
    st.title("Display Jupyter Notebook Code Only in Streamlit")
    ipynb_path = 'GROUP 7 REAL FINAL.ipynb'  # เปลี่ยนเป็น path ของไฟล์คุณ
    display_code_only(ipynb_path)

if __name__ == "__main__":
    app()
