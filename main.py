import numpy as np
import streamlit as st
import keras
from keras.preprocessing import image
import io

st.title('Model Klasifikasi Makanan')

# Sidebar
with st.sidebar:
    st.header("Navigasi")
    st.write("Selamat datang di aplikasi klasifikasi makanan!")
    st.write("Silakan unggah gambar atau gunakan kamera untuk prediksi.")

# Input type - File
uploaded_file = st.sidebar.file_uploader('Unggah Gambar', type=["png", "jpg", "jpeg"])

# Input type - Image
img_file_buffer = st.sidebar.camera_input('Kamera')

st.header('Hasil')

# Model
def load_model():
    model = keras.models.load_model("final-food-model.h5")
    return model

modelPrediction = load_model()

# Classes
specific_classes =  ['baby_back_ribs','baklava','beef_carpaccio','bruschetta',\
                    'beet_salad','beignets','breakfast_burrito','donat','churros','fried_rice']

# Prediction for upload file
if uploaded_file is not None:
    # Pre processing
    img = image.load_img(io.BytesIO(uploaded_file.read()), target_size=(256, 256))
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Showing image that was uploaded
    st.image(img, caption="Gambar yang Diunggah", use_column_width=True)

    # Prediction
    predictions = modelPrediction.predict(img_array)

    # Label
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = specific_classes[predicted_class_index]

    # Print
    st.write("Kelas Prediksi:", predicted_class)


# Prediction for image from camera
if img_file_buffer is not None:
    # Pre processing
    img = image.load_img(io.BytesIO(img_file_buffer.read()), target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    predictions = modelPrediction.predict(img_array)

    # Label
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = specific_classes[predicted_class_index]

    # Print 
    st.write("Kelas Prediksi:", predicted_class)
