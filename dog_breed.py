import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Load the learned model form notebook
model = load_model('/home/ashna/jest/rgb/dog_breedd (1).h5')

# Name of the dog breeds classes
dog_classes = ['Golden Retriever','Bermaise','Pug','Chinese Crested','Siberian Husky', 'Shih-Tzu', 'Cockapoo']

# Setting the title of the APP
st.title('Dog Breed Prediction')
st.markdown('Upload an image of your dog')

# Read the uploaded image
dog_image = st.file_uploader('Choose an image ...', type='jpg')
submit = st.button('Predict')

# When the submit button is clicked
if submit:
    # Convert the uploaded file to an opencv image
    file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Displaying the image
    st.image(opencv_image, channels='BGR')
    # Resizing the image
    opencv_image = cv2.resize(opencv_image, (224, 224))
    # Convert image to 4 Dimenstion
    opencv_image.shape = (1, 224, 224, 3)
    # Predict the uploaded image
    Y_pred = model.predict(opencv_image)

    # show the prediction result on the page
    st.title(
        str(f'The uploaded dog breed is {dog_classes[np.argmax(Y_pred)]}'))