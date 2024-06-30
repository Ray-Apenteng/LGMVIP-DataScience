# import all yout dependencies
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from PIL import Image
import tempfile


# Load the model
json_file = open('cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("cnn_model.weights.h5")

# Define the symbol labels
symbol_labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                 10: '-', 11: '+', 12: '*'}

# Preprocess the input image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = ~img
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    ctrs, ret = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    symbols = []
    for c in cnt:
        x, y, w, h = cv2.boundingRect(c)
        im_crop = thresh[y:y+h+10, x:x+w+10]
        im_resize = cv2.resize(im_crop, (28, 28))
        im_resize = np.reshape(im_resize, (1, 28, 28, 1))
        im_resize = im_resize / 255.0
        symbols.append(im_resize)
    
    return symbols

# Streamlit app
st.title("Handwritten Equation Solver")

st.write("Draw an equation below and click 'Calculate'")

canvas_result = st_canvas(stroke_width=2, stroke_color="#000000", background_color="#FFFFFF", width=400, height=200)

if st.button("Calculate"):
    if canvas_result.image_data is not None:
        # Save the drawn image
        img = canvas_result.image_data
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
            image_path = temp.name
            cv2.imwrite(image_path, img)
        
        # Preprocess and predict
        symbols = preprocess_image(image_path)
        predicted_symbols = []
        for symbol in symbols:
            prediction = loaded_model.predict(symbol)
            predicted_label = np.argmax(prediction)
            predicted_symbols.append(symbol_labels[predicted_label])
        
        # Combine the predicted symbols to form the equation
        equation = ''.join(predicted_symbols)
        #st.write(f"Predicted Equation: {equation}")

        # Evaluate the equation
        try:
            result = eval(equation)
            st.write(f"Result: {result}")
        except Exception as e:
            st.write(f"Error evaluating the equation: {e}")

# Make sure to have the below line at the end of your script
st.stop()
