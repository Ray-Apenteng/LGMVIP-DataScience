# Handwritten Equation Solver using CNN

This project involves using a Convolutional Neural Network (CNN) to recognize and evaluate handwritten mathematical equations. Users can draw equations on a canvas, and the application predicts the equation and calculates the result.

<img src="path/to/your/image.png" width="1000">

## Dataset

The dataset used for training the CNN model consists of images of handwritten digits and mathematical symbols. Each image is labeled with the corresponding digit or symbol.

The dataset contains the following classes:

- Digits: 0-9
- Symbols: +, -, *

## Project Steps

1. **Data Loading and Preprocessing:**

   - Load and preprocess the dataset.
   - Binarize and resize the images to a uniform size.

2. **Model Training:**

   - Train a Convolutional Neural Network (CNN) to classify the digits and symbols.

3. **Model Evaluation:**

   - Evaluate the model using accuracy as the primary metric.
   - Use validation data to ensure the robustness of the model.

4. **Streamlit App:**

   - Develop a Streamlit application where users can draw equations on a canvas.
   - Preprocess the drawn image and predict the equation using the trained CNN model.
   - Evaluate the predicted equation and display the result.

## Requirements

- Python 3.x
- pandas
- numpy
- tensorflow
- keras
- matplotlib
- opencv-python
- streamlit
- streamlit-drawable-canvas

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/handwritten-equation-solver.git
   cd handwritten-equation-solver
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Code Explanation

1. **Loading the Model:**

   ```python
   json_file = open('cnn_model.json', 'r')
   loaded_model_json = json_file.read()
   json_file.close()
   loaded_model = model_from_json(loaded_model_json)
   loaded_model.load_weights("cnn_model.weights.h5")
   ```

2. **Preprocessing the Input Image:**

   ```python
   def preprocess_image(image):
       img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
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
   ```

3. **Streamlit App:**

   ```python
   import streamlit as st
   from streamlit_drawable_canvas import st_canvas
   import cv2
   import numpy as np
   from keras.models import model_from_json
   import matplotlib.pyplot as plt

   st.title("Handwritten Equation Solver")

   canvas_result = st_canvas(
       stroke_width=2,
       stroke_color="#000000",
       background_color="#FFFFFF",
       width=400,
       height=200,
       drawing_mode="freedraw",
       key="canvas"
   )

   if canvas_result.image_data is not None:
       st.image(canvas_result.image_data, caption="Your Drawing", use_column_width=True)
       symbols = preprocess_image(canvas_result.image_data)
       symbol_labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                        10: '-', 11: '+', 12: '*'}
       predicted_symbols = []
       for symbol in symbols:
           prediction = loaded_model.predict(symbol)
           predicted_label = np.argmax(prediction)
           predicted_symbols.append(symbol_labels[predicted_label])
       equation = ''.join(predicted_symbols)
       st.write(f"Predicted Equation: {equation}")
       try:
           result = eval(equation)
           st.write(f"Result: {result}")
       except Exception as e:
           st.write(f"Error evaluating the equation: {e}")
   ```

## Conclusion

This project demonstrates the use of Convolutional Neural Networks for recognizing and evaluating handwritten mathematical equations. The Streamlit app provides an interactive interface for drawing equations and obtaining results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The dataset used in this project is inspired by common data science exercises and tutorials involving handwritten digit and symbol recognition.
- This project was developed using the TensorFlow and Streamlit libraries.