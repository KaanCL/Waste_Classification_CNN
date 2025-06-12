
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

model = tf.lite.Interpreter(model_path='../WasteClassification_model.tflite')
model.allocate_tensors()

input_details = model.get_input_details()
output_details = model.get_output_details()

class_labels = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 'PCB', 
               'Player', 'Printer', 'Television', 'Washing Machine', 
               'cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic', 'trash']

class TFLiteModel:
    def __init__(self,file):
        self.file = file
    
    def predict(self):
        try:
            img = Image.open(self.file)
            img = img.resize((224, 224))
            img = img.convert('RGB')  
            img_array = np.array(img).astype(np.float32)
            img_array = img_array / 255.0  
            img_array = np.expand_dims(img_array, axis=0)
          
            
            model.set_tensor(input_details[0]['index'], img_array)
            model.invoke()
            
            output_data = model.get_tensor(output_details[0]['index'])
            class_index = np.argmax(output_data[0])
            confidence = output_data[0][class_index]
            
            return f"{class_labels[class_index]} ({class_index}) - Confidence: {confidence:.2f}"
        
        except Exception as e:
            return str(e)

    
        