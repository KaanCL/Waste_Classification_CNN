import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


model = load_model('../WasteClassification_model.h5')
class_labels = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 'PCB', 'Player', 'Printer', 'Television', 'Washing Machine', 'cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic', 'trash']

class KerasModel:
    def __init__(self,file):
        self.file=file

    def predict(self):
        try:
        
            img = image.load_img(self.file, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0 
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)
            class_index = np.argmax(pred[0])
            confidence = pred[0][class_index]
            return f"{class_labels[class_index]} ({class_index}) - Confidence: {confidence:.2f}"

        except Exception as e:
            return str(e)
            
        
      