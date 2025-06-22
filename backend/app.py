from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')

app = FastAPI()

class Item(BaseModel):
    data : list 
    

@app.post("/predict")
def predict(item: Item):
    try: 
        input_data = np.array(item.data).reshape(1,28,28)
        prediction = model.predict(input_data)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        return {"predicted_class": predicted_class, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}
 