from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Configure CORS middleware if needed
# origins = [
#    "http://localhost",
#    "http://localhost:3000",
# ]
# app.add_middleware(
#    CORSMiddleware,
#    allow_origins=origins,
#    allow_credentials=True,
#    allow_methods=["*"],
#    allow_headers=["*"],
# )

# Load the model
MODEL = tf.saved_model.load("D:/Works/M.6/New Soil/DeployModel/FastAPI/SoilModel")

CLASS_NAMES = ["ดินเหนียว", "ดินร่วน", "ดินแดง", "ดินทราย"]

@app.get("/home")
async def ping():
    return "Hello, Welcome to CropChat"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((256, 256))  # Resize image to 256x256
    image = np.array(image)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    # Convert the image to a tensor
    img_tensor = tf.convert_to_tensor(img_batch)

    # Run the prediction using the model
    predictions = MODEL(img_tensor)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'prediction': f'ชนิดของดิน คือ {predicted_class}' 
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
