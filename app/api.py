import time
import random
import numpy as np
from typing import Dict
from fastapi import FastAPI, HTTPException, status, Header, BackgroundTasks
from queue import Queue
from sklearn.linear_model import LinearRegression
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import threading
from main import my_app  # Import the FastAPI instance

# Linear Regression Model
linear_model = LinearRegression()

# In-memory queue for asynchronous predictions
prediction_queue = Queue()

# Enable CORS for all origins (you can customize as needed)
my_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary to store asynchronous results
async_results = {}

@my_app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Part A - Synchronous Model Prediction
@my_app.post("/predict", status_code=200)
@my_app.get("/predict", status_code=200)
def predict(input_data: dict):
    try:
        input_text = input_data.get("input", "")
        result = linear_model_predict(input_text)
        return {"input": input_text, "result": result}
    except Exception as e:
        # Log the exception details for debugging
        print(f"Exception: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Part B - Asynchronous Model Prediction
@my_app.post("/predict", status_code=202)
async def predict_async(
    input_data: dict, background_tasks: BackgroundTasks,
    async_mode: bool = Header(False)
):
    if async_mode:
        prediction_id = generate_prediction_id()

        # Schedule background task for asynchronous processing
        background_tasks.add_task(process_async_predictions, prediction_id, input_data)

        return {
            "message": "Request received. Processing asynchronously.",
            "prediction_id": prediction_id,
        }
    else:
        raise HTTPException(status_code=400, detail="Async-Mode header not set to true")
    
@my_app.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: str):
    # Your logic to retrieve predictions
    if prediction_id not in async_results:
        raise HTTPException(status_code=404, detail="Prediction ID not found.")
    elif async_results[prediction_id] is None:
        raise HTTPException(status_code=400, detail="Prediction is still being processed.")
    else:
        result = async_results[prediction_id]
        return {"prediction_id": prediction_id, "output": result}

def generate_prediction_id():
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890', k=12))

def process_async_predictions(prediction_id: str, input_data: dict):
    # Replace the following line with your actual async processing logic
    result = linear_model_predict(input_data["input"])

    # Store the result in async_results
    async_results[prediction_id] = {"input": input_data["input"], "result": result}

# Background task for asynchronous processing
async def background_task():
    while True:
        if not prediction_queue.empty():
            # Retrieve prediction info from the queue and process asynchronously
            prediction_info = prediction_queue.get()
            prediction_id = prediction_info["prediction_id"]
            input_data = prediction_info["input_data"]

            # Use asyncio.sleep for asynchronous sleep
            await asyncio.sleep(0.1)

            # Use threading to run the async function
            thread = threading.Thread(target=process_async_predictions, args=(prediction_id, input_data))
            thread.start()

            print(f"Processing prediction {prediction_id} for input {input_data['input']} asynchronously.")

# Linear Regression Model Prediction Function with Random Dataset
def linear_model_predict(input_data: str) -> float:
    try:
        input_value = float(input_data)
    except ValueError:
        raise HTTPException(status_code=422, detail="Input must be a valid numeric value")
    # Generate a random dataset for demonstration
    np.random.seed(42)
    X_train = np.random.rand(100, 1) * 10  # Random input values
    y_train = 2 * X_train.squeeze() + np.random.randn(100)  # Linear relationship with noise

    # Train the linear regression model
    linear_model.fit(X_train, y_train)

    # Perform prediction
    input_value = float(input_data)
    prediction = linear_model.predict([[input_value]])

    return prediction[0]

# Async version of linear model prediction
async def async_linear_model_predict(input_data: str) -> float:
    return linear_model_predict(input_data)

if __name__ == "__main__":
    asyncio.create_task(background_task())
