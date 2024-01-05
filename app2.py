# from fastapi import FastAPI, HTTPException, Header
# from fastapi import FastAPI

# app = FastAPI()

# # Mock model prediction function
# def mock_model_predict(input_data):
#     # Replace this with your actual model prediction logic
#     return {"input": input_data, "result": "1234"}

# # Dictionary to store asynchronous predictions
# async_predictions = {}

# @app.get("/")
# def read_root():
#     return {"message": "Hello, World!"}


# @app.post("/predict")
# async def predict(input_data: dict, async_mode: bool = Header(False)):
#     if async_mode:
#         # Asynchronous processing
#         prediction_id = generate_unique_id()
#         async_predictions[prediction_id] = None
#         return {"message": "Request received. Processing asynchronously.", "prediction_id": prediction_id}
#     else:
#         # Synchronous processing
#         result = mock_model_predict(input_data["input"])
#         return {"input": input_data["input"], "result": result["result"]}

# @app.get("/predict/{prediction_id}")
# async def get_prediction(prediction_id: str):
#     if prediction_id not in async_predictions:
#         raise HTTPException(status_code=404, detail="Prediction ID not found.")
#     elif async_predictions[prediction_id] is None:
#         raise HTTPException(status_code=400, detail="Prediction is still being processed.")
#     else:
#         result = async_predictions[prediction_id]
#         return {"prediction_id": prediction_id, "output": result}

# # Function to simulate model prediction asynchronously
# async def simulate_async_model_prediction(prediction_id, input_data):
#     result = mock_model_predict(input_data)
#     async_predictions[prediction_id] = result

# # Helper function to generate unique IDs (replace with your preferred method)
# def generate_unique_id():
#     import uuid
#     return str(uuid.uuid4())

# # Simulate asynchronous processing in the background
# @app.on_event("startup")
# async def on_startup():
#     import asyncio
#     for prediction_id, input_data in async_predictions.items():
#         asyncio.create_task(simulate_async_model_prediction(prediction_id, input_data))

# app.py
import time
import random
import numpy as np
from typing import Dict
from fastapi import FastAPI, HTTPException, status, Header, BackgroundTasks
from queue import Queue
from sklearn.linear_model import LinearRegression
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, status, Header, BackgroundTasks
from fastapi import FastAPI, HTTPException
import random



app = FastAPI()

# Linear Regression Model
linear_model = LinearRegression()

# In-memory queue for asynchronous predictions
prediction_queue = Queue()

# Enable CORS for all origins (you can customize as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Part A - Synchronous Model Prediction
@app.get("/predict", status_code=200)
@app.post("/predict", status_code=200)
def predict(input_data: dict):
    try:
        input_text = input_data.get("input", "")
        result = linear_model_predict(input_text)
        return {"input": input_text, "result": result}
    except Exception as e:
        # Log the exception details for debugging
        print(f"Exception: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.post("/predict", status_code=200)
@app.get("/predict", status_code=200)
def predict(input_data: dict):
    input_text = input_data.get("input", "")
    result = linear_model_predict(input_text)
    return {"input": input_text, "result": result}

# @app.post("/predict")
# def predict(input_data: dict):
#     input_text = input_data.get("input", "")
#     result = linear_model_predict(input_text)
#     return {"input": input_text, "result": result}

# Part B - Asynchronous Model Prediction
@app.post("/predict", status_code=202)
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

def generate_prediction_id():
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890', k=6))

def process_async_predictions(prediction_id: str, input_data: dict):
    # Replace the following line with your actual async processing logic
    # For simplicity, we're using the synchronous prediction here
    result = linear_model_predict(input_data["input"])

    # Store the result somewhere for later retrieval (e.g., in-memory dictionary)
    async_results[prediction_id] = {"input": input_data["input"], "result": result}
# Background task for asynchronous processing
async_results = {}  # In-memory storage for asynchronous results

def process_async_predictions(prediction_id: str):
    # Retrieve input_data from the queue and perform model prediction
    prediction_info = prediction_queue.get()
    input_data = prediction_info["input_data"]

    # Replace the following line with your actual async processing logic
    # For simplicity, we're using the synchronous prediction here
    result = linear_model_predict(input_data["input"])

    # Store the result somewhere for later retrieval (e.g., in-memory dictionary)
    async_results[prediction_id] = {"input": input_data["input"], "result": result}

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

if __name__ == "__main__":
    import nest_asyncio
    from fastapi import FastAPI
    import uvicorn

    nest_asyncio.apply()
    uvicorn.run(app, host="127.0.0.1", port=8000)


