# import time
# import random
# import numpy as np
# from fastapi import FastAPI, HTTPException, status, BackgroundTasks
# from queue import Queue
# from sklearn.linear_model import LinearRegression
# from fastapi.middleware.cors import CORSMiddleware
# import asyncio

# app = FastAPI()

# # Linear Regression Model
# linear_model = LinearRegression()

# # In-memory queue for asynchronous predictions
# prediction_queue = Queue()

# # Enable CORS for all origins (you can customize as needed)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Dictionary to store asynchronous results
# async_results = {}

# # Part A - Synchronous Model Prediction
# @app.post("/predict", status_code=200)
# @app.get("/predict", status_code=200)
# def predict(input_data: dict):
#     try:
#         input_text = input_data.get("input", "")
#         result = linear_model_predict(input_text)
#         return {"input": input_text, "result": result}
#     except Exception as e:
#         # Log the exception details for debugging
#         print(f"Exception: {e}")
#         raise HTTPException(status_code=500, detail="Internal Server Error")

# # Part B - Asynchronous Model Prediction
# @app.post("/predict", status_code=202)
# async def predict_async(
#     input_data: dict, background_tasks: BackgroundTasks,
#     async_mode: bool = False
# ):
#     if async_mode:
#         prediction_id = generate_prediction_id()

#         # Schedule background task for asynchronous processing
#         background_tasks.add_task(process_async_predictions, prediction_id, input_data)

#         return {
#             "message": "Request received. Processing asynchronously.",
#             "prediction_id": prediction_id,
#         }
#     else:
#         raise HTTPException(status_code=400, detail="Async-Mode header not set to true")

# def generate_prediction_id():
#     return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890', k=12))

# async def process_async_predictions(prediction_id: str, input_data: dict):
#     # Replace the following line with your actual async processing logic
#     result = await async_linear_model_predict(input_data["input"])

#     # Store the result in async_results
#     async_results[prediction_id] = {"input": input_data["input"], "result": result}

# # Linear Regression Model Prediction Function with Random Dataset
# def linear_model_predict(input_data: str) -> float:
#     try:
#         input_value = float(input_data)
#     except ValueError:
#         raise HTTPException(status_code=422, detail="Input must be a valid numeric value")
#     # Generate a random dataset for demonstration
#     np.random.seed(42)
#     X_train = np.random.rand(100, 1) * 10  # Random input values
#     y_train = 2 * X_train.squeeze() + np.random.randn(100)  # Linear relationship with noise

#     # Train the linear regression model
#     linear_model.fit(X_train, y_train)

#     # Perform prediction
#     input_value = float(input_data)
#     prediction = linear_model.predict([[input_value]])

#     return prediction[0]

# # Async version of linear model prediction
# async def async_linear_model_predict(input_data: str) -> float:
#     return linear_model_predict(input_data)

# # Asynchronous startup event to initiate the background task
# @app.on_event("startup")
# async def on_startup():
#     asyncio.create_task(background_task())

# # Background task for asynchronous processing
# async def background_task():
#     while True:
#         if not prediction_queue.empty():
#             # Retrieve prediction info from the queue and process asynchronously
#             prediction_info = prediction_queue.get()
#             prediction_id = prediction_info["prediction_id"]
#             input_data = prediction_info["input_data"]

#             # Use asyncio.sleep for asynchronous sleep
#             await asyncio.sleep(0.1)

#             # Process the async function
#             await process_async_predictions(prediction_id, input_data)

#             print(f"Processing prediction {prediction_id} for input {input_data['input']} asynchronously.")

# # For cProfile compatibility
# def run_server():
#     import nest_asyncio
#     from fastapi import FastAPI
#     import uvicorn

#     nest_asyncio.apply()
#     uvicorn.run(app, host="127.0.0.1", port=8080)

# if __name__ == "__main__":
#     run_server()

# app.py
import time
import random
import numpy as np
from typing import Dict
from fastapi import FastAPI, HTTPException, status, Header, BackgroundTasks
from queue import Queue
from sklearn.linear_model import LinearRegression
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

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
@app.post("/predict")
def predict(input_data: dict):
    input_text = input_data.get("input", "")
    result = linear_model_predict(input_text)
    return {"input": input_text, "result": result}

# Part B - Asynchronous Model Prediction
@app.post("/predict", status_code=status.HTTP_202_ACCEPTED)
async def predict_async(
    input_data: dict, background_tasks: BackgroundTasks, async_mode: bool = Header(False)
):
    if async_mode:
        prediction_id = str(random.randint(1000, 9999))

        # Save input_data and prediction_id for later retrieval
        prediction_queue.put({"prediction_id": prediction_id, "input_data": input_data})

        # Schedule background task for asynchronous processing
        background_tasks.add_task(process_async_predictions, prediction_id)

        return {
            "message": "Request received. Processing asynchronously.",
            "prediction_id": prediction_id,
        }
    else:
        input_text = input_data.get("input", "")
        result = linear_model_predict(input_text)
        return {"input": input_text, "result": result}

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
        # Try to convert the input_data to a float
        input_value = float(input_data)
    except ValueError:
        # Handle the case where the conversion fails (e.g., invalid input)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid input. Please provide a valid numeric value.",
        )

    # Generate a random dataset for demonstration
    np.random.seed(42)
    X_train = np.random.rand(100, 1) * 10  # Random input values
    y_train = 2 * X_train.squeeze() + np.random.randn(100)  # Linear relationship with noise

    # Train the linear regression model
    linear_model.fit(X_train, y_train)

    # Perform prediction
    prediction = linear_model.predict([[input_value]])

    return prediction[0]

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    uvicorn.run(app, host="127.0.0.1", port=8000)









