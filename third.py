# # app.py
# import time
# import random
# import numpy as np
# from typing import Dict
# from fastapi import FastAPI, HTTPException, status, Header, BackgroundTasks
# from queue import Queue
# from sklearn.linear_model import LinearRegression
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# from pydantic import BaseModel
# import uuid
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

# # Define a Pydantic model for input validation
# class PredictionInput(BaseModel):
#     input: str

# # Define a Pydantic model for asynchronous result
# class AsynchronousResult(BaseModel):
#     prediction_id: str
#     output: Dict[str, str]

# async_results = {}  # In-memory storage for asynchronous results

# async def process_async_predictions(prediction_id: str, input_data: str):
#     # Replace the following line with your actual async processing logic
#     # For simplicity, we're using the mock_model_predict here
#     result = await mock_model_predict(input_data)

#     # Store the result somewhere for later retrieval (e.g., in-memory dictionary)
#     async_results[prediction_id] = {"input": input_data, "result": result}

# # Mock asynchronous processing function
# async def mock_model_predict(input_data: str):
#     # Simulate asynchronous processing by sleeping for a few seconds
#     await asyncio.sleep(5)
    
#     # Perform some asynchronous processing logic here
#     # For now, let's just return a placeholder result
#     return f"Result for input: {input_data}"

# # Part A - Synchronous Model Prediction
# @app.post("/predict")
# def predict(input_data: PredictionInput):
#     result = linear_model_predict(input_data.input)
#     return {"input": input_data.input, "result": result}

# # Part B - Asynchronous Model Prediction
# @app.post("/predict", status_code=status.HTTP_202_ACCEPTED)
# async def predict_async(
#     input_data: PredictionInput, background_tasks: BackgroundTasks, async_mode: bool = Header(False)
# ):
#     if async_mode:
#         prediction_id = str(uuid.uuid4())  # Use UUID to generate a unique prediction_id

#         # Save input_data and prediction_id for later retrieval
#         prediction_queue.put({"prediction_id": prediction_id, "input_data": input_data.dict()})

#         # Schedule background task for asynchronous processing
#         background_tasks.add_task(process_async_predictions, prediction_id, input_data.input)

#         return {
#             "message": "Request received. Processing asynchronously.",
#             "prediction_id": prediction_id,
#         }
#     else:
#         result = linear_model_predict(input_data.input)
#         return {"input": input_data.input, "result": result}

# # Additional Endpoint for Asynchronous Results
# @app.get("/predicted/{prediction_id}", response_model=AsynchronousResult)
# async def get_asynchronous_result(prediction_id: str):
#     if prediction_id not in async_results:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction ID not found.")
    
#     result = async_results.get(prediction_id)
#     if result is None:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Prediction is still being processed.")
    
#     return {"prediction_id": prediction_id, "output": result}

# # Linear Regression Model Prediction Function with Random Dataset
# def linear_model_predict(input_data: str) -> float:
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

# if __name__ == "__main__":
#     import nest_asyncio
#     nest_asyncio.apply()
#     uvicorn.run(app, host="127.0.0.1", port=8000)

#Combined code :

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
import uuid
import asyncio

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

# Define a Pydantic model for input validation
class PredictionInput(BaseModel):
    input: str

# Define a Pydantic model for asynchronous result
class AsynchronousResult(BaseModel):
    prediction_id: str
    output: Dict[str, str]

# Background task for asynchronous processing
async_results = {}  # In-memory storage for asynchronous results

async def process_async_predictions(prediction_id: str, input_data: str):
    # Replace the following line with your actual async processing logic
    # For simplicity, we're using the mock_model_predict here
    result = await mock_model_predict(input_data)

    # Store the result somewhere for later retrieval (e.g., in-memory dictionary)
    async_results[prediction_id] = {"input": input_data, "result": result}

# Mock asynchronous processing function
async def mock_model_predict(input_data: str):
    # Simulate asynchronous processing by sleeping for a few seconds
    await asyncio.sleep(5)
    
    # Perform some asynchronous processing logic here
    # For now, let's just return a placeholder result
    return f"Result for input: {input_data}"

# Part A - Synchronous Model Prediction
@app.post("/predict")
def predict(input_data: PredictionInput):
    result = linear_model_predict(input_data.input)
    return {"input": input_data.input, "result": result}

# Part B - Asynchronous Model Prediction
@app.post("/predictioned", status_code=status.HTTP_202_ACCEPTED)
async def predict_async(
    input_data: PredictionInput, background_tasks: BackgroundTasks, async_mode: bool = Header(False)
):
    if async_mode:
        prediction_id = str(uuid.uuid4())  # Use UUID to generate a unique prediction_id

        # Save input_data and prediction_id for later retrieval
        prediction_queue.put({"prediction_id": prediction_id, "input_data": input_data.dict()})

        # Schedule background task for asynchronous processing
        background_tasks.add_task(process_async_predictions, prediction_id, input_data.input)

        return {
            "message": "Request received. Processing asynchronously.",
            "prediction_id": prediction_id,
        }
    else:
        result = linear_model_predict(input_data.input)
        return {"input": input_data.input, "result": result}

# Additional Endpoint for Asynchronous Results
@app.get("/predicted/{prediction_id}", response_model=AsynchronousResult)
async def get_asynchronous_result(prediction_id: str):
    if prediction_id not in async_results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction ID not found.")
    
    result = async_results.get(prediction_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Prediction is still being processed.")
    
    return {"prediction_id": prediction_id, "output": result}

# Linear Regression Model Prediction Function with Random Dataset
def linear_model_predict(input_data: str) -> float:
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
    nest_asyncio.apply()
    uvicorn.run(app, host="127.0.0.1", port=8000)


