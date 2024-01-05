import nest_asyncio
from fastapi import FastAPI
import uvicorn

nest_asyncio.apply()

my_app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(my_app, host="127.0.0.1", port=8080)
