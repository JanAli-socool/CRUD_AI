from app.main import my_app

if __name__ == "__main__":
    import nest_asyncio
    import uvicorn

    nest_asyncio.apply()
    uvicorn.run(my_app, host="127.0.0.1", port=8080)
