from main import app as application

# This file serves as an entry point for Railway deployment
# It imports the FastAPI app from main.py and renames it to "application"
# which is a common name expected by WSGI/ASGI servers

app = application

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
