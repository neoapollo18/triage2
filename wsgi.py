from main import app

# Standard WSGI entry point
application = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("wsgi:application", host="0.0.0.0", port=8000, reload=True)
