from fastapi import FastAPI

# API initialization using FastAPI.
app = FastAPI()

@app.get('/')
def home():
    return {'Data': 'Test'}