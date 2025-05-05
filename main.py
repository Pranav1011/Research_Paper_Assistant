# Entry point

from fastapi import FastAPI
app = FastAPI()

@app.get('/')
def read_root():
    return {"msg": "Research Paper Assistant API is running!"}