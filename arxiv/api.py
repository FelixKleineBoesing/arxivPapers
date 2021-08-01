from fastapi import FastAPI

app = FastAPI()


@app.get("/test")
def read_root():
    return {"msg": "Test"}


@app.get("/models/{model_id}")
def read_model(model_id: int):
    return {"model_id": model_id}