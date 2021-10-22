from fastapi import FastAPI


app = FastAPI()




@app.get("/get-nodes")
def get_nodes():
    return nodes


@app.get("/get-categories")
def get_categories():
    return categories


@app.get("/models/{model_id}")
def read_model(model_id: int):
    return {"model_id": model_id}


if __name__ == "__main__":
    pass