from pathlib import Path

from fastapi import FastAPI
from arxiv.jobs.prepare_data_for_api import main

app = FastAPI()

edges, nodes, abstracts, categories, positions, calculated_algorithms = main(download_dir=Path("..", "..", "data",  "raw"), extract_dir=Path("..", "..", "data", "processed"))


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