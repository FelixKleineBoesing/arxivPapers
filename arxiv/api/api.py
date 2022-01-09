from pathlib import Path

from fastapi import FastAPI

from arxiv.api.helpers import get_return_message, Status
from arxiv.jobs.prepare_data_for_api import DataSupplier

app = FastAPI()

data_supplier = DataSupplier(download_dir=Path("..", "..", "data",  "raw"),
                             extract_dir=Path("..", "..", "data", "processed"),
                             asynchron=False)
data_supplier.run()


@app.get("/get-nodes")
def get_nodes():
    nodes = data_supplier.get_nodes()
    if nodes is None:
        return get_return_message(status=Status.calculating, msg="The nodes are still in initializing. Please wait a moment!")
    else:
        return get_return_message(status=Status.ok, msg="Nodes calculated", data=nodes)


@app.get("/get-categories")
def get_categories():
    categories = data_supplier.get_categories()
    if categories is None:
        return get_return_message(status=Status.calculating,
                                  msg="The nodes are still in initializing. Please wait a moment!")
    else:
        return get_return_message(status=Status.ok, msg="Nodes calculated", data=categories)


@app.get("/models/{model_id}")
def read_model(model_id: int):
    return {"model_id": model_id}


if __name__ == "__main__":
    pass