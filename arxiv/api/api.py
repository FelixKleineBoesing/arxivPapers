from pathlib import Path

from fastapi import FastAPI

from arxiv.extraction import download_arxiv_dataset, read_mappings, get_mapping, read_meta_data

app = FastAPI()

download_dir=Path("..", "..", "data",  "raw")

dataset = download_arxiv_dataset(download_dir)
edges = dataset.graph[0]
meta_data = read_meta_data(Path(download_dir, "titleabs.tsv"))
categories, node_ids = read_mappings(download_dir)
node_data = get_mapping(dataset, meta_data, node_ids, categories)
categories = categories.to_dict()["category"]
print("Bla")


@app.get("/get-nodes")
def get_nodes():
    return {"msg": "Test"}


@app.get("/models/{model_id}")
def read_model(model_id: int):
    return {"model_id": model_id}


if __name__ == "__main__":
    pass