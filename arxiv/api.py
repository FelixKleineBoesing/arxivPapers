from fastapi import FastAPI

from arxiv.extraction import download_arxiv_dataset, read_mappings, get_mapping, read_meta_data

app = FastAPI()


dataset = download_arxiv_dataset()
meta_data = read_meta_data()
categories, node_ids = read_mappings()
node_data = get_mapping(dataset, meta_data, node_ids, categories)


@app.get("/test")
def read_root():
    return {"msg": "Test"}


@app.get("/models/{model_id}")
def read_model(model_id: int):
    return {"model_id": model_id}


if __name__ == "__main__":
    pass