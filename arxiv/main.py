from arxiv.extraction import download_arxiv_dataset
from pathlib  import Path


def main():
    download_arxiv_dataset(Path("..",  "data", "raw"))
