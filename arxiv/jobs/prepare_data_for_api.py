from pathlib import Path
import pickle

from arxiv.extraction.extraction import download_arxiv_dataset, read_mappings, get_mapping, read_meta_data, \
    convert_edges_to_sigma_format, convert_nodes_to_sigma_format
from arxiv.extraction.positioning import get_node_positions

import multiprocessing as mp
from pathlib import Path


class DataSupplier:

    def __init__(self, download_dir=Path("..", "..", "data", "raw"), extract_dir=Path("..", "..", "data", "processed"),
                 overwrite: bool = False, base_position_algorithm: str = "kamada_kawai"):
        self.download_dir = download_dir
        self.extract_dir = extract_dir
        self.overwrite = overwrite
        self.base_position_algorithm = base_position_algorithm
        self.lock = mp.Lock()
        self.m_dict = mp.Manager().dict()

    def run(self):
        p = mp.Process(target=self.import_data, args=())
        p.start()
        len_items = 0
        while len_items < 5:
            with self.lock:
                len_items = len(self.m_dict)
        print("All items imported")

    def get_edges(self):
        return self._get_attribute("edges")

    def get_nodes(self):
        return self._get_attribute("nodes")

    def get_categories(self):
        return self._get_attribute("categories")

    def get_abstracts(self):
        return self._get_attribute("abstracts")

    def get_positions(self):
        return self._get_attribute("positions")

    def get_cal_pos_algorithms(self):
        return self._get_attribute("cal_pos_algorithms")

    def _get_attribute(self, attr: str):
        with self.lock:
            if attr in self.m_dict:
                return self.m_dict[attr]
            else:
                return None

    def import_data(self):
        dataset = download_arxiv_dataset(self.download_dir)

        meta_data = read_meta_data(Path(self.download_dir, "titleabs.tsv"))
        categories, node_ids = read_mappings(self.download_dir)

        paths = get_paths(self.extract_dir)
        if not paths["edges"].exists() or self.overwrite:
            edges = convert_edges_to_sigma_format(dataset)
            with open(paths["edges"], "wb") as f:
                pickle.dump(edges, f)
        else:
            with open(paths["edges"], "rb") as f:
                edges = pickle.load(f)
        with self.lock:
            self.m_dict["edges"] = edges

        if not paths["categories"].exists() or self.overwrite:
            categories_dict = categories.to_dict()["category"]
            with open(paths["categories"], "wb") as f:
                pickle.dump(categories_dict, f)
        else:
            with open(paths["categories"], "rb") as f:
                categories_dict = pickle.load(f)

        with self.lock:
            self.m_dict["categories"] = categories_dict

        if not paths["nodes"].exists() or self.overwrite:
            node_data = get_mapping(dataset, meta_data, node_ids, categories)
            nodes = convert_nodes_to_sigma_format(node_data, dataset)
            with open(paths["nodes"], "wb") as f:
                pickle.dump(nodes, f)
        else:
            with open(paths["nodes"], "rb") as f:
                nodes = pickle.load(f)

        with self.lock:
            self.m_dict["nodes"] = nodes

        if not paths["abstracts"].exists() or self.overwrite:
            abstracts = {row["ID"]: row["Abstract"] for row in meta_data.loc[:, ["ID", "Abstract"]].to_records()}
            with open(paths["abstracts"], "wb") as f:
                pickle.dump(abstracts, f)
        else:
            with open(paths["abstracts"], "rb") as f:
                abstracts = pickle.load(f)
        with self.lock:
            self.m_dict["abstracts"] = abstracts

        if not paths["positions"].exists() or self.overwrite:
            positions = get_node_positions(dataset, algorithm=self.base_position_algorithm)
            calculated_algorithms = [self.base_position_algorithm]
            with open(paths["positions"], "wb") as f:
                pickle.dump(positions, f)
            with open(paths["cal_pos_algorithms"], "wb") as f:
                pickle.dump(calculated_algorithms, f)
        else:
            with open(paths["positions"], "rb") as f:
                positions = pickle.load(f)
            with open(paths["cal_pos_algorithms"], "rb") as f:
                calculated_algorithms = pickle.load(f)
        with self.lock:
            self.m_dict["positions"] = positions
            self.m_dict["cal_pos_algorithms"] = calculated_algorithms


def get_paths(path: Path):
    edges_path = Path(path, "edges.pckl")
    cat_path = Path(path, "categories.pckl")
    nodes_path = Path(path, "nodes.pckl")
    abstract_path = Path(path, "abstracts.pckl")
    positions_path = Path(path, "positions.pckl")
    cal_pos_algo_path = Path(path, "cal_pos_algo.pckl")
    paths = {"edges": edges_path, "category": cat_path, "nodes": nodes_path, "abstracts": abstract_path,
             "positions": positions_path, "cal_pos_algorithms": cal_pos_algo_path}
    return paths


if __name__ == "__main__":
    pass
