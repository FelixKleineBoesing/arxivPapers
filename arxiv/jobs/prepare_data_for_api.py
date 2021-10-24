from pathlib import Path
import pickle

from arxiv.extraction.extraction import download_arxiv_dataset, read_mappings, get_mapping, read_meta_data, \
    convert_edges_to_sigma_format, convert_nodes_to_sigma_format
from arxiv.extraction.positioning import get_node_positions


def main(download_dir = Path("..", "..", "data",  "raw"), extract_dir = Path("..", "..", "data", "processed"),
         overwrite: bool = False, base_position_algorithm: str = "kamada_kawai"):
    dataset = download_arxiv_dataset(download_dir)

    meta_data = read_meta_data(Path(download_dir, "titleabs.tsv"))
    categories, node_ids = read_mappings(download_dir)

    paths = get_paths(extract_dir)
    if not paths["edges"].exists() or overwrite:
        edges = convert_edges_to_sigma_format(dataset)
        with open(paths["edges"], "wb") as f:
            pickle.dump(edges, f)
    else:
        with open(paths["edges"], "rb") as f:
            edges = pickle.load(f)

    if not paths["categories"].exists() or overwrite:
        categories_dict = categories.to_dict()["category"]
        with open(paths["categories"], "wb") as f:
            pickle.dump(categories_dict, f)
    else:
        with open(paths["categories"], "rb") as f:
            categories_dict = pickle.load(f)

    if not paths["nodes"].exists() or overwrite:
        node_data = get_mapping(dataset, meta_data, node_ids, categories)
        nodes = convert_nodes_to_sigma_format(node_data, dataset)
        with open(paths["nodes"], "wb") as f:
            pickle.dump(nodes, f)
    else:
        with open(paths["nodes"], "rb") as f:
            nodes = pickle.load(f)

    if not paths["abstracts"].exists() or overwrite:
        abstracts = {row["ID"]: row["Abstract"] for row in meta_data.loc[:, ["ID", "Abstract"]].to_records()}
        with open(paths["abstracts"], "wb") as f:
            pickle.dump(abstracts, f)
    else:
        with open(paths["abstracts"], "rb") as f:
            abstracts = pickle.load(f)

    if not paths["positions"].exists() or overwrite:
        positions = get_node_positions(dataset, algorithm=base_position_algorithm)
        calculated_algorithms = [base_position_algorithm]
        with open(paths["positions"], "wb") as f:
            pickle.dump(positions, f)
        with open(paths["cal_pos_algorithms"], "wb") as f:
            pickle.dump(calculated_algorithms, f)
    else:
        with open(paths["positions"], "rb") as f:
            positions = pickle.load(f)
        with open(paths["cal_pos_algorithms"], "rb") as f:
            calculated_algorithms = pickle.load(f)
    return edges, nodes, abstracts, categories_dict, positions, calculated_algorithms


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
    main(download_dir=Path("..", "..", "data", "raw"), extract_dir=Path("..", "..", "data", "processed"),
         overwrite=False)
