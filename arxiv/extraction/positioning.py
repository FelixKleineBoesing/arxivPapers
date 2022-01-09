from dgl import to_networkx
from networkx.drawing import kamada_kawai_layout, circular_layout, planar_layout, random_layout, shell_layout, spring_layout


AVAILABLE_LAYOUTS = ["kamada_kawai", "circular", "planar", "random", "shell", "spring"]


def get_node_positions(dataset, algorithm: str = "kamada_kawai"):
    network = to_networkx(dataset.graph[0])
    if algorithm == "kamda_kawai":
        positions = kamada_kawai_layout(network)
    elif algorithm == "circular":
        positions = circular_layout(network)
    elif algorithm == "planar":
        positions = planar_layout(network)
    elif algorithm == "random":
        positions = random_layout(network)
    elif algorithm == "shell":
        positions = shell_layout(network)
    elif algorithm == "spring":
        positions = spring_layout(network)
    elif algorithm == "embedding":
        positions = get_positions_by_embeddings(dataset)
    else:
        raise AssertionError(f"The algorithm {algorithm} is not available!")
    return positions


def get_positions_by_embeddings(dataset):
    dataset