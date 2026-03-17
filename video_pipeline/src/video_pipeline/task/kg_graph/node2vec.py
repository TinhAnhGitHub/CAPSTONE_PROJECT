import random
import re
from functools import partial
from typing import Callable, List
import warnings
import numpy as np
import networkx as nx
import numpy as np
import networkx as nx
from tqdm.auto import trange
from gensim.models.word2vec import Word2Vec


class RandomWalker:
    """
    Class to do fast first-order random walks.

    Args:
        walk_length (int): Number of random walks.
        walk_number (int): Number of nodes in truncated walk.
    """

    def __init__(self, walk_length: int, walk_number: int):
        self.walk_length = walk_length
        self.walk_number = walk_number

    def do_walk(self, node: int) -> list[str]:
        """
        Doing a single truncated random walk from a source node.

        Arg types:
            * **node** *(int)* - The source node of the random walk.

        Return types:
            * **walk** *(list of strings)* - A single truncated random walk.
        """
        walk: list[int] = [node]
        for _ in range(self.walk_length - 1):
            nebs = [node for node in self.graph.neighbors(walk[-1])]
            if len(nebs) > 0:
                walk = walk + random.sample(nebs, 1)
        walk = [str(w) for w in walk] #type: ignore
        return walk #type: ignore

    def do_walks(self, graph):
        """
        Doing a fixed number of truncated random walk from every node in the graph.

        Arg types:`
            * **graph** *(NetworkX graph)* - The graph to run the random walks on.
        """
        self.walks: list[list[str]] = []
        self.graph = graph
        for node in self.graph.nodes():
            for _ in range(self.walk_number):
                walk_from_node = self.do_walk(node)
                self.walks.append(walk_from_node)


def _check_value(value, name):
    try:
        _ = 1 / value

    except ZeroDivisionError:
        raise ValueError(
            f"The value of {name} is too small " f"or zero to be used in 1/{name}."
        )


def _undirected(node, graph) -> list[tuple]:
    edges = graph.edges(node)

    return edges


def _directed(node, graph) -> list[tuple]:
    edges = graph.out_edges(node, data=True)

    return edges


def _get_edge_fn(graph) -> Callable:
    fn = _directed if nx.classes.function.is_directed(graph) else _undirected

    fn = partial(fn, graph=graph)
    return fn


def _unweighted(edges: list[tuple]) -> np.ndarray:
    return np.ones(len(edges))


def _weighted(edges: list[tuple]) -> np.ndarray:
    weights = map(lambda edge: edge[-1]["weight"], edges)

    return np.array([*weights])


def _get_weight_fn(graph) -> Callable:
    fn = _weighted if nx.classes.function.is_weighted(graph) else _unweighted

    return fn


class BiasedRandomWalker:
    """
    Class to do biased second order random walks.

    Args:
        walk_length (int): Number of random walks.
        walk_number (int): Number of nodes in truncated walk.
        p (float): Return parameter (1/p transition probability) to move towards previous node.
        q (float): In-out parameter (1/q transition probability) to move away from previous node.
    """

    walks: list
    graph: nx.classes.graph.Graph
    edge_fn: Callable
    weight_fn: Callable

    def __init__(self, walk_length: int, walk_number: int, p: float, q: float):
        self.walk_length = walk_length
        self.walk_number = walk_number

        _check_value(p, "p")
        self.p = p

        _check_value(q, "q")
        self.q = q

    def do_walk(self, node: int) -> list[str]:
        """
        Doing a single truncated second order random walk from a source node.

        Arg types:
            * **node** *(int)* - The source node of the random walk.

        Return types:
            * **walk** *(list of strings)* - A single truncated random walk.
        """
        walk = [node]
        previous_node = None
        previous_node_neighbors = []
        for _ in range(self.walk_length - 1):
            current_node = walk[-1]
            edges = self.edge_fn(current_node)
            current_node_neighbors = np.array([edge[1] for edge in edges])

            weights = self.weight_fn(edges)
            probability = np.piecewise(
                weights,
                [
                    current_node_neighbors == previous_node,
                    np.isin(current_node_neighbors, previous_node_neighbors),
                ],
                [lambda w: w / self.p, lambda w: w / 1, lambda w: w / self.q],
            )

            norm_probability = probability / sum(probability)
            selected = np.random.choice(current_node_neighbors, 1, p=norm_probability)[
                0
            ]
            walk.append(selected)

            previous_node_neighbors = current_node_neighbors
            previous_node = current_node

        walk = [str(w) for w in walk]
        return walk

    def do_walks(self, graph) -> None:
        """
        Doing a fixed number of truncated random walk from every node in the graph.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to run the random walks on.
        """
        self.walks = []
        self.graph = graph

        self.edge_fn = _get_edge_fn(graph)
        self.weight_fn = _get_weight_fn(graph)

        for node in self.graph.nodes():
            for _ in range(self.walk_number):
                walk_from_node = self.do_walk(node)
                self.walks.append(walk_from_node)
                
                
class Estimator(object):
    """Estimator base class with constructor and public methods."""

    seed: int

    def __init__(self):
        """Creating an estimator."""

    def fit(self, graph: nx.classes.graph.Graph):
        """Fitting a model."""

    def get_embedding(self) -> np.ndarray:  # type: ignore
        """Getting the embeddings (graph or node level)."""

    def get_memberships(self):
        """Getting the membership dictionary."""

    def get_cluster_centers(self):
        """Getting the cluster centers."""

    def get_params(self):
        """Get parameter dictionary for this estimator.."""
        rx = re.compile(r"^\_")
        params = self.__dict__
        params = {key: params[key] for key in params if not rx.search(key)}
        return params

    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _set_seed(self):
        """Creating the initial random seed."""
        random.seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def _ensure_walk_traversal_conditions(
        graph: nx.classes.graph.Graph,
    ) -> nx.classes.graph.Graph:
        """Ensure walk traversal conditions."""
        for node_index in trange(
            graph.number_of_nodes(),
            # We do not leave the bar.
            leave=False,
            # We only show this bar when we can expect
            # for this process to take a bit of time.
            disable=graph.number_of_nodes() < 10_000,
            desc="Checking main diagonal existance",
            dynamic_ncols=True,
        ):
            if not graph.has_edge(node_index, node_index):
                warnings.warn(
                    "Please do be advised that "
                    "the graph you have provided does not "
                    "contain (some) edges in the main "
                    "diagonal, for instance the self-loop "
                    f"constitued of ({node_index}, {node_index}). These selfloops "
                    "are necessary to ensure that the graph "
                    "is traversable, and for this reason we "
                    "create a copy of the graph and add therein "
                    "the missing edges. Since we are creating "
                    "a copy, this will immediately duplicate "
                    "the memory requirements. To avoid this double "
                    "allocation, you can provide the graph with the selfloops."
                )
                # We create a copy of the graph
                graph = graph.copy()
                # And we add the missing edges
                # for filling the main diagonal
                graph.add_edges_from(
                    (
                        (index, index)
                        for index in range(graph.number_of_nodes())
                        if not graph.has_edge(index, index)
                    )
                )
                break

        return graph

    @staticmethod
    def _check_indexing(graph: nx.classes.graph.Graph):
        """Checking the consecutive numeric indexing."""
        numeric_indices = [index for index in range(graph.number_of_nodes())]
        node_indices = sorted([node for node in graph.nodes()])

        assert numeric_indices == node_indices, "The node indexing is wrong."

    def _check_graph(self, graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """Check the Karate Club assumptions about the graph."""
        self._check_indexing(graph)
        graph = self._ensure_walk_traversal_conditions(graph)

        return graph

    def _check_graphs(self, graphs: list[nx.classes.graph.Graph]):
        """Check the Karate Club assumptions for a list of graphs."""
        graphs = [self._check_graph(graph) for graph in graphs]

        return graphs
    


class Node2Vec(Estimator):
    r"""An implementation of `"Node2Vec" <https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf>`_
    from the KDD '16 paper "node2vec: Scalable Feature Learning for Networks".
    The procedure uses biased second order random walks to approximate the pointwise mutual information
    matrix obtained by pooling normalized adjacency matrix powers. This matrix
    is decomposed by an approximate factorization technique.

    Args:
        walk_number (int): Number of random walks. Default is 10.
        walk_length (int): Length of random walks. Default is 80.
        p (float): Return parameter (1/p transition probability) to move towards from previous node.
        q (float): In-out parameter (1/q transition probability) to move away from previous node.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        window_size (int): Matrix power order. Default is 5.
        epochs (int): Number of epochs. Default is 1.
        use_hierarchical_softmax (bool): Whether to use hierarchical softmax or negative sampling to train the model. Default is True.
        number_of_negative_samples (int): Number of negative nodes to sample (usually between 5-20). If set to 0, no negative sampling is used. Default is 5.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        min_count (int): Minimal count of node occurrences. Default is 1.
        seed (int): Random seed value. Default is 42.
    """
    _embedding: List[np.ndarray]

    def __init__(
        self,
        walk_number: int = 10,
        walk_length: int = 80,
        p: float = 1.0,
        q: float = 1.0,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        use_hierarchical_softmax: bool = False,
        number_of_negative_samples: int = 5,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):
        super(Node2Vec, self).__init__()

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.use_hierarchical_softmax = use_hierarchical_softmax
        self.number_of_negative_samples = number_of_negative_samples
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a DeepWalk model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        graph = self._check_graph(graph)
        walker = BiasedRandomWalker(self.walk_length, self.walk_number, self.p, self.q)
        walker.do_walks(graph)

        model = Word2Vec(
            walker.walks,
            hs=1 if self.use_hierarchical_softmax else 0,
            negative=self.number_of_negative_samples,
            alpha=self.learning_rate,
            epochs=self.epochs,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=self.min_count,
            workers=self.workers,
            seed=self.seed,
        )

        n_nodes = graph.number_of_nodes()
        self._embedding = [model.wv[str(n)] for n in range(n_nodes)]

    def get_embedding(self) -> np.ndarray:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.array(self._embedding)