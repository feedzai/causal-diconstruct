import functools
import time
import networkx as nx
import matplotlib.pyplot as plt
import hashlib
from sklearn.metrics import jaccard_score
import json
from typing import *
import yaml
from sklearn.model_selection import ParameterSampler
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
from torch import nn
import torch

import numpy as np


def timeit(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        run_time_str = (
            f"{round(run_time // 60)} mins and {round(run_time % 60)} secs"
            if run_time // 60 > 0
            else f"{round(run_time)} secs"
        )
        print(f"Finished {func.__name__!r} in {run_time_str}")
        return value

    return wrapper_timer


def plot_graph(
    graph,
    fig=None,
    ax=None,
    pos=None,
    node_size: int = 5000,
    return_pos=False,
    return_edges=False,
    return_fig=False,
    figsize=(16, 12),
    edge_widths=None,
    edge_colors=None,
    edge_labels=None,
    pos_seed=None,
    arrows: bool = True,
):
    pos = pos or nx.spring_layout(graph, k=3, seed=pos_seed or 42)

    fig = fig or plt.figure(figsize=figsize)
    ax = ax or fig.gca()

    nx.draw_networkx(
        graph,
        ax=ax,
        pos=pos,
        arrows=arrows,
        node_size=node_size,
        arrowsize=20,
        edge_color=edge_colors or "k",
        width=edge_widths or 2,
        with_labels=False,
    )

    label_pos = dict()
    labels = dict()
    for node, coords in pos.items():
        label_pos[node] = (coords[0], coords[1] + 0.1)
        labels[node] = node

    nx.draw_networkx_labels(graph, label_pos, labels=labels, ax=ax)

    drawn_edges = None
    if edge_labels:
        drawn_edges = nx.draw_networkx_edge_labels(
            graph, pos, edge_labels=edge_labels, font_size=12, font_weight="bold", ax=ax
        )
    ax.grid(False)
    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    y_max = max(y_values)
    y_min = min(y_values)
    x_margin = (x_max - x_min) * 0.28
    y_margin = (y_max - y_min) * 0.07
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(ymax=y_max + y_margin)

    fig.canvas.draw()
    fig.canvas.flush_events()
    if return_fig:
        return fig
    if return_pos:
        return pos
    if return_edges and drawn_edges:
        return drawn_edges


def generate_uuid(inputs) -> str:
    """Generates a unique identifier based on a set of inputs."""
    return hashlib.md5(json.dumps(inputs, sort_keys=True).encode("utf-8")).hexdigest()


def compute_jaccard_similarities(labels: np.ndarray):
    nr_labels = labels.shape[1]
    jaccard_similarities = np.eye(nr_labels)
    for variable_i in range(nr_labels):
        for variable_j in range(nr_labels):
            if variable_i != variable_j:
                if jaccard_similarities[variable_j, variable_i] == 0:
                    jaccard_similarities[variable_i, variable_j] = jaccard_score(
                        labels[:, variable_i], labels[:, variable_j]
                    )
                else:
                    jaccard_similarities[variable_i, variable_j] = jaccard_similarities[
                        variable_j, variable_i
                    ]

    return jaccard_similarities


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def logit(x: np.ndarray):
    return np.log(
        np.divide(
            x, (1 - x), out=np.full_like(x, np.inf, dtype=np.float64), where=x < 1
        ),
        out=np.full_like(x, -np.inf, dtype=np.float64),
        where=x > 0,
    )


def simplified_emb_func(nr_embeddings: int) -> int:
    return 1 if nr_embeddings < 10 else np.floor(np.log(nr_embeddings)).astype(int)


def create_parameter_grid(
    config_path: str, n_trials: int, seed: int
) -> List[Dict[str, Any]]:
    with open(config_path, "r") as f:
        parameters = yaml.safe_load(f)

    parameter_grid_config = dict()
    for param_name, param_config in parameters.items():
        if param_config["type"] == "categorical":
            parameter_grid_config[param_name] = param_config["config"]
        elif param_config["type"] == "uniform":
            parameter_grid_config[param_name] = uniform(
                *param_config["config"].values()
            )
        elif param_config["type"] == "loguniform":
            parameter_grid_config[param_name] = loguniform(
                *param_config["config"].values()
            )
        else:
            raise ValueError(
                f"Invalid configuration type ({param_config['type']}) for parameter "
                f"{param_name}."
            )

    return list(
        ParameterSampler(parameter_grid_config, n_iter=n_trials, random_state=seed)
    )


PARAMS_TO_ROUND = ["train_batch_size", "up_to_batch"]


def round_parameters(
    config: Dict[str, Any], params_to_round: List[str] = None
) -> Dict[str, Any]:
    params_to_round = params_to_round or PARAMS_TO_ROUND
    for parameter_name in params_to_round:
        if parameter_name in config:
            config[parameter_name] = round(config[parameter_name])

    return config


def init_weights(m, activation: str = "relu"):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(activation))
