from typing import *
from copy import deepcopy
from dataclasses import dataclass, field
import torch
from torch import nn
import numpy as np
import pickle
from causal_concept_distil.utils import generate_uuid
from causal_concept_distil.multi_label_prior_model import (
    MultiLabelPriorModelNN,
    LoadedMultilabelPriorModel,
    ImageMultiLabelPriorModelNN,
)
from causal_concept_distil.independent_components import (
    IndependentPriorModelNN,
    ImageIndependentPriorModelNN,
)

UUID_PARAMS = [
    "lr",
    "l2_decay",
    "train_every",
    "gamma",
    "independence_discriminator_dropouts",
    "independence_discriminator_layers",
    "local_dropouts",
    "local_layers",
    "concept_dropouts",
    "concept_layers",
    "common_dropouts",
    "common_layers",
    "n_batches",
    "batch_size",
    "only_causal_sources",
    "concept_graph_edges",
    "uses_target_for_distillation",
    "has_causal_biases",
]


@dataclass(eq=False)
class TrainableConceptDistil:
    mean_training_losses_per_epoch: List[float] = field(default_factory=list)
    training_losses_per_epoch: List[List[float]] = field(default_factory=list)
    mean_roc_auc_per_epoch: List[float] = field(default_factory=list)
    roc_aucs_per_epoch: List[Dict[str, float]] = field(default_factory=list)
    concept_rocs_per_epoch: List[Dict[str, Dict[str, np.ndarray]]] = field(
        default_factory=list
    )
    abs_fidelity_per_epoch: List[float] = field(default_factory=list)
    rel_fidelity_per_epoch: List[float] = field(default_factory=list)
    norm_fidelity_per_epoch: List[float] = field(default_factory=list)
    recall_at_10fpr_per_epoch: List[float] = field(default_factory=list)
    recall_at_5fpr_per_epoch: List[float] = field(default_factory=list)
    recall_at_3fpr_per_epoch: List[float] = field(default_factory=list)
    recall_at_1fpr_per_epoch: List[float] = field(default_factory=list)
    diversity_max_0_per_epoch: List[float] = field(default_factory=list)
    diversity_max_1_per_epoch: List[float] = field(default_factory=list)
    diversity_dataset_0_per_epoch: List[float] = field(default_factory=list)
    diversity_dataset_1_per_epoch: List[float] = field(default_factory=list)
    diversity_pairwise_0_per_epoch: List[float] = field(default_factory=list)
    diversity_pairwise_1_per_epoch: List[float] = field(default_factory=list)
    avg_bce_loss_per_epoch: List[float] = field(default_factory=list)
    avg_base_bce_loss_per_epoch: List[float] = field(default_factory=list)
    best_model_state_dict: OrderedDict[str, torch.Tensor] = field(default_factory=dict)
    independence_roc_auc_per_epoch: List[float] = field(default_factory=list)
    independence_bce_loss_per_epoch: List[float] = field(default_factory=list)
    concept_acc_per_epoch: List[float] = field(default_factory=list)
    optimizer: torch.optim.Optimizer = None
    n_batches: int = None
    batch_size: int = None
    data_augmentation: bool = None
    soft_labels: bool = None
    cat_feat_embedding: str = None
    test_results: Dict[str, Any] = None

    @property
    def common_layers(self) -> List[int]:
        if hasattr(self, "concept_predictor") and (
            isinstance(self.concept_predictor, MultiLabelPriorModelNN)
            or isinstance(self.concept_predictor, IndependentPriorModelNN)
        ):
            return self.concept_predictor.common_ff.linear_layers.hidden_layers_dims
        elif hasattr(self, "concept_predictor") and isinstance(
            self.concept_predictor, LoadedMultilabelPriorModel
        ):
            return self.concept_predictor.model.common_hidden_layers

    @property
    def common_dropouts(self) -> List[float]:
        if hasattr(self, "concept_predictor") and (
            isinstance(self.concept_predictor, MultiLabelPriorModelNN)
            or isinstance(self.concept_predictor, IndependentPriorModelNN)
        ):
            return self.concept_predictor.common_ff.linear_layers.dropouts
        elif hasattr(self, "concept_predictor") and isinstance(
            self.concept_predictor, LoadedMultilabelPriorModel
        ):
            return self.concept_predictor.model.common_dropout

    @property
    def common_batch_norm(self) -> bool:
        if hasattr(self, "concept_predictor") and (
            isinstance(self.concept_predictor, MultiLabelPriorModelNN)
            or isinstance(self.concept_predictor, IndependentPriorModelNN)
        ):
            return self.concept_predictor.common_ff.linear_layers.batch_norm

    @property
    def concept_layers(self) -> List[int]:
        if hasattr(self, "concept_predictor") and (
            isinstance(self.concept_predictor, MultiLabelPriorModelNN)
            or isinstance(self.concept_predictor, IndependentPriorModelNN)
            or isinstance(self.concept_predictor, ImageIndependentPriorModelNN)
            or isinstance(self.concept_predictor, ImageMultiLabelPriorModelNN)
        ):
            return next(
                iter(self.concept_predictor.label_ffs.values())
            ).hidden_layers_dims
        elif hasattr(self, "concept_predictor") and isinstance(
            self.concept_predictor, LoadedMultilabelPriorModel
        ):
            return self.concept_predictor.model.concept_hidden_layers

    @property
    def concept_dropouts(self) -> List[float]:
        if hasattr(self, "concept_predictor") and (
            isinstance(self.concept_predictor, MultiLabelPriorModelNN)
            or isinstance(self.concept_predictor, IndependentPriorModelNN)
            or isinstance(self.concept_predictor, ImageIndependentPriorModelNN)
            or isinstance(self.concept_predictor, ImageMultiLabelPriorModelNN)
        ):
            return next(iter(self.concept_predictor.label_ffs.values())).dropouts
        elif hasattr(self, "concept_predictor") and isinstance(
            self.concept_predictor, LoadedMultilabelPriorModel
        ):
            return self.concept_predictor.model.concept_dropout

    @property
    def concept_batch_norm(self) -> bool:
        if hasattr(self, "concept_predictor") and (
            isinstance(self.concept_predictor, MultiLabelPriorModelNN)
            or isinstance(self.concept_predictor, IndependentPriorModelNN)
            or isinstance(self.concept_predictor, ImageIndependentPriorModelNN)
            or isinstance(self.concept_predictor, ImageMultiLabelPriorModelNN)
        ):
            return next(
                iter(self.concept_predictor.label_ffs.values())
            ).linear_layers.batch_norm

    @property
    def local_layers(self) -> List[int]:
        if hasattr(self, "causal_factors") and isinstance(
            self.causal_factors, nn.ModuleDict
        ):
            return next(iter(self.causal_factors.values())).hidden_layers_dims
        elif hasattr(self, "post_bottleneck_model"):
            return self.post_bottleneck_model.hidden_layers_dims

    @property
    def local_dropouts(self) -> List[float]:
        if hasattr(self, "causal_factors") and isinstance(
            self.causal_factors, nn.ModuleDict
        ):
            return next(iter(self.causal_factors.values())).dropouts
        elif hasattr(self, "post_bottleneck_model"):
            return self.post_bottleneck_model.dropouts

    @property
    def local_batch_norm(self) -> List[float]:
        if hasattr(self, "causal_factors") and isinstance(
            self.causal_factors, nn.ModuleDict
        ):
            return next(iter(self.causal_factors.values())).batch_norm
        elif hasattr(self, "post_bottleneck_model"):
            return self.post_bottleneck_model.batch_norm

    @property
    def independence_discriminator_layers(self) -> List[int]:
        if hasattr(self, "concept_predictor") and (
            isinstance(self.concept_predictor, IndependentPriorModelNN)
            or isinstance(self.concept_predictor, ImageIndependentPriorModelNN)
        ):
            return (
                self.concept_predictor.independent_components.discriminator.hidden_layers_dims
            )

    @property
    def independence_discriminator_dropouts(self) -> List[float]:
        if hasattr(self, "concept_predictor") and (
            isinstance(self.concept_predictor, IndependentPriorModelNN)
            or isinstance(self.concept_predictor, ImageIndependentPriorModelNN)
        ):
            return self.concept_predictor.independent_components.discriminator.dropouts

    @property
    def gamma(self) -> float:
        if hasattr(self, "concept_predictor") and (
            isinstance(self.concept_predictor, IndependentPriorModelNN)
            or isinstance(self.concept_predictor, ImageIndependentPriorModelNN)
        ):
            return self.concept_predictor.independent_components.gamma

    @property
    def train_every(self) -> int:
        if hasattr(self, "concept_predictor") and (
            isinstance(self.concept_predictor, IndependentPriorModelNN)
            or isinstance(self.concept_predictor, ImageIndependentPriorModelNN)
        ):
            return self.concept_predictor.train_independence_every

    @property
    def only_causal_sources(self) -> bool:
        if hasattr(self, "use_only_causal_sources"):
            return self.use_only_causal_sources

    @property
    def uses_target_for_distillation(self) -> bool:
        if hasattr(self, "requires_target_for_distillation"):
            return self.requires_target_for_distillation

    @property
    def has_causal_biases(self) -> bool:
        if hasattr(self, "with_biases"):
            return self.with_biases

    @property
    def lr(self) -> float:
        if self.optimizer:
            return self.optimizer.defaults["lr"]

    @property
    def l2_decay(self) -> float:
        if self.optimizer:
            return self.optimizer.defaults["weight_decay"]

    @property
    def uuid_params(self):
        return UUID_PARAMS

    @property
    def model_parameter_dict(self) -> Dict[str, Any]:
        return {
            k: getattr(self, k)
            for k in [
                "model_id",
                "optimizer",
                "n_epochs",
                "cat_feat_embedding",
                "data_augmentation",
                "soft_labels",
            ]
            + self.uuid_params
        }

    @property
    def model_validation_results_dict(self) -> Dict[str, Any]:
        return {
            "mean_golden_roc_auc": self.mean_roc_auc_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "mean_roc_auc_per_epoch") and self.mean_roc_auc_per_epoch
            else None,
            "abs_fidelity": self.abs_fidelity_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "abs_fidelity_per_epoch") and self.abs_fidelity_per_epoch
            else None,
            "rel_fidelity": self.rel_fidelity_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "rel_fidelity_per_epoch") and self.rel_fidelity_per_epoch
            else None,
            "norm_fidelity": self.norm_fidelity_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "norm_fidelity_per_epoch") and self.norm_fidelity_per_epoch
            else None,
            "recall_at_10fpr": self.recall_at_10fpr_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "recall_at_10fpr_per_epoch")
            and self.recall_at_10fpr_per_epoch
            else None,
            "recall_at_5fpr": self.recall_at_5fpr_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "recall_at_5fpr_per_epoch")
            and self.recall_at_5fpr_per_epoch
            else None,
            "recall_at_3fpr": self.recall_at_3fpr_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "recall_at_3fpr_per_epoch")
            and self.recall_at_3fpr_per_epoch
            else None,
            "recall_at_1fpr": self.recall_at_1fpr_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "recall_at_1fpr_per_epoch")
            and self.recall_at_1fpr_per_epoch
            else None,
            "diversity_max_0": self.diversity_max_0_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "diversity_max_0_per_epoch")
            and self.diversity_max_0_per_epoch
            else None,
            "diversity_max_1": self.diversity_max_1_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "diversity_max_1_per_epoch")
            and self.diversity_max_1_per_epoch
            else None,
            "diversity_dataset_0": self.diversity_dataset_0_per_epoch[
                self.best_epoch
                if hasattr(self, "best_epoch")
                and self.best_epoch < len(self.diversity_dataset_0_per_epoch)
                else -1
            ]
            if hasattr(self, "diversity_dataset_0_per_epoch")
            and self.diversity_dataset_0_per_epoch
            else None,
            "diversity_dataset_1": self.diversity_dataset_1_per_epoch[
                self.best_epoch
                if hasattr(self, "best_epoch")
                and self.best_epoch < len(self.diversity_dataset_1_per_epoch)
                else -1
            ]
            if hasattr(self, "diversity_dataset_1_per_epoch")
            and self.diversity_dataset_1_per_epoch
            else None,
            "diversity_pairwise_0": self.diversity_pairwise_0_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "diversity_pairwise_0_per_epoch")
            and self.diversity_pairwise_0_per_epoch
            else None,
            "diversity_pairwise_1": self.diversity_pairwise_1_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "diversity_pairwise_1_per_epoch")
            and self.diversity_pairwise_1_per_epoch
            else None,
            "avg_bce_loss": self.avg_bce_loss_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "avg_bce_loss_per_epoch") and self.avg_bce_loss_per_epoch
            else None,
            "avg_base_bce_loss": self.avg_base_bce_loss_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "avg_base_bce_loss_per_epoch")
            and self.avg_base_bce_loss_per_epoch
            else None,
            "independence_roc_auc": self.independence_roc_auc_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "independence_roc_auc_per_epoch")
            and self.independence_roc_auc_per_epoch
            else None,
            "independence_bce_loss": self.independence_bce_loss_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "independence_bce_loss_per_epoch")
            and self.independence_bce_loss_per_epoch
            else None,
            "concept_acc": self.concept_acc_per_epoch[
                self.best_epoch if hasattr(self, "best_epoch") else -1
            ]
            if hasattr(self, "concept_acc_per_epoch") and self.concept_acc_per_epoch
            else None,
        }

    @property
    def model_test_results_dict(self) -> Dict[str, Any]:
        keys = self.model_validation_results_dict.keys()
        if self.test_results:
            test_results = {k: v for k, v in self.test_results.items() if k in keys}
            return test_results

    @property
    def concept_graph_edges(self) -> List[Tuple[str, str]]:
        if hasattr(self, "concept_graph"):
            return list(self.concept_graph.edges)
        else:
            return list()

    @property
    def model_id(self) -> str:
        uuid_params = {
            k: getattr(self, k)
            for k in self.uuid_params
            if getattr(self, k) is not None
        }
        return generate_uuid(uuid_params)

    @property
    def n_epochs(self) -> int:
        return len(self.mean_roc_auc_per_epoch)

    def set_training_parameters(self, parameters: Dict[str, Any]):
        self.optimizer = parameters.pop("optimizer", None)
        self.n_batches = parameters.get("n_batches", None)
        self.batch_size = parameters.get("batch_size", None)
        self.cat_feat_embedding = parameters.get("cat_feat_embedding", None)
        self.data_augmentation = parameters.get("data_augmentation", None)
        self.soft_labels = parameters.get("soft_labels", True)

    def add_training_losses(self, training_losses: List[float]):
        self.training_losses_per_epoch.append(training_losses)
        self.mean_training_losses_per_epoch.append(np.array(training_losses).mean())

    def add_epoch_validation_results(self, epoch_val_results: Dict[str, Any]):
        self.mean_roc_auc_per_epoch.append(epoch_val_results["mean_golden_roc_auc"])
        self.abs_fidelity_per_epoch.append(epoch_val_results["abs_fidelity"])
        self.rel_fidelity_per_epoch.append(epoch_val_results["rel_fidelity"])
        self.norm_fidelity_per_epoch.append(epoch_val_results["norm_fidelity"])
        self.recall_at_10fpr_per_epoch.append(epoch_val_results["recall_at_10fpr"])
        self.recall_at_5fpr_per_epoch.append(epoch_val_results["recall_at_5fpr"])
        self.recall_at_3fpr_per_epoch.append(epoch_val_results["recall_at_3fpr"])
        self.recall_at_1fpr_per_epoch.append(epoch_val_results["recall_at_1fpr"])
        self.diversity_max_0_per_epoch.append(epoch_val_results["diversity_max_0"])
        self.diversity_max_1_per_epoch.append(epoch_val_results["diversity_max_1"])
        self.diversity_dataset_0_per_epoch.append(
            epoch_val_results["diversity_dataset_0"]
        )
        self.diversity_dataset_1_per_epoch.append(
            epoch_val_results["diversity_dataset_1"]
        )
        self.diversity_pairwise_0_per_epoch.append(
            epoch_val_results["diversity_pairwise_0"]
        )
        self.diversity_pairwise_1_per_epoch.append(
            epoch_val_results["diversity_pairwise_1"]
        )
        self.avg_bce_loss_per_epoch.append(epoch_val_results["avg_bce_loss"])
        self.avg_base_bce_loss_per_epoch.append(epoch_val_results["avg_base_bce_loss"])
        self.roc_aucs_per_epoch.append(epoch_val_results["concept_aucs"])
        self.concept_rocs_per_epoch.append(epoch_val_results["concept_rocs"])
        self.independence_roc_auc_per_epoch.append(
            epoch_val_results["independence_roc_auc"]
        )
        self.independence_bce_loss_per_epoch.append(
            epoch_val_results["independence_bce_loss"]
        )
        self.concept_acc_per_epoch.append(epoch_val_results["concept_acc"])
        if (
            isinstance(self, nn.Module)
            and hasattr(self, "best_performance")
            and hasattr(self, "performance_metric_per_epoch")
        ):
            if self.best_performance == self.performance_metric_per_epoch[-1]:
                # If this model is the best up until now, add it's state dict to the
                # list.
                self.best_model_state_dict = deepcopy(self.state_dict())

    def add_test_results(self, test_results: Dict[str, Any]):
        self.test_results = test_results

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "TrainableConceptDistil":
        with open(filepath, "rb") as f:
            return pickle.load(f)
