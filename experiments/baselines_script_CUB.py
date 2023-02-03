#!/usr/bin/python
import sys
import yaml
import os
import json
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd

from causal_concept_distil.datasets import CUBDataset
from causal_concept_distil.utils import (
    simplified_emb_func,
    create_parameter_grid,
    round_parameters,
)
from causal_concept_distil.multi_label_prior_model import ImageMultiLabelPriorModelNN
from causal_concept_distil.independent_components import ImageIndependentPriorModelNN
from causal_concept_distil.baselines import (
    ExplainabilityBaseline,
    ImageDistillationBaseline,
    IndependentComponentsBaseline,
)
from causal_concept_distil.loops import *

##################################
# General configurations
##################################
CONFIG_FILE = sys.argv[1]
with open(CONFIG_FILE, "r") as f:
    CONFIG = yaml.safe_load(f)

print(json.dumps(CONFIG, indent=4))
BASELINE = CONFIG["baseline"]
DEVICE_ID = CONFIG["device_id"]
EXPERIMENT_NAME = CONFIG["experiment_name"]
NUM_WORKERS = CONFIG["num_workers"]
SEED = CONFIG["seed"]  # Do not change this!
N_TRIALS = CONFIG["nr_trials"]
OPTIMIZER_CLS = torch.optim.Adam
EVAL_BATCH_SIZE = CONFIG["eval_batch_size"]
SAVE_BASE_PATH = CONFIG["save_base_path"]  # "causal_concept_distil_experiments"
CONFIG_PATH = CONFIG[
    "parameter_config_path"
]  # "causal_concept_distil_parameters_config.yaml"
CONCEPT_NAMES = CONFIG["used_concepts"]
NO_IMPROVEMENT_EPOCHS = CONFIG.get("no_improvement_epochs", 5)

##################################
# Global variables' setup.
##################################
METADATA_PATH = "cub_dataset_metadata.csv"
CUB_200_2011_PATH = "/mnt/home/ricardo.moreira/concept-based-explanations/cub_dataset"
IMAGES_PATH = os.path.join(CUB_200_2011_PATH, "CUB_200_2011/images")
BB_SCORE_COL = "is_Warbler_score"
LABEL_COL = "is_Warbler"
DEVICE = torch.device(f"cuda:{DEVICE_ID}")
LABEL_DIMS = 2

##################################
# Load datasets
##################################
dataset_configs = {
    "images_path": IMAGES_PATH,
    "metadata_df": pd.read_csv(METADATA_PATH),
    "label_col": LABEL_COL,
    "concept_names": CONCEPT_NAMES,
    "blackbox_score_col": BB_SCORE_COL,
}
VAL_DATASET = CUBDataset(
    train_val_test="val", data_augmentation=False, soft_labels=False, **dataset_configs
)
TEST_DATASET = CUBDataset(
    train_val_test="test", data_augmentation=False, soft_labels=False, **dataset_configs
)

##################################
# Define Dataloaders.
##################################
VAL_DATALOADER = DataLoader(
    VAL_DATASET,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    prefetch_factor=3,
    pin_memory=True,
)
TEST_DATALOADER = DataLoader(
    TEST_DATASET,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    prefetch_factor=3,
    pin_memory=True,
)


##################################
# Train and evaluate loop
##################################
def train_and_evaluate(**configs):
    train_dataset = CUBDataset(
        train_val_test="train",
        data_augmentation=configs["data_augmentation"],
        soft_labels=configs["soft_labels"],
        **dataset_configs,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs["train_batch_size"],
        shuffle=True,
        num_workers=NUM_WORKERS,
        prefetch_factor=3,
        pin_memory=True,
    )

    ##################################
    # Define Baseline Model.
    ##################################
    if BASELINE == "distillation":
        model = ImageDistillationBaseline(
            distil_hidden_layers=configs["distil_hidden_layers"],
            distil_dropouts=configs["distil_dropouts"],
            distil_batch_norm=configs["distil_batch_norm"],
            device=DEVICE,
            seed=SEED,
            no_improvement_epochs=NO_IMPROVEMENT_EPOCHS,
        )
    elif BASELINE == "explainability":
        prior_model = ImageMultiLabelPriorModelNN(
            label_names=CONCEPT_NAMES,
            label_dims=LABEL_DIMS,
            label_hidden_layers=configs["concept_hidden_layers"],
            label_dropouts=configs["concept_dropouts"],
            label_batch_norm=configs["concept_batch_norm"],
            device=DEVICE,
        )
        model = ExplainabilityBaseline(
            concept_predictor=prior_model,
            device=DEVICE,
            no_improvement_epochs=NO_IMPROVEMENT_EPOCHS,
        )
    elif BASELINE == "independent_components":
        prior_model = ImageIndependentPriorModelNN(
            label_names=CONCEPT_NAMES,
            label_dims=LABEL_DIMS,
            label_hidden_layers=configs["concept_hidden_layers"],
            label_dropouts=configs["concept_dropouts"],
            label_batch_norm=configs["concept_batch_norm"],
            ind_comp_discriminator_hidden_layers=configs[
                "independent_discriminator_hidden_layers"
            ],
            ind_comp_discriminator_dropouts=configs[
                "independent_discriminator_dropouts"
            ],
            ind_comp_gamma=configs["independent_loss_gamma"],
            ind_comp_lr=configs["ind_comp_lr"],
            train_independence_every=configs["train_independence_every"],
            device=DEVICE,
        )
        model = IndependentComponentsBaseline(
            concept_predictor=prior_model,
            no_improvement_epochs=NO_IMPROVEMENT_EPOCHS,
            local_hidden_layers=configs["local_hidden_layers"],
            local_dropouts=configs["local_dropouts"],
            local_batch_norm=configs["local_batch_norm"],
            with_biases=configs["with_biases"],
            requires_target_for_distillation=configs["uses_target_for_distillation"],
        )
    else:
        raise ValueError(f"Invalid config PRIOR_MODEL ({BASELINE})")

    optimizer = OPTIMIZER_CLS(
        model.parameters(),
        lr=configs["learning_rate"],
        weight_decay=configs["l2_decay"],
    )

    model.set_training_parameters(
        {
            "optimizer": optimizer,
            "n_batches": min(configs["up_to_batch"], len(train_dataloader))
            if configs["up_to_batch"]
            else len(train_dataloader),
            "batch_size": train_dataloader.batch_size,
            "cat_feat_embedding": configs["cat_features_embeddings"],
            "data_augmentation": configs["data_augmentation"],
            "soft_labels": configs["soft_labels"],
        }
    )

    base_path = os.path.join(os.path.abspath(SAVE_BASE_PATH), EXPERIMENT_NAME)
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    save_path = os.path.join(base_path, model.model_id + ".pkl")

    if os.path.exists(save_path):
        print(
            f"Skipping training of model with id: {model.model_id}."
            f"This model already exists!"
        )
    else:
        training_loop(
            trainable_concept_distil=model,
            train_dataloader=train_dataloader,
            val_dataloader=VAL_DATALOADER,
            test_dataloader=TEST_DATALOADER,
        )
        model.save(save_path)


def run_experiment():
    parameter_grid = create_parameter_grid(
        config_path=CONFIG_PATH, n_trials=N_TRIALS, seed=SEED
    )

    for parameter_config in tqdm(
        parameter_grid, total=N_TRIALS, desc=f"Running experiment: {EXPERIMENT_NAME}"
    ):
        parameter_config = round_parameters(parameter_config)
        print("===================================================")
        print(f"Parameter configs:", json.dumps(parameter_config, indent=4))
        print("===================================================")
        train_and_evaluate(**parameter_config)


run_experiment()
