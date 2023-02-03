#!/usr/bin/python
import pandas as pd
import os
import json
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from causal_concept_distil.datasets import CUBDataset
from causal_concept_distil.utils import create_parameter_grid, round_parameters
from causal_concept_distil.black_box_nn import ImageBlackBoxNN
from causal_concept_distil.loops import *

##################################
# General configurations
##################################
DEVICE_ID = 0
EXPERIMENT_NAME = "black_box_nn_CUB"
NUM_WORKERS = 6
SEED = 73
N_TRIALS = 200
OPTIMIZER_CLS = torch.optim.Adam
EVAL_BATCH_SIZE = 200
SAVE_BASE_PATH = "causal_concept_distil_black_box_nn_CUB"
CONFIG_PATH = "black_box_nn_CUB_parameters_config.yaml"
NO_IMPROVEMENT_EPOCHS = 15
N_EPOCHS = 200

CUB_200_2011_PATH = "/mnt/home/ricardo.moreira/concept-based-explanations/cub_dataset"
IMAGES_PATH = os.path.join(CUB_200_2011_PATH, "CUB_200_2011/images")
METADATA_PATH = "cub_dataset_metadata.csv"
LABEL_COL = "is_Warbler"
CONCEPT_NAMES = [
    "is_bill_shape_all-purpose",
    "is_wing_color_black",
    "is_underparts_color_white",
    "is_breast_pattern_solid",
    "is_eye_color_black",
    "is_bill_length_shorter_than_head",
    "is_size_small_(5_-_9_in)",
    "is_shape_perching-like",
    "is_belly_pattern_solid",
    "is_bill_color_black",
]


DEVICE = torch.device(f"cuda:{DEVICE_ID}")

##################################
# Load datasets
##################################
dataset_configs = {
    "images_path": IMAGES_PATH,
    "metadata_df": pd.read_csv(METADATA_PATH),
    "label_col": LABEL_COL,
    "concept_names": CONCEPT_NAMES,
    "soft_labels": False,
    "blackbox_score_col": None,
}
VAL_DATASET = CUBDataset(
    train_val_test="val", data_augmentation=False, **dataset_configs
)
TEST_DATASET = CUBDataset(
    train_val_test="test", data_augmentation=False, **dataset_configs
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
)
TEST_DATALOADER = DataLoader(
    TEST_DATASET,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    prefetch_factor=3,
)


##################################
# Train and evaluate loop
##################################
def train_and_evaluate(**configs):
    train_dataset = CUBDataset(
        train_val_test="train",
        data_augmentation=configs["data_augmentation"],
        **dataset_configs,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs["train_batch_size"],
        shuffle=True,
        num_workers=NUM_WORKERS,
        prefetch_factor=3,
        drop_last=True,
    )

    ##################################
    # Create Black-box Model.
    ##################################
    model = ImageBlackBoxNN(device=DEVICE, no_improvement_epochs=NO_IMPROVEMENT_EPOCHS)

    optimizer = OPTIMIZER_CLS(
        filter(lambda p: p.requires_grad, model.parameters()),
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
            "cat_feat_embedding": None,
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
            n_epochs=N_EPOCHS,
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
