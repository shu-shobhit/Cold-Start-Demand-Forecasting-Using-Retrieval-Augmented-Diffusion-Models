"""Main experiment entrypoint for RATD forecasting runs.

This script wires together configuration loading, dataloader creation,
model construction, training, checkpoint loading, and evaluation for the
forecasting experiments described in the repository README.
"""

import argparse
import datetime
import json
import os

import torch
import yaml

from dataset_forecasting import get_dataloader
from main_model import RATD_Forecasting
from utils import evaluate, train


parser = argparse.ArgumentParser(description="RATD")
parser.add_argument("--config", type=str, default="base_forecasting.yaml")
parser.add_argument("--datatype", type=str, default="electricity")
parser.add_argument("--device", default="cuda:5", help="Device for Attack")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--target_dim", type=int, default=321)
parser.add_argument("--h_size", type=int, default=96)
parser.add_argument("--ref_size", type=int, default=168)

args = parser.parse_args()
print(args)

# The original research snapshot expects configs to live under the authors'
# absolute experiment directory. The script preserves that assumption.
path = "/data/0shared/liujingwei/RATD/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

# CLI flags overwrite a small subset of model settings so experiments can be
# launched without editing YAML files for each run.
config["model"]["is_unconditional"] = args.unconditional
config["diffusion"]["h_size"] = args.h_size
config["diffusion"]["ref_size"] = args.ref_size
print(json.dumps(config, indent=4))

# Each run writes outputs into a timestamped directory so that checkpoints,
# config snapshots, and generated forecasts stay grouped together.
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/forecasting_" + args.datatype + "_" + current_time + "/"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# The dataloader factory encapsulates the electricity-specific dataset setup
# used by this forecasting variant.
train_loader, valid_loader, test_loader = get_dataloader(
    device=args.device,
    batch_size=config["train"]["batch_size"],
)

# The forecasting model wraps the diffusion network together with all masking,
# conditioning, and reverse-sampling logic.
model = RATD_Forecasting(config, args.device, args.target_dim).to(args.device)

if args.modelfolder == "":
    # Train from scratch when no saved checkpoint folder is supplied.
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    # Otherwise load the requested checkpoint from the authors' save layout.
    model.load_state_dict(
        torch.load("/data/0shared/liujingwei/RATD/save/" + args.modelfolder + "/model.pth")
    )

# The target dimension is restored explicitly before evaluation because some
# experiments may sample subsets of features during training.
model.target_dim = args.target_dim
evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=1,
    mean_scaler=0,
    foldername=foldername,
)
