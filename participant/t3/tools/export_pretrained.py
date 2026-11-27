"""Export the two Walk These Ways TorchScript networks to portable NumPy weights."""

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    layers = {}
    for prefix, filename in (("adaptation", "adaptation_module_latest.jit"), ("actor", "body_latest.jit")):
        module = torch.jit.load(str(args.checkpoints / filename), map_location="cpu")
        for key, value in module.state_dict().items():
            index, param = key.split(".")
            suffix = "w" if param == "weight" else "b"
            layers[f"{prefix}_{int(index) // 2}_{suffix}"] = value.cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **layers)


if __name__ == "__main__":
    main()
